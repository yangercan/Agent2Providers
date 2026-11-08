//! WPS 灵犀（Lingxi Claw）提供商实现。
//!
//! ── 上游是什么 ────────────────────────────────────────────────
//! 灵犀只有一条对话通路，且不是标准 OpenAI：三步 + 会话/run 生命周期。
//!   1. POST /api/aioffice/v1/cowork/sessions               建会话 → session_id
//!   2. POST /api/public/v1/credits/runs                    领 run   → run_id（计费单位）
//!   3. POST /api/public/v1/agent-runtime/chat/completions  SSE 对话
//! 第 3 步的帧长这样（注意没有 [DONE]，正文在 choices[].text 而不是 delta.content）：
//!   data:{"code":"Success","usage":{…},"choices":[{"text","reasoning_content","tool_calls","finish_reason"}]}
//!
//! ── 为什么必须走 conversation 而不是 build_chat ────────────────
//! build_chat 只返回「请求计划」，HTTP 与流式翻译都交给网关；而网关的
//! UpstreamKind::Chat 翻译机认的是标准 OpenAI 的 delta.content 与 [DONE]。
//! 灵犀两者都不符合 → 只能 stateful: true + conversation()，由本插件跑完
//! 三步，把帧翻译成标准 OpenAI SSE 再推给网关。
//! （官方 19 家实测口径：chat 恒为 true，走会话通路的再叠 stateful: true。）
//!
//! ── 协议事实来源 ─────────────────────────────────────────────
//! github.com/KevinFX-code/magpie-wps-lingxi（MIT），其实现标注
//! 「灵犀 1.3.31 安装包取证」。本文件是它的 Rust 重写，逻辑逐条对应。
//!
//! ── 本家实现要点 ─────────────────────────────────────────────
//!   · 出网一律走 crate::http（控制面 20s 上限 / 流式只限建连与读空闲）；
//!   · 流式用 crate::response_stream::sse 装配：自带空闲守卫（上游僵死
//!     检测）、[DONE] 截断与完整错误链 —— 灵犀不发 [DONE]，守卫照常生效；
//!   · 提示词预算：上游按 token 限量，超了不报「太长」而是 400「AI服务异
//!     常」，所以按 UTF-8 字节预裁（320KB / 单条 48KB，与上游插件同参）；
//!   · 工具 schema 压平：上游只认最朴素的 JSON Schema，anyOf/oneOf/allOf/
//!     not/const/$ref/type 数组会让整轮 400 —— 先压平再发。

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

use agent2api_provider_protocol as proto;
use agent2api_provider_sdk::{ConversationSink, PluginCtx, Provider, ProviderMeta, SendFuture};
use serde_json::{json, Value};

use crate::http;
use crate::logging;
use crate::store::{pick, token_tail};

/// provider id：落盘契约，与 pack.json 的 providers[].id 逐字一致。
pub const ID: &str = "lingxi";

const BASE: &str = "https://lingxi.wps.cn";
const BASE_KDOCS: &str = "https://lingxi.kdocs.cn";
const SESSIONS_PATH: &str = "/api/aioffice/v1/cowork/sessions";
const RUNS_PATH: &str = "/api/public/v1/credits/runs";
const CHAT_PATH: &str = "/api/public/v1/agent-runtime/chat/completions";
const BALANCE_PATH: &str = "/api/public/v1/credits/balance";
const TASKS_PATH: &str = "/api/public/v1/tasks";
const PLANS_PATH: &str = "/api/aioffice/v1/sessions/plans";

const CLIENT_TYPE: &str = "lingxi_claw";
const CLIENT_LANG: &str = "zh-CN";
const DEFAULT_MODEL: &str = "deepseek-v4.1-flash";

/// session/run 按「账号 × 模型」复用 1 小时：一个 run 内多次调用只扣一次智点。
const RUN_TTL_MS: i64 = 60 * 60 * 1000;

const UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36";
const KSO_APP_NAME: &str = "x-kso-app-name=win-lingxi-desktop";
const KSO_APP_VERSION: &str = "x-kso-app-version=1.3.31";

/// 提示词预算（UTF-8 字节）。上游实测约 150K token 上限，超限整轮 400「AI服务异常」；
/// 320KB 是上游插件的保守值（中文约 175 token/KB，混合约 255 token/KB，留一半余量）。
const MAX_PROMPT_BYTES: usize = 320 * 1024;
/// 单条消息上限：超了掐头去尾而不是整条丢（一次大粘贴不该牺牲全部历史）。
const MAX_MESSAGE_BYTES: usize = 48 * 1024;

pub struct LingxiProvider;

// ────────────────────────── 账号与凭证 ──────────────────────────

/// 从 secret 里取出 Cookie 串（wps_sid + uid + 可选灰度区域）。
///
/// 只保留这三个键：kso_sid / rtk / lang 这类噪声不该发给上游。
fn cookie_of(secret: &Value) -> String {
    let sid = pick(secret, &["sid", "wps_sid"]);
    let uid = pick(secret, &["uid"]);
    let region = pick(secret, &["region", "copilot_server_branch"]);
    let mut out = format!("wps_sid={sid}");
    if !uid.is_empty() {
        out.push_str(&format!("; uid={uid}"));
    }
    if !region.is_empty() {
        out.push_str(&format!("; copilot_server_branch={region}"));
    }
    out
}

fn sid_of(secret: &Value) -> String {
    pick(secret, &["sid", "wps_sid"])
}

/// 稳定身份：有 uid 用 uid，否则用 sid 尾部 8 位（与上游插件同一兜底链）。
fn identity_of(secret: &Value) -> String {
    let uid = pick(secret, &["uid"]);
    if !uid.is_empty() {
        return format!("wps-{uid}");
    }
    format!("wps-{}", tail_n(&sid_of(secret), 8))
}

/// 取字符串末尾 n 个**字符**（不是字节 —— 按字节切会切进多字节字符中间）。
fn tail_n(value: &str, n: usize) -> String {
    let chars: Vec<char> = value.chars().collect();
    let start = chars.len().saturating_sub(n);
    chars[start..].iter().collect()
}

/// 解析用户粘贴的东西：整段 Cookie、wps_sid=... 或裸 sid。
/// 返回 (sid, uid, region)；sid 短于 16 视为误粘贴（与上游同判据）。
fn parse_pasted(raw: &str) -> Option<(String, String, String)> {
    let mut text = raw.trim().to_string();
    if text.is_empty() {
        return None;
    }
    if let Some(head) = text.get(..7) {
        if head.eq_ignore_ascii_case("cookie:") {
            text = text.get(7..).unwrap_or("").trim().to_string();
        }
    }
    text = text.trim_matches(|c| c == '"' || c == '\'').trim().to_string();

    let mut sid = String::new();
    let mut uid = String::new();
    let mut region = String::new();
    for part in text.split(';') {
        let Some(eq) = part.find('=') else { continue };
        if eq == 0 {
            continue;
        }
        let name = part[..eq].trim();
        let value = part[eq + 1..].trim();
        match name {
            "wps_sid" => sid = value.to_string(),
            "uid" => uid = value.to_string(),
            "copilot_server_branch" => region = value.to_string(),
            _ => {}
        }
    }
    if sid.is_empty() && !text.contains('=') && text.len() >= 16 {
        sid = text.clone();
    }
    if sid.len() < 16 {
        return None;
    }
    Some((sid, uid, region))
}

// ────────────────────────── 请求头 ──────────────────────────

/// 与客户端 app-server（1.3.31 取证）一致的头。
/// Content-Type 由 crate::http::post_json 统一加，这里不重复。
fn agent_headers(cookie: &str, base: &str, extra: Vec<(String, String)>) -> Vec<(String, String)> {
    let mut headers: Vec<(String, String)> = vec![
        ("Cookie".into(), client_cookie(cookie)),
        ("X-Client-Type".into(), CLIENT_TYPE.into()),
        ("X-Client-Lang".into(), CLIENT_LANG.into()),
        ("X-Cc-Agent-Type".into(), CLIENT_TYPE.into()),
        ("X-Request-Id".into(), request_id()),
        ("Referer".into(), format!("{base}/")),
        ("User-Agent".into(), UA.into()),
    ];
    // 灰度区域：只在账号 Cookie 带了 copilot_server_branch 时才加。
    if let Some(region) = cookie_region(cookie) {
        headers.push(("X-Cc-Region".into(), region));
    }
    headers.extend(extra);
    headers
}

/// GET 类（目录 / 余额 / 任务）用的精简头。
fn basic_headers(cookie: &str, base: &str) -> Vec<(String, String)> {
    vec![
        ("Cookie".into(), client_cookie(cookie)),
        ("User-Agent".into(), UA.into()),
        ("Referer".into(), format!("{base}/")),
        ("Accept".into(), "application/json".into()),
    ]
}

/// 客户端 main 进程会给所有 *.wps.cn 请求追加 x-kso-* 设备 Cookie；
/// 服务端判断「是不是桌面客户端」很可能看它 —— 照拼（不带需要 RSA 签名的那部分）。
fn client_cookie(cookie: &str) -> String {
    let device = format!("{KSO_APP_NAME}; {KSO_APP_VERSION}; x-kso-device-id={}", device_id());
    if cookie.is_empty() {
        device
    } else {
        format!("{cookie}; {device}")
    }
}

fn cookie_region(cookie: &str) -> Option<String> {
    for part in cookie.split(';') {
        if let Some(value) = part.trim().strip_prefix("copilot_server_branch=") {
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

fn request_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos:x}-{:x}", std::process::id())
}

fn header_map(headers: &[(String, String)]) -> reqwest::header::HeaderMap {
    let mut map = reqwest::header::HeaderMap::new();
    for (name, value) in headers {
        if let (Ok(name), Ok(value)) = (
            reqwest::header::HeaderName::from_bytes(name.as_bytes()),
            reqwest::header::HeaderValue::from_str(value),
        ) {
            map.insert(name, value);
        }
    }
    map
}

// ────────────────────────── 错误识别 ──────────────────────────

fn brief(value: &Value) -> String {
    value.to_string().chars().take(300).collect()
}

/// 错误文案取值链：hint → msg → message → error / error.message。
fn text_of(payload: &Value) -> Option<String> {
    for key in ["hint", "msg", "message"] {
        if let Some(text) = payload.get(key).and_then(Value::as_str) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    if let Some(inner) = payload.get("error") {
        if let Some(text) = inner.as_str() {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        if let Some(text) = inner.get("message").and_then(Value::as_str) {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// 信封错误：result 非 ok、code 非 Success，或存在 error 对象。
fn envelope_error(payload: &Value) -> Option<String> {
    if let Some(result) = payload.get("result").and_then(Value::as_str) {
        if !result.is_empty() && !result.eq_ignore_ascii_case("ok") {
            return Some(text_of(payload).unwrap_or_else(|| format!("result={result}")));
        }
    }
    if let Some(code) = payload.get("code").and_then(Value::as_str) {
        if !code.is_empty() && !code.eq_ignore_ascii_case("success") {
            return Some(text_of(payload).unwrap_or_else(|| format!("code={code}")));
        }
    }
    payload.get("error").and_then(text_of).filter(|message| !message.is_empty())
}

/// 内容审核拦截（请求级 400 + AuditFailed）。
/// 普通「审核」字样可能是服务故障，不足以判定拦截 —— 只认已报告的标识。
fn is_audit_error(value: &Value) -> bool {
    let text = value.to_string();
    text.contains("AuditFailed") || text.contains("audit_blocked") || text.contains("暂时无法回答此问题")
}

// ────────────────────────── 会话 / run 缓存 ──────────────────────────

#[derive(Clone)]
struct Run {
    base: String,
    session_id: String,
    run_id: String,
    at: i64,
}

fn runs() -> &'static Mutex<HashMap<String, Run>> {
    static RUNS: OnceLock<Mutex<HashMap<String, Run>>> = OnceLock::new();
    RUNS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn run_get(key: &str) -> Option<Run> {
    let guard = runs().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.get(key).cloned().filter(|run| http::now_ms() - run.at < RUN_TTL_MS)
}

fn run_put(key: &str, run: Run) {
    let mut guard = runs().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.insert(key.to_string(), run);
}

fn run_drop(key: &str) {
    let mut guard = runs().lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.remove(key);
}

/// 设备 id：本地常量派生（上游只要求稳定，不要求密码学强度）。
fn device_id() -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in b"agent2api-lingxi-device" {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

// ────────────────────────── HTTP 步骤 ──────────────────────────

/// 建会话（控制面：20 秒总超时）。
async fn create_session(
    proxy: Option<&proto::WireProxy>,
    gateway_proxy: &str,
    cookie: &str,
    base: &str,
) -> Result<String, String> {
    let client = http::control_client(proxy, gateway_proxy);
    let body = json!({
        "session_type": "cowork_client",
        "agent_version": 2,
        "env": {
            "device_id": device_id(),
            "device_name": "agent2api",
            "work_space_dir": std::env::temp_dir().to_string_lossy(),
        },
    });
    let (status, payload) =
        http::post_json(&client, &format!("{base}{SESSIONS_PATH}"), &agent_headers(cookie, base, vec![]), &body)
            .await?;
    if is_audit_error(&payload) {
        return Err(format!("灵犀内容审核拦截：{}", brief(&payload)));
    }
    if !(200..300).contains(&status) {
        let detail = envelope_error(&payload).unwrap_or_else(|| brief(&payload));
        return Err(format!("灵犀上游返回 {status}：{detail}"));
    }
    if let Some(message) = envelope_error(&payload) {
        return Err(format!("灵犀上游返回错误：{message}"));
    }
    let id = payload
        .pointer("/data/session_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if id.is_empty() {
        return Err("灵犀未返回 session_id".to_string());
    }
    Ok(id)
}

/// 领 run（控制面：20 秒总超时）。run 是计费单位：一个 run 内多次调用只扣一次智点。
async fn create_run(
    proxy: Option<&proto::WireProxy>,
    gateway_proxy: &str,
    cookie: &str,
    base: &str,
    session_id: &str,
    model: &str,
    title: &str,
) -> Result<String, String> {
    let client = http::control_client(proxy, gateway_proxy);
    let body = json!({
        "client_request_id": format!("{}", http::now_ms()),
        "session_id": session_id,
        "model_provider": "lingxi",
        "model_id": model,
        "agent_input": {
            "text": title,
            "files": [],
            "file_ids": [],
            "expect_model": model,
            "refs": [],
            "permission_mode": "default",
            "current_mode": "agent",
            "browser_plugin_enabled": false,
        },
    });
    let (status, payload) =
        http::post_json(&client, &format!("{base}{RUNS_PATH}"), &agent_headers(cookie, base, vec![]), &body)
            .await?;
    if is_audit_error(&payload) {
        return Err(format!("灵犀内容审核拦截：{}", brief(&payload)));
    }
    if !(200..300).contains(&status) {
        let detail = envelope_error(&payload).unwrap_or_else(|| brief(&payload));
        return Err(format!("灵犀上游返回 {status}：{detail}"));
    }
    if let Some(message) = envelope_error(&payload) {
        return Err(format!("灵犀上游返回错误：{message}"));
    }
    let run_id = payload
        .pointer("/data/run_id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if run_id.is_empty() {
        return Err("灵犀未返回 run_id".to_string());
    }
    Ok(run_id)
}

/// 取或建 session/run：先试 wps.cn（参考实现的实测平面），失败回落 kdocs 域名。
///
/// ── 审核错误为什么不回落第二台域名 ──────────────────────────
/// 审核拦截是内容问题，换域名重试没有意义，还会让「每台打一次」的排障
/// 证据变乱（与上游插件同口径）。
async fn establish(
    proxy: Option<&proto::WireProxy>,
    gateway_proxy: &str,
    cookie: &str,
    key: &str,
    model: &str,
    title: &str,
) -> Result<Run, String> {
    let mut last_error = String::new();
    for base in [BASE, BASE_KDOCS] {
        match create_session(proxy, gateway_proxy, cookie, base).await {
            Ok(session_id) => {
                match create_run(proxy, gateway_proxy, cookie, base, &session_id, model, title).await {
                    Ok(run_id) => {
                        let run = Run {
                            base: base.to_string(),
                            session_id,
                            run_id,
                            at: http::now_ms(),
                        };
                        run_put(key, run.clone());
                        return Ok(run);
                    }
                    // 审核错误不回落第二台域名：那是内容问题，换域名没有意义。
                    Err(error) => {
                        if error.contains("内容审核拦截") {
                            return Err(error);
                        }
                        last_error = error;
                    }
                }
            }
            Err(error) => {
                if error.contains("内容审核拦截") {
                    return Err(error);
                }
                last_error = error;
            }
        }
    }
    Err(if last_error.is_empty() {
        "WPS 灵犀未能建立会话".to_string()
    } else {
        last_error
    })
}

/// 打开对话 SSE 流（流式客户端：只限建连与读空闲，无响应体总超时）。
async fn send_chat(
    proxy: Option<&proto::WireProxy>,
    gateway_proxy: &str,
    cookie: &str,
    run: &Run,
    payload: &Value,
) -> Result<reqwest::Response, String> {
    let client = http::stream_client(proxy, gateway_proxy);
    let mut headers = agent_headers(cookie, &run.base, vec![]);
    // run 头与客户端一致：全小写键名。
    headers.push(("x-cc-session-id".into(), run.session_id.clone()));
    headers.push(("x-cc-run-id".into(), run.run_id.clone()));
    // 解构而不是借字段：error_body 被 unwrap_or_default 拿走后，
    // 再用 sse.error_head() 会撞 E0382（部分移动后再借用）。
    let crate::http::SseResponse { status, response, error_body } =
        http::post_stream(&client, &format!("{}{CHAT_PATH}", run.base), &headers, payload, "*/*").await?;
    if (200..300).contains(&status) {
        return response.ok_or_else(|| "灵犀上游没有返回响应体".to_string());
    }
    let text = error_body.unwrap_or_default();
    let parsed: Value = serde_json::from_str(&text).unwrap_or_else(|_| json!({ "message": text }));
    if is_audit_error(&parsed) {
        return Err(format!("灵犀内容审核拦截：{}", brief(&parsed)));
    }
    Err(format!("灵犀上游返回 {}：{}", status, text.chars().take(240).collect::<String>()))
}

/// 会话/run 失效（上游回收或重建过）——重建一次再试。
/// 判据照搬上游插件 looksLikeStaleRun。
fn looks_like_stale_run(message: &str) -> bool {
    let lower = message.to_lowercase();
    if lower.contains("no rows") || message.contains("会话不存在") || message.contains("数据不存在") {
        return true;
    }
    lower.contains("session") && lower.contains("not found")
}

// ──────────────── 帧翻译（LingxiTranslator 的 Rust 版）────────────────
//
// 上游帧与 OpenAI 的三个差异，全部在这里消化：
//   ① 正文在 choices[].text（不是 delta.content）；
//   ② 没有 [DONE]，收尾靠 choices[].finish_reason（没有则视为被截断）；
//   ③ usage.total_tokens 是「当前上下文大小」而非本轮用量，必须重算。

struct Translator {
    id: String,
    created: i64,
    model: String,
    buffer: String,
    started: bool,
    saw_finish: bool,
    finished: bool,
    finish_reason: String,
    usage: Option<Value>,
}

impl Translator {
    fn new(model: &str) -> Self {
        let created = http::now_ms() / 1000;
        Self {
            id: format!("chatcmpl-lingxi-{created}"),
            created,
            model: model.to_string(),
            buffer: String::new(),
            started: false,
            saw_finish: false,
            finished: false,
            finish_reason: String::new(),
            usage: None,
        }
    }

    fn frame(&self, value: Value) -> String {
        format!("data: {value}\n\n")
    }

    fn chunk(&self, delta: Value, finish_reason: Option<&str>) -> String {
        self.frame(json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
        }))
    }

    fn usage_frame(&self) -> Option<String> {
        self.usage.as_ref().map(|usage| {
            self.frame(json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [],
                "usage": usage,
            }))
        })
    }

    fn start(&mut self, out: &mut Vec<String>) {
        if self.started {
            return;
        }
        self.started = true;
        let frame = self.chunk(json!({ "role": "assistant", "content": "" }), None);
        out.push(frame);
    }

    /// 增量喂入上游字节（已解码），按行切分后逐帧翻译。
    fn push(&mut self, text: &str, out: &mut Vec<String>) -> Result<(), String> {
        self.buffer.push_str(text);
        while let Some(index) = self.buffer.find('\n') {
            let line = self.buffer[..index].trim_end_matches('\r').to_string();
            self.buffer = self.buffer[index + 1..].to_string();
            self.consume(&line, out)?;
            if self.finished {
                break;
            }
        }
        Ok(())
    }

    /// 上游流结束。没见过 finish 事件就是被截断 —— 报错，不假装成功收尾。
    fn finish(&mut self, out: &mut Vec<String>) -> Result<(), String> {
        if self.finished {
            return Ok(());
        }
        let rest = std::mem::take(&mut self.buffer);
        if !rest.trim().is_empty() {
            let line = rest.trim_end_matches('\r').to_string();
            self.consume(&line, out)?;
        }
        if self.finished {
            return Ok(());
        }
        if !self.saw_finish {
            return Err("灵犀上游流在完成事件之前结束（响应被截断）".to_string());
        }
        self.finished = true;
        if let Some(frame) = self.usage_frame() {
            out.push(frame);
        }
        out.push("data: [DONE]\n\n".to_string());
        Ok(())
    }

    fn consume(&mut self, raw: &str, out: &mut Vec<String>) -> Result<(), String> {
        if self.finished {
            return Ok(());
        }
        let line = raw.trim();
        if line.is_empty() || line.starts_with(':') || line.starts_with("event:") {
            return Ok(());
        }
        let payload = line.strip_prefix("data:").unwrap_or(line).trim();
        if payload.is_empty() {
            return Ok(());
        }
        if payload == "[DONE]" {
            self.saw_finish = true;
            return Ok(());
        }
        let Ok(frame) = serde_json::from_str::<Value>(payload) else {
            return Ok(());
        };
        self.consume_frame(&frame, out)
    }

    fn consume_frame(&mut self, frame: &Value, out: &mut Vec<String>) -> Result<(), String> {
        if is_audit_error(frame) {
            self.finished = true;
            return Err(format!(
                "灵犀内容审核拦截：{}",
                text_of(frame).unwrap_or_else(|| brief(frame))
            ));
        }
        if let Some(code) = frame.get("code").and_then(Value::as_str) {
            if !code.is_empty() && !code.eq_ignore_ascii_case("success") {
                self.finished = true;
                return Err(format!(
                    "灵犀上游返回 {code}：{}",
                    text_of(frame).unwrap_or_else(|| brief(frame))
                ));
            }
        }
        if let Some(usage) = normalize_usage(frame.get("usage")) {
            self.usage = Some(usage);
        }

        let choices: Vec<Value> = frame
            .get("choices")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for choice in choices.iter() {
            let text = choice.get("text").and_then(Value::as_str).unwrap_or("");
            let reasoning = choice.get("reasoning_content").and_then(Value::as_str).unwrap_or("");
            let calls: Vec<Value> = choice
                .get("tool_calls")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();

            if !text.is_empty() || !reasoning.is_empty() || !calls.is_empty() {
                self.start(out);
                let mut delta = serde_json::Map::new();
                if !text.is_empty() {
                    delta.insert("content".into(), Value::String(text.to_string()));
                }
                if !reasoning.is_empty() {
                    delta.insert("reasoning_content".into(), Value::String(reasoning.to_string()));
                }
                if !calls.is_empty() {
                    delta.insert("tool_calls".into(), map_tool_calls(&calls));
                }
                let frame = self.chunk(Value::Object(delta), None);
                out.push(frame);
            }

            if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                let trimmed = reason.trim();
                if !trimmed.is_empty() {
                    self.saw_finish = true;
                    self.finish_reason = trimmed.to_string();
                }
            }
        }

        if self.saw_finish && !self.finished {
            self.finished = true;
            let reason = if self.finish_reason.is_empty() { "stop" } else { self.finish_reason.as_str() };
            let frame = self.chunk(json!({}), Some(reason));
            out.push(frame);
            if let Some(usage) = self.usage_frame() {
                out.push(usage);
            }
            out.push("data: [DONE]\n\n".to_string());
        }
        Ok(())
    }
}

fn map_tool_calls(calls: &[Value]) -> Value {
    let mapped: Vec<Value> = calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let item_index = call.get("index").and_then(Value::as_i64).unwrap_or(index as i64);
            let mut item = serde_json::Map::new();
            item.insert("index".into(), json!(item_index));
            // id / type / name / arguments 只在非空时写：工具调用分帧到达
            //（name 先来、arguments 后续拼接），空串帧不能覆盖前帧的内容。
            if let Some(id) = call.get("id").and_then(Value::as_str).filter(|id| !id.is_empty()) {
                item.insert("id".into(), Value::String(id.to_string()));
                item.insert("type".into(), Value::String("function".into()));
            }
            let mut function = serde_json::Map::new();
            if let Some(name) = call.pointer("/function/name").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                function.insert("name".into(), Value::String(name.to_string()));
            }
            if let Some(args) = call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
            {
                function.insert("arguments".into(), Value::String(args.to_string()));
            }
            item.insert("function".into(), Value::Object(function));
            Value::Object(item)
        })
        .collect();
    Value::Array(mapped)
}

/// 上游 usage 的 total_tokens 是「当前上下文大小」而非本轮用量
///（实测 42240 cache + 1097 prompt + 226 completion = 43563）。原样透传会让
/// 客户端把它当每轮用量累加，会话总量虚高几十倍 —— 所以按 prompt+completion 重算。
fn normalize_usage(usage: Option<&Value>) -> Option<Value> {
    let usage = usage?;
    if !usage.is_object() {
        return None;
    }
    let prompt = usage.get("prompt_tokens").and_then(Value::as_i64).unwrap_or(0);
    let completion = usage.get("completion_tokens").and_then(Value::as_i64).unwrap_or(0);
    let mut out = serde_json::Map::new();
    out.insert("prompt_tokens".into(), json!(prompt));
    out.insert("completion_tokens".into(), json!(completion));
    out.insert("total_tokens".into(), json!(prompt + completion));

    let cached = usage
        .get("cached_tokens")
        .or_else(|| usage.pointer("/prompt_tokens_details/cached_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    if cached > 0 {
        out.insert("prompt_tokens_details".into(), json!({ "cached_tokens": cached }));
    }
    let reasoning = usage
        .get("reasoning_tokens")
        .or_else(|| usage.pointer("/completion_tokens_details/reasoning_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    if reasoning > 0 {
        out.insert("completion_tokens_details".into(), json!({ "reasoning_tokens": reasoning }));
    }
    Some(Value::Object(out))
}

// ────────────────────────── 工具 schema 压平 ──────────────────────────
//
// 上游的 schema 转换器只认最朴素的 JSON Schema（上游插件逐条实测）：
// anyOf / oneOf / allOf / not / const / $ref 会让对话直接回 AIError（AI服务异常），
// type 用数组则回 400 InvalidArgument。这里把它们压平成上游认的形状：
// 宁可参数变宽，也别让整轮对话失败。

fn schema_rank(value: &Value) -> usize {
    if value.get("properties").map(Value::is_object).unwrap_or(false) {
        return 3;
    }
    if value.get("items").is_some() {
        return 2;
    }
    if value.get("type").is_some() {
        return 1;
    }
    0
}

fn merge_objects(base: &mut serde_json::Map<String, Value>, overlay: &serde_json::Map<String, Value>) {
    for (key, value) in overlay {
        match (base.get_mut(key), value) {
            (Some(Value::Object(existing)), Value::Object(add)) => merge_objects(existing, add),
            (Some(Value::Array(existing)), Value::Array(add)) if key == "required" => {
                for item in add {
                    if !existing.contains(item) {
                        existing.push(item.clone());
                    }
                }
            }
            _ => {
                base.insert(key.clone(), value.clone());
            }
        }
    }
}

fn resolve_ref(root: &Value, reference: &str, seen: &mut HashSet<String>, depth: usize) -> Value {
    let Some(path) = reference.strip_prefix("#/") else {
        return json!({ "type": "string" });
    };
    // 环防护：同一引用只解析一次。
    if !seen.insert(reference.to_string()) {
        return json!({ "type": "string" });
    }
    let mut node = root.clone();
    let mut resolved = true;
    for part in path.split('/') {
        let key = part.replace("~1", "/").replace("~0", "~");
        match node.get(key.as_str()) {
            Some(next) => node = next.clone(),
            None => {
                resolved = false;
                break;
            }
        }
    }
    let result = if resolved {
        sanitize_node(&node, root, seen, depth + 1)
    } else {
        json!({ "type": "string" })
    };
    seen.remove(reference);
    result
}

fn sanitize_node(value: &Value, root: &Value, seen: &mut HashSet<String>, depth: usize) -> Value {
    if depth > 24 {
        // 深度兜底：宁可参数变宽，也不要在恶意/病态 schema 上栈溢出。
        return json!({ "type": "string" });
    }
    let Some(object) = value.as_object() else {
        return value.clone();
    };

    // $ref → 内联展开
    if let Some(reference) = object.get("$ref").and_then(Value::as_str) {
        return resolve_ref(root, reference, seen, depth);
    }

    // allOf → 逐支清洗后合并，再叠自身其余键
    if let Some(branches) = object.get("allOf").and_then(Value::as_array) {
        let mut merged = serde_json::Map::new();
        for branch in branches {
            let sanitized = sanitize_node(branch, root, seen, depth + 1);
            if let Some(branch_object) = sanitized.as_object() {
                merge_objects(&mut merged, branch_object);
            }
        }
        for (key, val) in object {
            if key == "allOf" {
                continue;
            }
            let single = sanitize_node(&json!({ key.clone(): val.clone() }), root, seen, depth + 1);
            if let Some(single_object) = single.as_object() {
                merge_objects(&mut merged, single_object);
            }
        }
        if !merged.contains_key("type") {
            merged.insert("type".into(), json!("object"));
        }
        return Value::Object(merged);
    }

    // anyOf / oneOf → 选信息量最大的一支（object > array > 其他）
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = object.get(key).and_then(Value::as_array) {
            if let Some(best) = branches.iter().max_by_key(|branch| schema_rank(branch)) {
                let mut merged = sanitize_node(best, root, seen, depth + 1);
                for desc_key in ["description", "title"] {
                    if let Some(text) = object.get(desc_key) {
                        if let Some(target) = merged.as_object_mut() {
                            target.entry(desc_key.to_string()).or_insert_with(|| text.clone());
                        }
                    }
                }
                return merged;
            }
        }
    }

    let mut out = serde_json::Map::new();
    for (key, val) in object {
        match key.as_str() {
            // not / $defs / definitions：上游不认，直接丢
            "not" | "$defs" | "definitions" | "$ref" | "allOf" | "anyOf" | "oneOf" => {}
            // const → enum:[value]
            "const" => {
                out.insert("enum".into(), json!([val]));
            }
            "type" => {
                if let Some(types) = val.as_array() {
                    // type 数组：去掉 null 取单值
                    let picked = types
                        .iter()
                        .find(|item| item.as_str() != Some("null"))
                        .cloned()
                        .unwrap_or_else(|| json!("string"));
                    out.insert("type".into(), picked);
                } else {
                    out.insert("type".into(), val.clone());
                }
            }
            "properties" => {
                if let Some(props) = val.as_object() {
                    let mut next = serde_json::Map::new();
                    for (name, schema) in props {
                        next.insert(name.clone(), sanitize_node(schema, root, seen, depth + 1));
                    }
                    out.insert("properties".into(), Value::Object(next));
                }
            }
            "items" => match val {
                Value::Object(_) => {
                    out.insert("items".into(), sanitize_node(val, root, seen, depth + 1));
                }
                // 元组 schema：上游只认单 schema，取第一支
                Value::Array(list) => {
                    if let Some(first) = list.first() {
                        out.insert("items".into(), sanitize_node(first, root, seen, depth + 1));
                    }
                }
                _ => {}
            },
            "additionalProperties" => {
                if val.is_object() {
                    out.insert("additionalProperties".into(), sanitize_node(val, root, seen, depth + 1));
                } else if val.is_boolean() {
                    out.insert("additionalProperties".into(), val.clone());
                }
            }
            _ => {
                out.insert(key.clone(), val.clone());
            }
        }
    }

    // 强制补 type：上游转换器假定它存在
    if !out.contains_key("type") {
        let inferred = if out.contains_key("properties") {
            json!("object")
        } else if out.contains_key("items") {
            json!("array")
        } else {
            json!("string")
        };
        out.insert("type".into(), inferred);
    }
    Value::Object(out)
}

fn sanitize_schema(parameters: &Value) -> Value {
    let mut seen = HashSet::new();
    sanitize_node(parameters, parameters, &mut seen, 0)
}

/// 只保留上游认的工具：type === "function" 且 name 非空；其余丢弃。
fn filter_tools(tools: Option<&Value>) -> (Vec<Value>, usize) {
    let empty = Vec::new();
    let list = tools.and_then(Value::as_array).unwrap_or(&empty);
    let mut kept = Vec::new();
    let mut dropped = 0usize;
    for item in list {
        let name = item.pointer("/function/name").and_then(Value::as_str).unwrap_or("").trim();
        let is_function = item.get("type").and_then(Value::as_str) == Some("function");
        if !is_function || name.is_empty() {
            dropped += 1;
            continue;
        }
        let mut next = item.clone();
        if let Some(parameters) = item.pointer("/function/parameters") {
            if parameters.is_object() {
                if let Some(function) = next.pointer_mut("/function").and_then(Value::as_object_mut) {
                    function.insert("parameters".into(), sanitize_schema(parameters));
                }
            }
        }
        kept.push(next);
    }
    (kept, dropped)
}

// ────────────────────────── 提示词预算 ──────────────────────────

/// 按字节切字符串，并对齐 UTF-8 字符边界（按字符切会超限 3 倍，按字节切会切进
/// 多字节字符中间 —— 两头都要防）。
fn slice_bytes(text: &str, max_bytes: usize, from_end: bool) -> String {
    if max_bytes == 0 || text.is_empty() {
        return String::new();
    }
    if text.len() <= max_bytes {
        return text.to_string();
    }
    if from_end {
        let mut start = text.len().saturating_sub(max_bytes);
        while start < text.len() && !text.is_char_boundary(start) {
            start += 1;
        }
        text[start..].to_string()
    } else {
        let mut end = max_bytes;
        while end > 0 && !text.is_char_boundary(end) {
            end -= 1;
        }
        text[..end].to_string()
    }
}

/// 掐头去尾（头 60% + 尾 40%），中间留标记说明原长度。
fn head_tail_bytes(text: &str, limit: usize) -> String {
    let head = limit * 6 / 10;
    let tail = limit.saturating_sub(head);
    format!(
        "{}\n…[本条消息过长，已截断（原 {} 字节）]…\n{}",
        slice_bytes(text, head, false),
        text.len(),
        slice_bytes(text, tail, true)
    )
}

/// 单条消息超限时掐头去尾，返回新消息；没超限返回 None（调用方据此计数）。
fn elide_message(message: &Value) -> Option<Value> {
    let content = message.get("content")?;
    match content {
        Value::String(text) => {
            if text.len() <= MAX_MESSAGE_BYTES {
                return None;
            }
            let mut next = message.clone();
            if let Some(object) = next.as_object_mut() {
                object.insert("content".into(), Value::String(head_tail_bytes(text, MAX_MESSAGE_BYTES)));
            }
            Some(next)
        }
        Value::Array(parts) => {
            let mut changed = false;
            let mut next_parts = Vec::with_capacity(parts.len());
            for part in parts {
                let text = part.get("text").and_then(Value::as_str).unwrap_or("");
                if !text.is_empty() && text.len() > MAX_MESSAGE_BYTES {
                    let mut next_part = part.clone();
                    if let Some(object) = next_part.as_object_mut() {
                        object.insert("text".into(), Value::String(head_tail_bytes(text, MAX_MESSAGE_BYTES)));
                    }
                    next_parts.push(next_part);
                    changed = true;
                } else {
                    next_parts.push(part.clone());
                }
            }
            if !changed {
                return None;
            }
            let mut next = message.clone();
            if let Some(object) = next.as_object_mut() {
                object.insert("content".into(), Value::Array(next_parts));
            }
            Some(next)
        }
        _ => None,
    }
}

fn serialized_len(value: &Value) -> usize {
    serde_json::to_string(value).map(|text| text.len()).unwrap_or(0)
}

fn role_of(message: &Value) -> &str {
    message.get("role").and_then(Value::as_str).unwrap_or("")
}

/// 提示词预算（上游插件 fitAgentBody 的逐条移植）：
///   1. 掐长条：单条超限 → 掐头去尾留标记，不整条丢；
///   2. 丢历史：还超就从最老的一端整条丢，开头连续的 system 一律保留
///      （工具协议在里面）；
///   3. 保配对：丢完若头一条是 tool 结果，继续丢 —— tool_calls 与结果必须
///      成对，孤儿结果会被上游拒；
///   4. 留标记：插一条 system 说明历史不完整；标记自身字节也计入预算。
/// 返回 (丢弃条数, 截断条数)。
fn fit_agent_messages(body: &mut Value) -> (usize, usize) {
    let count = body.get("messages").and_then(Value::as_array).map(Vec::len).unwrap_or(0);
    if count <= 1 {
        return (0, 0);
    }
    let mut without = body.clone();
    if let Some(object) = without.as_object_mut() {
        object.remove("messages");
    }
    let fixed = serialized_len(&without);
    let size_of = |list: &[Value]| -> usize {
        fixed + list.iter().map(serialized_len).sum::<usize>()
    };

    let mut messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    // 步骤 1：掐长条
    let mut elided = 0usize;
    if size_of(&messages) > MAX_PROMPT_BYTES {
        let mut next = Vec::with_capacity(messages.len());
        for message in messages.iter() {
            match elide_message(message) {
                Some(replacement) => {
                    elided += 1;
                    next.push(replacement);
                }
                None => next.push(message.clone()),
            }
        }
        if elided > 0 {
            messages = next;
        }
    }
    if size_of(&messages) <= MAX_PROMPT_BYTES {
        if let Some(object) = body.as_object_mut() {
            object.insert("messages".into(), Value::Array(messages));
        }
        return (0, elided);
    }

    // 步骤 2：丢最老（保留开头的 system）
    let len = messages.len();
    let sizes: Vec<usize> = messages.iter().map(serialized_len).collect();
    let mut total = size_of(&messages);
    let mut head = 0usize;
    while head < len - 1 && role_of(&messages[head]) == "system" {
        head += 1;
    }
    let mut start = head;
    while total > MAX_PROMPT_BYTES && start < len - 1 {
        total -= sizes.get(start).copied().unwrap_or(0);
        start += 1;
    }
    // 步骤 3：不留孤儿 tool 结果
    while start < len - 1 && role_of(&messages[start]) == "tool" {
        total -= sizes.get(start).copied().unwrap_or(0);
        start += 1;
    }

    // 步骤 4：插标记；标记自身也占字节，装不下就再多丢一条
    let note = |count: usize| {
        json!({
            "role": "system",
            "content": format!(
                "[历史已压缩：更早的 {count} 条消息被省略（上游单次请求长度上限 {}KB）]",
                MAX_PROMPT_BYTES / 1024
            ),
        })
    };
    while start > head {
        let mut candidate: Vec<Value> = messages[..head].to_vec();
        candidate.push(note(start - head));
        candidate.extend(messages[start..].iter().cloned());
        if size_of(&candidate) <= MAX_PROMPT_BYTES {
            messages = candidate;
            break;
        }
        if start >= len - 1 {
            break;
        }
        start += 1;
        while start < len - 1 && role_of(&messages[start]) == "tool" {
            start += 1;
        }
    }
    let dropped = len - messages.len();
    if let Some(object) = body.as_object_mut() {
        object.insert("messages".into(), Value::Array(messages));
    }
    (dropped, elided)
}

// ────────────────────────── 请求体组装 ──────────────────────────

/// agent-runtime 的请求体。messages 原生透传；max_tokens 要包进
/// base_llm_arguments；reasoning_effort 顶层与 llm_arguments 各发一份。
fn build_agent_body(body: &Value, model: &str) -> Value {
    let mut out = serde_json::Map::new();
    out.insert("stream".into(), Value::Bool(true));
    out.insert("model".into(), Value::String(model.to_string()));
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    out.insert("messages".into(), Value::Array(messages));

    let (kept, dropped) = filter_tools(body.get("tools"));
    if dropped > 0 {
        logging::warn("[Lingxi]", &format!("丢弃 {dropped} 个上游不认的工具（只保留 function 型）"));
    }
    if !kept.is_empty() {
        out.insert("tools".into(), Value::Array(kept));
        if let Some(choice) = body.get("tool_choice") {
            if !choice.is_null() {
                out.insert("tool_choice".into(), choice.clone());
            }
        }
    }
    if let Some(temperature) = body.get("temperature").and_then(Value::as_f64) {
        out.insert("temperature".into(), json!(temperature));
    }
    if let Some(max_tokens) = body.get("max_tokens").and_then(Value::as_i64) {
        if max_tokens > 0 {
            out.insert("base_llm_arguments".into(), json!({ "max_tokens": max_tokens }));
        }
    }
    let effort = body
        .get("reasoning_effort")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if !effort.is_empty() {
        out.insert("reasoning_effort".into(), Value::String(effort.clone()));
        out.insert("llm_arguments".into(), json!({ "reasoning_effort": effort }));
    }

    let mut out = Value::Object(out);
    let (dropped_history, elided) = fit_agent_messages(&mut out);
    if dropped_history > 0 || elided > 0 {
        logging::warn(
            "[Lingxi]",
            &format!(
                "提示词超过 {}KB，已省略 {dropped_history} 条历史、截断 {elided} 条长消息",
                MAX_PROMPT_BYTES / 1024
            ),
        );
    }
    out
}

/// run 的 agent_input.text：取最后一条 user 消息（截断到 60 字符，与上游同参）。
fn last_user_text(body: &Value) -> String {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    for message in messages.iter().rev() {
        if message.get("role").and_then(Value::as_str) != Some("user") {
            continue;
        }
        let text = match message.get("content") {
            Some(Value::String(text)) => text.clone(),
            Some(Value::Array(parts)) => parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n"),
            _ => String::new(),
        };
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            return trimmed.chars().take(60).collect();
        }
    }
    String::new()
}

/// 上游只认已知的聊天模型；auto / plan_pro / plan_max 不是 credits/runs 模型。
fn is_chat_model(id: &str) -> bool {
    let id = id.trim();
    !id.is_empty() && !matches!(id, "auto" | "plan_pro" | "plan_max")
}

/// 去掉客户端可能带的前缀；空 / default 回落到默认模型。
fn wire_model(requested: &str) -> String {
    let mut model = requested.trim().to_string();
    if let Some(rest) = model.strip_prefix("wps-lingxi/") {
        model = rest.to_string();
    }
    if let Some(rest) = model.strip_prefix("lingxi/") {
        model = rest.to_string();
    }
    if let Some(rest) = model.strip_prefix("lingxi-") {
        model = rest.to_string();
    }
    if model.is_empty() || model == "default" {
        DEFAULT_MODEL.to_string()
    } else {
        model
    }
}

// ────────────────────────── 目录 / 余额 / 任务 ──────────────────────────

fn control_client(proxy: Option<&proto::WireProxy>, ctx: &PluginCtx) -> reqwest::Client {
    http::control_client(proxy, &ctx.env.proxy)
}

/// 模型目录：只读 sessions/plans 的 data.models（data.plans 是模式，不是模型）。
async fn remote_models(
    proxy: Option<&proto::WireProxy>,
    ctx: &PluginCtx,
    cookie: &str,
) -> Result<Vec<Value>, String> {
    let client = control_client(proxy, ctx);
    let (status, payload) =
        http::get_json(&client, &format!("{BASE}{PLANS_PATH}"), &basic_headers(cookie, BASE)).await?;
    if !(200..300).contains(&status) {
        return Err(format!("读取模型目录失败：上游返回 {status}"));
    }
    let models: Vec<Value> = payload
        .pointer("/data/models")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut list = Vec::new();
    for item in models.iter() {
        let id = item.get("key").and_then(Value::as_str).unwrap_or("").trim();
        if !is_chat_model(id) {
            continue;
        }
        let reasoning = item
            .get("reasoning_efforts")
            .and_then(Value::as_array)
            .map(|efforts| !efforts.is_empty())
            .unwrap_or(false);
        let mut record = json!({
            "id": id,
            "name": item.get("name").and_then(Value::as_str).unwrap_or(id),
            "context_length": item
                .get("max_context_tokens")
                .and_then(Value::as_i64)
                .unwrap_or(1_048_576),
            "supportsReasoning": reasoning,
        });
        if let Some(multiplier) = item.get("multiplier").and_then(Value::as_str) {
            if let Some(object) = record.as_object_mut() {
                object.insert("multiplier".into(), Value::String(multiplier.to_string()));
            }
        }
        if item.get("support_multimodal").and_then(Value::as_bool).unwrap_or(false) {
            if let Some(object) = record.as_object_mut() {
                object.insert("supportsImages".into(), Value::Bool(true));
            }
        }
        list.push(record);
    }
    Ok(list)
}

/// 剩余智点（签到用它算领前领后差）。拿不到返回 None —— 上游幂等假成功时
/// 这条链不能把签到判成失败。
async fn balance_total(
    proxy: Option<&proto::WireProxy>,
    ctx: &PluginCtx,
    cookie: &str,
) -> Option<f64> {
    let client = control_client(proxy, ctx);
    let (status, payload) =
        http::get_json(&client, &format!("{BASE}{BALANCE_PATH}"), &basic_headers(cookie, BASE))
            .await
            .ok()?;
    if !(200..300).contains(&status) {
        return None;
    }
    payload
        .pointer("/data/total_balance")
        .and_then(Value::as_f64)
        .or_else(|| payload.pointer("/data/balance").and_then(Value::as_f64))
}

/// 任务中心列表。
async fn fetch_tasks(
    proxy: Option<&proto::WireProxy>,
    ctx: &PluginCtx,
    cookie: &str,
) -> Result<Vec<Value>, String> {
    let client = control_client(proxy, ctx);
    let (status, payload) =
        http::get_json(&client, &format!("{BASE}{TASKS_PATH}"), &basic_headers(cookie, BASE)).await?;
    if status == 401 || status == 403 {
        return Err("WPS 登录态已失效，请重新登录".to_string());
    }
    if !(200..300).contains(&status) {
        return Err(format!("任务中心响应 HTTP {status}"));
    }
    if payload.get("result").and_then(Value::as_str) == Some("UserUnauthorized") {
        return Err("WPS 登录凭据失效，请重新登录".to_string());
    }
    Ok(payload
        .pointer("/data/tasks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default())
}

/// 领取任务。date 只在每日签到时给（上游按日期记账）。
async fn claim_task(
    proxy: Option<&proto::WireProxy>,
    ctx: &PluginCtx,
    cookie: &str,
    key: &str,
    date: &str,
) -> Result<Value, String> {
    let client = control_client(proxy, ctx);
    let url = if date.is_empty() {
        format!("{BASE}{TASKS_PATH}/{key}/claim")
    } else {
        format!("{BASE}{TASKS_PATH}/{key}/claim?date={date}")
    };
    let mut headers = basic_headers(cookie, BASE);
    headers.push(("Content-Type".into(), "application/json".into()));
    let (status, payload) = http::post_json(&client, &url, &headers, &json!({})).await?;
    if status == 401 || status == 403 {
        return Err("WPS 登录态已失效，请重新登录".to_string());
    }
    if !(200..300).contains(&status) {
        return Err(format!("签到失败：HTTP {status}"));
    }
    Ok(payload)
}

/// 领取时要带的日期：优先今天那一档，其次显式 available 的那档，最后今天。
/// （上游实际只会给 claimed / locked 两种状态，available 档是防御性的。）
fn claim_date(daily: &Value) -> String {
    let today = today_string();
    if let Some(days) = daily.pointer("/extra/check_in_days").and_then(Value::as_array) {
        let hit = days
            .iter()
            .find(|day| day.get("date").and_then(Value::as_str) == Some(today.as_str()));
        if let Some(date) = hit.and_then(|day| day.get("date")).and_then(Value::as_str) {
            return date.to_string();
        }
        let available = days.iter().find(|day| {
            day.get("status").and_then(Value::as_str) == Some("available")
                && day.get("date").and_then(Value::as_str).is_some()
        });
        if let Some(date) = available.and_then(|day| day.get("date")).and_then(Value::as_str) {
            return date.to_string();
        }
    }
    today
}

/// 本机本地日期 YYYY-MM-DD（灵犀按北京时间记账，这里按 UTC+8 折算）。
fn today_string() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    civil_from_days((seconds + 8 * 3600).div_euclid(86_400))
}

/// 「自 1970-01-01 起的天数」→ YYYY-MM-DD（Howard Hinnant 的 civil_from_days）。
fn civil_from_days(days: i64) -> String {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };
    format!("{year:04}-{month:02}-{day:02}")
}

fn format_number(value: f64) -> String {
    if value.fract().abs() < f64::EPSILON {
        format!("{}", value as i64)
    } else {
        format!("{value:.2}")
    }
}

/// 未登录时的真实模型快照（客户端 1.3.31 目录的逐条搬运）；不假装实时。
fn default_models() -> Vec<Value> {
    fn model(id: &str, name: &str, context: i64, reasoning: bool) -> Value {
        json!({ "id": id, "name": name, "context_length": context, "supportsReasoning": reasoning })
    }
    vec![
        model("deepseek-v4.1-flash", "DeepSeek V4.1 Flash", 1_048_576, true),
        model("deepseek-v4-flash-0731", "DeepSeek V4 Flash", 1_048_576, true),
        model("deepseek-v4-pro-0813", "DeepSeek V4 Pro", 1_048_576, true),
        model("kimi-k3", "Kimi-K3", 1_048_576, true),
        model("GLM-5.3", "GLM-5.3", 1_048_576, true),
        model("GLM-5.3-Flash", "GLM-5.3-Flash", 1_048_576, true),
        model("qwen3.8_max", "Qwen3.8-Max", 1_048_576, true),
        model("qwen3.8_flash", "Qwen3.8-Flash", 1_048_576, true),
        model("minimax_m3", "MiniMax-M3", 1_048_576, false),
        model("doubao_seed_2_1_pro_260628", "Doubao-Seed-2.1 Pro", 262_144, true),
        model("mimo-v2.6-pro", "Xiaomi MiMo-V2.6 Pro", 1_048_576, false),
    ]
}

// ────────────────────────── Provider 实现 ──────────────────────────

impl Provider for LingxiProvider {
    fn meta(&self) -> ProviderMeta {
        ProviderMeta::new(ID, "WPS 灵犀")
            .with_capabilities(proto::Capabilities {
                chat: true,
                // 三步通路 + 会话亲和（session/run 缓存）→ 走会话式转发
                stateful: true,
                // 客户端自己的目录可拉
                model_refresh: true,
                usage: true,
                checkin: true,
                // Cookie 登录态没有刷新接口，真判据是上游 401/403
                refresh: false,
                web_login: false,
                // 灵犀的模型名由 run 决定，不接受客户端任意模型名注入
                supports_default_model: false,
                ..Default::default()
            })
            .with_login(vec![proto::LoginMethod::Token {
                label: "Cookie / wps_sid".into(),
                placeholder: "粘贴整段 Cookie，或只粘贴 wps_sid 的值".into(),
                extra_fields: vec![],
            }])
            .with_models(default_models())
            .with_account_features(proto::AccountFeatures {
                usage: true,
                identifier: vec!["uid".to_string()],
                // Cookie 有效期上游不给读数，不申报 expiry（列显示空，不造假）
                expiry: vec![],
                ..Default::default()
            })
            .with_checkin(proto::CheckinInfo {
                description: Some("灵犀任务中心的每日签到（daily_check_in）".to_string()),
                daily: true,
                ..Default::default()
            })
    }

    fn ensure_token<'a>(&'a self, ctx: &'a PluginCtx, account: &'a str) -> SendFuture<'a, String> {
        Box::pin(async move {
            let found = ctx.store_get(ID, account).await.map_err(String::from)?;
            if sid_of(&found.secret).is_empty() {
                return Err("账号里没有 wps_sid，请重新登录".to_string());
            }
            Ok(cookie_of(&found.secret))
        })
    }

    // ── 账号添加 ──────────────────────────────────────────────
    fn normalize_account<'a>(
        &'a self,
        _ctx: &'a PluginCtx,
        input: proto::NormalizeInput,
    ) -> SendFuture<'a, proto::NormalizedAccount> {
        Box::pin(async move {
            let raw = match &input.source {
                proto::NormalizeSource::Manual { payload } => pick(
                    payload,
                    &["value", "cookie", "token", "sid", "wps_sid", "access"],
                ),
                proto::NormalizeSource::FileText { content, .. } => content.trim().to_string(),
                _ => return Err("这家只支持粘贴 Cookie / wps_sid".to_string()),
            };
            let (sid, uid, region) = parse_pasted(&raw).ok_or_else(|| {
                "没有在输入里找到 wps_sid（可粘贴整段 Cookie 或 wps_sid 的值）".to_string()
            })?;

            let secret = json!({ "sid": sid, "uid": uid, "region": region });
            let identity = identity_of(&secret);
            Ok(proto::NormalizedAccount {
                secret,
                public: json!({
                    "name": input.name.clone().unwrap_or_else(|| "WPS 灵犀".to_string()),
                    "uid": identity.clone(),
                    "tokenTail": token_tail(&sid),
                }),
                identity,
                desktop: false,
            })
        })
    }

    /// 稳定身份（去重用）。算法必须与 normalize_account 里算 identity 的逐字一致。
    fn account_identity(&self, secret: &Value) -> Result<String, String> {
        if sid_of(secret).is_empty() {
            return Err("缺少 wps_sid".to_string());
        }
        Ok(identity_of(secret))
    }

    /// 公开视图。**绝不能出现 Cookie 本体** —— 只给身份与尾号（安全边界）。
    fn account_view(&self, secret: &Value) -> Result<Value, String> {
        let sid = sid_of(secret);
        if sid.is_empty() {
            return Err("缺少 wps_sid".to_string());
        }
        Ok(json!({
            "uid": identity_of(secret),
            "tokenTail": token_tail(&sid),
            "hasSession": true,
            // 没有续期接口：canRefresh 恒 false（真判据是上游 401）
            "canRefresh": false,
        }))
    }

    // ── 模型目录 ──────────────────────────────────────────────
    fn models<'a>(&'a self, ctx: &'a PluginCtx, account: &'a str) -> SendFuture<'a, proto::ModelList> {
        Box::pin(async move {
            let found = ctx.store_get(ID, account).await.map_err(String::from)?;
            let cookie = cookie_of(&found.secret);
            if sid_of(&found.secret).is_empty() {
                // 没有账号也回答：内置快照兜底，不报错（目录是「未登录也有数」的）。
                return Ok(proto::ModelList {
                    models: default_models(),
                    meta: proto::CatalogMeta::default(),
                });
            }
            match remote_models(None, ctx, &cookie).await {
                Ok(list) if !list.is_empty() => Ok(proto::ModelList {
                    models: list,
                    meta: proto::CatalogMeta { remote: true, refreshed_at: http::now_ms() },
                }),
                // 拉不到就回内置快照，不假装实时（与上游插件同口径）。
                _ => Ok(proto::ModelList {
                    models: default_models(),
                    meta: proto::CatalogMeta::default(),
                }),
            }
        })
    }

    fn refresh_models<'a>(
        &'a self,
        ctx: &'a PluginCtx,
        account: &'a str,
        _force: bool,
    ) -> SendFuture<'a, proto::RefreshOutcome> {
        Box::pin(async move {
            let found = ctx.store_get(ID, account).await.map_err(String::from)?;
            if sid_of(&found.secret).is_empty() {
                // 没有账号不是失败：自动刷新路径按 unchanged 跳过，
                // 别把「没登录」刷成 CI 红色的错误。
                return Ok(proto::RefreshOutcome::unchanged());
            }
            let cookie = cookie_of(&found.secret);
            match remote_models(None, ctx, &cookie).await {
                Ok(list) if !list.is_empty() => Ok(proto::RefreshOutcome::refreshed(list.len())),
                Ok(_) => Ok(proto::RefreshOutcome::unchanged()),
                Err(message) => Ok(proto::RefreshOutcome::failed(message)),
            }
        })
    }

    // ── 智点余额 ──────────────────────────────────────────────
    fn usage<'a>(&'a self, ctx: &'a PluginCtx, account: &'a str) -> SendFuture<'a, proto::UsageReport> {
        Box::pin(async move {
            let found = ctx.store_get(ID, account).await.map_err(String::from)?;
            let cookie = cookie_of(&found.secret);
            if sid_of(&found.secret).is_empty() {
                return Err("账号里没有 wps_sid，请重新登录".to_string());
            }
            let client = control_client(None, ctx);
            let (status, payload) =
                http::get_json(&client, &format!("{BASE}{BALANCE_PATH}"), &basic_headers(&cookie, BASE))
                    .await?;
            if status == 401 || status == 403 {
                return Err("登录态失效，请重新登录".to_string());
            }
            if !(200..300).contains(&status) {
                return Err(format!("查询智点失败：上游返回 {status}"));
            }
            let data = payload.get("data").cloned().unwrap_or(Value::Null);
            let total = data
                .get("total_balance")
                .and_then(Value::as_f64)
                .or_else(|| data.get("balance").and_then(Value::as_f64))
                .unwrap_or(0.0);
            let granted = data
                .pointer("/bonus_summary/total_value")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let used = data.pointer("/bonus_summary/total_consumed").and_then(Value::as_f64);

            // used 必须是有限数字：钱包型账号（没有总额）给 0，
            // 否则界面判「剩余额度未知」（上游插件的实测教训）。
            let has_quota = granted > 0.0 && used.is_some();
            let used_value = used.unwrap_or(0.0);
            let mut window = serde_json::Map::new();
            window.insert("name".into(), Value::String("智点".into()));
            window.insert("used".into(), json!(used_value));
            window.insert("unit".into(), Value::String("credits".into()));
            if has_quota {
                window.insert("limit".into(), json!(granted));
            }
            let display = if has_quota {
                format!("{} / {} 智点", format_number(total), format_number(granted))
            } else {
                format!("{} 智点", format_number(total))
            };
            window.insert("display".into(), Value::String(display));

            Ok(json!({
                "plan": "WPS 灵犀",
                "user": found.public.get("name").and_then(Value::as_str).unwrap_or("WPS 账户"),
                "balance": format!("{} 智点", format_number(total)),
                "windows": [Value::Object(window)],
            }))
        })
    }

    // ── 每日签到 ──────────────────────────────────────────────
    // 判据照搬上游：不是看领取接口的返回值，而是看**领前领后的智点差**。
    // 上游重复领取是「幂等假成功」，直接返回成功会让界面显示假的奖励。
    fn checkin<'a>(&'a self, ctx: &'a PluginCtx, account: &'a str) -> SendFuture<'a, proto::CheckinReport> {
        Box::pin(async move {
            let found = ctx.store_get(ID, account).await.map_err(String::from)?;
            let cookie = cookie_of(&found.secret);
            if sid_of(&found.secret).is_empty() {
                return Err("账号里没有 wps_sid，请重新登录".to_string());
            }

            let tasks = fetch_tasks(None, ctx, &cookie).await?;
            let daily = tasks
                .into_iter()
                .find(|task| task.get("task_key").and_then(Value::as_str) == Some("daily_check_in"))
                .ok_or_else(|| "上游任务中心里没有每日签到".to_string())?;

            let streak = daily
                .pointer("/extra/consecutive_days")
                .and_then(Value::as_i64)
                .filter(|value| *value > 0)
                .map(|value| value as u32);

            if daily.get("status").and_then(Value::as_str) == Some("claimed") {
                return Ok(proto::CheckinReport {
                    outcome: proto::CheckinOutcome::Done,
                    credit: None,
                    streak,
                    message: Some("今日已签到".to_string()),
                });
            }

            let before = balance_total(None, ctx, &cookie).await;
            let payload = claim_task(None, ctx, &cookie, "daily_check_in", &claim_date(&daily)).await?;
            if let Some(result) = payload.get("result").and_then(Value::as_str) {
                if !result.is_empty() && !result.eq_ignore_ascii_case("ok") {
                    let hint = payload.get("hint").and_then(Value::as_str).unwrap_or(result);
                    return Err(format!("签到失败：{hint}"));
                }
            }
            let after = balance_total(None, ctx, &cookie).await;

            // 奖励 = 领前领后余额差；余额读不到时退回上游申报的奖励额。
            let delta = match (before, after) {
                (Some(before), Some(after)) => after - before,
                _ => payload
                    .pointer("/data/extra/today_reward")
                    .and_then(Value::as_f64)
                    .or_else(|| daily.get("reward_amount").and_then(Value::as_f64))
                    .unwrap_or(0.0),
            };

            if delta > 0.0 {
                Ok(proto::CheckinReport {
                    outcome: proto::CheckinOutcome::Claimed,
                    credit: Some(format!("+{} 智点", format_number(delta))),
                    streak: streak.map(|value| value + 1),
                    message: None,
                })
            } else {
                Ok(proto::CheckinReport {
                    outcome: proto::CheckinOutcome::Done,
                    credit: None,
                    streak,
                    message: Some("今日已签到（上游幂等，余额未变化）".to_string()),
                })
            }
        })
    }

    // ── 会话式转发（唯一的数据面通路）──────────────────────────
    //
    // 注意：本函数体内**不得**出现 control_client / .timeout() ——
    // build.rs 的流式闸门会扫这个函数（见 STREAM_FUNCTIONS）。
    fn conversation<'a>(
        &'a self,
        ctx: &'a PluginCtx,
        req: proto::ConversationRequest,
        mut sink: ConversationSink<'a>,
    ) -> SendFuture<'a, ()> {
        Box::pin(async move {
            let found = ctx.store_get(ID, &req.account).await.map_err(String::from)?;
            let cookie = cookie_of(&found.secret);
            if sid_of(&found.secret).is_empty() {
                return Err("账号里没有 wps_sid，请重新登录".to_string());
            }

            let requested = if req.wire_model.trim().is_empty() {
                req.body.get("model").and_then(Value::as_str).unwrap_or("").to_string()
            } else {
                req.wire_model.clone()
            };
            let model = wire_model(&requested);
            if !is_chat_model(&model) {
                return Err(format!("WPS 灵犀不支持此聊天模型：{model}"));
            }

            let key = format!("{}|{model}", identity_of(&found.secret));
            let payload = build_agent_body(&req.body, &model);
            let title = last_user_text(&req.body);

            // ① 取或建 session/run（按账号 × 模型缓存 1 小时）
            let run = match run_get(&key) {
                Some(run) => run,
                None => {
                    establish(req.proxy.as_ref(), &ctx.env.proxy, &cookie, &key, &model, &title).await?
                }
            };

            // ② 发对话请求；上游报会话失效时删缓存重建一次（只重试一次）
            let response = match send_chat(req.proxy.as_ref(), &ctx.env.proxy, &cookie, &run, &payload).await {
                Ok(response) => response,
                Err(error) => {
                    if !looks_like_stale_run(&error) {
                        return Err(error);
                    }
                    run_drop(&key);
                    logging::log("[Lingxi]", "会话已失效，重建 run 后重试一次");
                    let fresh =
                        establish(req.proxy.as_ref(), &ctx.env.proxy, &cookie, &key, &model, &title).await?;
                    send_chat(req.proxy.as_ref(), &ctx.env.proxy, &cookie, &fresh, &payload).await?
                }
            };

            // ③ 先报头：网关据此决定状态码与翻译机
            let mut head_headers = BTreeMap::new();
            head_headers.insert("content-type".to_string(), "text/event-stream".to_string());
            sink.head(proto::StreamHead {
                status: 200,
                headers: head_headers,
                upstream: proto::UpstreamKind::Chat,
            })
            .await?;

            // ④ 逐段翻译：上游帧 → 标准 OpenAI SSE。response_stream::sse 自带
            //    空闲守卫（上游僵死检测）与完整错误链；send 的 await 就是背压，
            //    不要把整段回答缓冲到自己内存里。
            let mut translator = Translator::new(&model);
            let mut pending: Vec<String> = Vec::new();
            let mut source = crate::response_stream::sse(response);
            use futures::StreamExt;
            while let Some(item) = source.next().await {
                match item {
                    Err(error) => {
                        // 传输中断 / 空闲超时：先把已翻译的帧吐完再报错。
                        for frame in pending.drain(..) {
                            let _ = sink.send(frame.as_bytes()).await;
                        }
                        return Err(format!("灵犀上游流中断：{error}"));
                    }
                    Ok(chunk) => {
                        let text = String::from_utf8_lossy(&chunk);
                        if let Err(error) = translator.push(&text, &mut pending) {
                            for frame in pending.drain(..) {
                                let _ = sink.send(frame.as_bytes()).await;
                            }
                            return Err(error);
                        }
                        for frame in pending.drain(..) {
                            if sink.send(frame.as_bytes()).await.is_err() {
                                // 客户端断开是正常事件：直接收尾，不当故障冒泡。
                                return Ok(());
                            }
                        }
                    }
                }
            }
            if let Err(error) = translator.finish(&mut pending) {
                for frame in pending.drain(..) {
                    let _ = sink.send(frame.as_bytes()).await;
                }
                return Err(error);
            }
            for frame in pending.drain(..) {
                if sink.send(frame.as_bytes()).await.is_err() {
                    return Ok(());
                }
            }
            Ok(())
        })
    }

    // ── 错误分类（只有四档，判错会改变网关的账号轮换行为）────────
    fn classify(&self, status: u16, body: &Value) -> proto::ErrorClass {
        let message = text_of(body)
            .or_else(|| body.pointer("/error/message").and_then(Value::as_str).map(str::to_string))
            .unwrap_or_else(|| brief(body));
        if is_audit_error(body) {
            // 内容审核：不罚账号，换中性提示词重试（与上游把审核映射成 400 的口径一致）。
            return proto::ErrorClass::ContentBlocked { status, message, upstream_code: None };
        }
        match status {
            // 灵犀把限额与业务失败都放在 200 信封里，HTTP 429 罕见；保守仍归限额。
            429 => proto::ErrorClass::QuotaLimited {
                reset_at: None,
                message,
                upstream_code: None,
                status,
            },
            401 | 403 => proto::ErrorClass::TokenExpired { message },
            _ => proto::ErrorClass::Fatal { status, message, upstream_code: None },
        }
    }
}
