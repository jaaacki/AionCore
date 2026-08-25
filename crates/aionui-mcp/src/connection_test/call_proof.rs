use std::collections::HashMap;
use std::ffi::OsString;
use std::time::Duration;

use aionui_api_types::{
    MCP_CALL_PROOF_MAX_ARGUMENT_BYTES, MCP_CALL_PROOF_MAX_NAME_BYTES, MCP_CALL_PROOF_MAX_RESPONSE_BYTES,
    McpCallProofErrorCode, McpCallProofResult,
};
use aionui_runtime::{
    Builder as CmdBuilder, RuntimeCommandProbe, ensure_runtime_command_with_reporter, kill_process_tree,
    probe_runtime_command, resolve_command_path,
};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;
use tracing::warn;

use crate::types::McpServerTransport;

use super::McpConnectionTestService;
use super::protocol::{
    JsonRpcRequest, JsonRpcResponse, PROTOCOL_VERSION, SseEvent, build_http_headers, build_initialize_request,
    build_initialized_notification, build_tools_list_request, read_sse_events_bounded,
};

const CALL_PROOF_TIMEOUT: Duration = Duration::from_secs(30);
const TOOL_CALL_REQUEST_ID: u64 = 3;

#[derive(Debug, Clone, Copy)]
pub(crate) enum CallProofStage {
    Validate,
    Spawn,
    Connect,
    Initialize,
    Initialized,
    ToolsList,
    ToolsCall,
    Endpoint,
    Cleanup,
}

impl CallProofStage {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Validate => "validate",
            Self::Spawn => "spawn",
            Self::Connect => "connect",
            Self::Initialize => "initialize",
            Self::Initialized => "initialized",
            Self::ToolsList => "tools_list",
            Self::ToolsCall => "tools_call",
            Self::Endpoint => "endpoint",
            Self::Cleanup => "cleanup",
        }
    }
}

#[derive(Debug)]
pub struct McpCallProofFailure {
    code: McpCallProofErrorCode,
    stage: CallProofStage,
    transport: &'static str,
    rpc_code: Option<i64>,
}

impl McpCallProofFailure {
    pub fn code(&self) -> McpCallProofErrorCode {
        self.code
    }

    pub fn message(&self) -> &'static str {
        match self.code {
            McpCallProofErrorCode::InvalidRequest => "Invalid MCP call proof request",
            McpCallProofErrorCode::ArgumentsTooLarge => "MCP tool arguments exceed the allowed size",
            McpCallProofErrorCode::ResponseTooLarge => "MCP server response exceeds the allowed size",
            McpCallProofErrorCode::CommandNotFound => "MCP server command was not found",
            McpCallProofErrorCode::CommandPermissionDenied => "MCP server command cannot be executed",
            McpCallProofErrorCode::CommandStartFailed => "MCP server command failed to start",
            McpCallProofErrorCode::ProcessCleanupFailed => "MCP transport cleanup failed",
            McpCallProofErrorCode::ConnectionFailed => "MCP server connection failed",
            McpCallProofErrorCode::HttpError => "MCP server returned an HTTP error",
            McpCallProofErrorCode::RedirectRejected => "MCP transport redirect was rejected",
            McpCallProofErrorCode::Timeout => "MCP call proof timed out",
            McpCallProofErrorCode::RpcError => "MCP server returned a JSON-RPC error",
            McpCallProofErrorCode::ProtocolError => "MCP server returned an invalid protocol response",
            McpCallProofErrorCode::ToolNotFound => "Requested MCP tool was not advertised",
            McpCallProofErrorCode::ToolNotReadOnly => "Requested MCP tool is not declared read-only",
            McpCallProofErrorCode::ToolCallFailed => "MCP tool call reported failure",
        }
    }

    pub fn details(&self) -> Value {
        let mut details = serde_json::Map::from_iter([
            ("stage".to_owned(), json!(self.stage.as_str())),
            ("transport".to_owned(), json!(self.transport)),
        ]);
        if let Some(code) = self.rpc_code {
            details.insert("rpc_code".to_owned(), json!(code));
        }
        Value::Object(details)
    }
}

type CallProofResult<T> = Result<T, McpCallProofFailure>;

impl McpConnectionTestService {
    pub async fn call_proof(
        &self,
        name: &str,
        transport: &McpServerTransport,
        tool: &str,
        arguments: Value,
        authenticated_user_id: &str,
        runtime_scope_id: Option<&str>,
    ) -> CallProofResult<McpCallProofResult> {
        let transport_type = transport_type(transport);
        let prepared = PreparedCall::new(
            name,
            tool,
            arguments,
            authenticated_user_id,
            runtime_scope_id,
            transport_type,
        )?;
        match transport {
            McpServerTransport::Stdio { command, args, env } => {
                self.call_proof_stdio(command, args, env, &prepared, authenticated_user_id, runtime_scope_id)
                    .await
            }
            McpServerTransport::Http { url, headers } => self.call_proof_http(url, headers, &prepared).await,
            McpServerTransport::Sse { url, headers } => self.call_proof_sse(url, headers, &prepared).await,
        }
    }

    async fn call_proof_stdio(
        &self,
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
        prepared: &PreparedCall,
        authenticated_user_id: &str,
        runtime_scope_id: Option<&str>,
    ) -> CallProofResult<McpCallProofResult> {
        let reporter = runtime_scope_id
            .map(|scope_id| self.runtime_reporter(Some(authenticated_user_id.to_owned()), scope_id.to_owned()));
        let mut cmd = match probe_runtime_command(command) {
            RuntimeCommandProbe::NodeTool { .. } => {
                let resolved = ensure_runtime_command_with_reporter(command, reporter.as_deref())
                    .await
                    .map_err(|error| spawn_error(command, &runtime_resolution_error(&error.to_string())))?;
                CmdBuilder::from_resolved(&resolved)
            }
            _ => CmdBuilder::new(resolve_stdio_command(command)),
        };
        cmd.args(args)
            .envs(env.iter())
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());

        let mut child = cmd.spawn().map_err(|error| spawn_error(command, &error))?;
        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let result = match tokio::time::timeout(
            self.timeout.min(CALL_PROOF_TIMEOUT),
            run_stdio_call_protocol(stdin, stdout, prepared),
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(failure(
                McpCallProofErrorCode::Timeout,
                CallProofStage::ToolsCall,
                "stdio",
            )),
        };
        if let Err(error) = kill_process_tree(&mut child).await {
            warn!(%error, "failed to clean up MCP stdio call-proof process tree");
            return Err(failure(
                McpCallProofErrorCode::ProcessCleanupFailed,
                CallProofStage::Cleanup,
                "stdio",
            ));
        }
        result
    }

    async fn call_proof_http(
        &self,
        url: &str,
        headers: &HashMap<String, String>,
        prepared: &PreparedCall,
    ) -> CallProofResult<McpCallProofResult> {
        tokio::time::timeout(
            self.timeout.min(CALL_PROOF_TIMEOUT),
            self.call_proof_http_inner(url, headers, prepared),
        )
        .await
        .map_err(|_| failure(McpCallProofErrorCode::Timeout, CallProofStage::ToolsCall, "http"))?
    }

    async fn call_proof_http_inner(
        &self,
        url: &str,
        headers: &HashMap<String, String>,
        prepared: &PreparedCall,
    ) -> CallProofResult<McpCallProofResult> {
        validate_transport_url(url, "http")?;
        let mut req_headers = build_http_headers(headers);
        req_headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse().unwrap());
        req_headers.insert(
            reqwest::header::ACCEPT,
            "application/json, text/event-stream".parse().unwrap(),
        );

        let init = self
            .proof_http_post(
                url,
                &req_headers,
                &build_initialize_request(1),
                CallProofStage::Initialize,
            )
            .await?;
        expect_response_id(&init.rpc, 1, CallProofStage::Initialize, "http")?;
        let protocol_version = validate_initialize(&init.rpc, "http")?;
        let session_id = init.session_id;
        if let Some(session_id) = session_id.as_deref() {
            let value = reqwest::header::HeaderValue::from_str(session_id)
                .map_err(|_| failure(McpCallProofErrorCode::ProtocolError, CallProofStage::Initialize, "http"))?;
            req_headers.insert("mcp-session-id", value);
        }
        let mut cleanup_guard = session_id
            .as_ref()
            .map(|_| HttpSessionCleanupGuard::new(self.proof_http_client.clone(), url.to_owned(), req_headers.clone()));

        let result = async {
            self.proof_http_notification(
                url,
                &req_headers,
                &build_initialized_notification(),
                CallProofStage::Initialized,
            )
            .await?;
            let tools = self
                .proof_http_post(
                    url,
                    &req_headers,
                    &build_tools_list_request(2),
                    CallProofStage::ToolsList,
                )
                .await?;
            expect_response_id(&tools.rpc, 2, CallProofStage::ToolsList, "http")?;
            let tools_value = validate_tools_list(&tools.rpc, &prepared.tool, "http")?;
            let call = self
                .proof_http_post(
                    url,
                    &req_headers,
                    &build_tools_call_request(TOOL_CALL_REQUEST_ID, &prepared.tool, prepared.arguments.clone()),
                    CallProofStage::ToolsCall,
                )
                .await?;
            expect_response_id(&call.rpc, TOOL_CALL_REQUEST_ID, CallProofStage::ToolsCall, "http")?;
            build_proof(
                prepared,
                protocol_version,
                tools_value,
                validate_tool_call(&call.rpc, "http")?,
            )
        }
        .await;

        if session_id.is_some() {
            self.cleanup_http_session(url, &req_headers).await;
            if let Some(guard) = cleanup_guard.as_mut() {
                guard.disarm();
            }
        }
        result
    }

    async fn proof_http_post<T: Serialize + ?Sized>(
        &self,
        url: &str,
        headers: &reqwest::header::HeaderMap,
        body: &T,
        stage: CallProofStage,
    ) -> CallProofResult<ProofHttpResponse> {
        let response = self
            .proof_http_client
            .post(url)
            .headers(headers.clone())
            .json(body)
            .send()
            .await
            .map_err(|_| failure(McpCallProofErrorCode::ConnectionFailed, stage, "http"))?;
        validate_http_status(&response, stage, "http")?;
        let session_id = response
            .headers()
            .get("mcp-session-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let rpc = parse_bounded_http_response(response, stage, "http").await?;
        Ok(ProofHttpResponse { rpc, session_id })
    }

    async fn proof_http_notification<T: Serialize + ?Sized>(
        &self,
        url: &str,
        headers: &reqwest::header::HeaderMap,
        body: &T,
        stage: CallProofStage,
    ) -> CallProofResult<()> {
        let response = self
            .proof_http_client
            .post(url)
            .headers(headers.clone())
            .json(body)
            .send()
            .await
            .map_err(|_| failure(McpCallProofErrorCode::ConnectionFailed, stage, "http"))?;
        validate_http_status(&response, stage, "http")
    }

    async fn cleanup_http_session(&self, url: &str, headers: &reqwest::header::HeaderMap) {
        const CLEANUP_TIMEOUT: Duration = Duration::from_secs(2);
        let cleanup = self.proof_http_client.delete(url).headers(headers.clone()).send();
        match tokio::time::timeout(CLEANUP_TIMEOUT, cleanup).await {
            Ok(Ok(response))
                if response.status().is_success() || response.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED => {}
            Ok(Ok(response)) => warn!(status = %response.status(), "MCP HTTP call-proof session cleanup was rejected"),
            Ok(Err(_)) => warn!("MCP HTTP call-proof session cleanup failed"),
            Err(_) => warn!("MCP HTTP call-proof session cleanup timed out"),
        }
    }

    async fn call_proof_sse(
        &self,
        url: &str,
        headers: &HashMap<String, String>,
        prepared: &PreparedCall,
    ) -> CallProofResult<McpCallProofResult> {
        tokio::time::timeout(
            self.timeout.min(CALL_PROOF_TIMEOUT),
            self.call_proof_sse_inner(url, headers, prepared),
        )
        .await
        .map_err(|_| failure(McpCallProofErrorCode::Timeout, CallProofStage::ToolsCall, "sse"))?
    }

    async fn call_proof_sse_inner(
        &self,
        url: &str,
        headers: &HashMap<String, String>,
        prepared: &PreparedCall,
    ) -> CallProofResult<McpCallProofResult> {
        let base = validate_transport_url(url, "sse")?;
        let mut req_headers = build_http_headers(headers);
        let response = self
            .proof_http_client
            .get(url)
            .headers(req_headers.clone())
            .header(reqwest::header::ACCEPT, "text/event-stream")
            .send()
            .await
            .map_err(|_| failure(McpCallProofErrorCode::ConnectionFailed, CallProofStage::Connect, "sse"))?;
        validate_http_status(&response, CallProofStage::Connect, "sse")?;
        let (event_tx, mut event_rx) = mpsc::channel::<Result<SseEvent, ()>>(16);
        let reader = tokio::spawn(read_sse_events_bounded(
            response,
            event_tx,
            MCP_CALL_PROOF_MAX_RESPONSE_BYTES,
        ));
        req_headers.insert(reqwest::header::CONTENT_TYPE, "application/json".parse().unwrap());

        let result = async {
            let endpoint = wait_for_bounded_endpoint(&mut event_rx, &base).await?;
            self.proof_sse_post(
                &endpoint,
                &req_headers,
                &build_initialize_request(1),
                CallProofStage::Initialize,
            )
            .await?;
            let init = wait_for_bounded_sse_response(&mut event_rx, CallProofStage::Initialize).await?;
            expect_response_id(&init, 1, CallProofStage::Initialize, "sse")?;
            let protocol_version = validate_initialize(&init, "sse")?;
            self.proof_sse_post(
                &endpoint,
                &req_headers,
                &build_initialized_notification(),
                CallProofStage::Initialized,
            )
            .await?;
            self.proof_sse_post(
                &endpoint,
                &req_headers,
                &build_tools_list_request(2),
                CallProofStage::ToolsList,
            )
            .await?;
            let tools = wait_for_bounded_sse_response(&mut event_rx, CallProofStage::ToolsList).await?;
            expect_response_id(&tools, 2, CallProofStage::ToolsList, "sse")?;
            let tools_value = validate_tools_list(&tools, &prepared.tool, "sse")?;
            self.proof_sse_post(
                &endpoint,
                &req_headers,
                &build_tools_call_request(TOOL_CALL_REQUEST_ID, &prepared.tool, prepared.arguments.clone()),
                CallProofStage::ToolsCall,
            )
            .await?;
            let call = wait_for_bounded_sse_response(&mut event_rx, CallProofStage::ToolsCall).await?;
            expect_response_id(&call, TOOL_CALL_REQUEST_ID, CallProofStage::ToolsCall, "sse")?;
            build_proof(
                prepared,
                protocol_version,
                tools_value,
                validate_tool_call(&call, "sse")?,
            )
        }
        .await;
        reader.abort();
        result
    }

    async fn proof_sse_post<T: Serialize + ?Sized>(
        &self,
        endpoint: &str,
        headers: &reqwest::header::HeaderMap,
        body: &T,
        stage: CallProofStage,
    ) -> CallProofResult<()> {
        let response = self
            .proof_http_client
            .post(endpoint)
            .headers(headers.clone())
            .json(body)
            .send()
            .await
            .map_err(|_| failure(McpCallProofErrorCode::ConnectionFailed, stage, "sse"))?;
        validate_http_status(&response, stage, "sse")
    }
}

#[derive(Debug)]
struct PreparedCall {
    server_name: String,
    tool: String,
    arguments: Value,
    arguments_bytes: Vec<u8>,
    authenticated_subject_sha256: String,
    runtime_scope_id: Option<String>,
}

impl PreparedCall {
    fn new(
        name: &str,
        tool: &str,
        arguments: Value,
        authenticated_user_id: &str,
        runtime_scope_id: Option<&str>,
        transport: &'static str,
    ) -> CallProofResult<Self> {
        if name.is_empty()
            || name.len() > MCP_CALL_PROOF_MAX_NAME_BYTES
            || tool.is_empty()
            || tool.len() > MCP_CALL_PROOF_MAX_NAME_BYTES
            || authenticated_user_id.is_empty()
            || authenticated_user_id.len() > MCP_CALL_PROOF_MAX_NAME_BYTES
            || !arguments.is_object()
        {
            return Err(failure(
                McpCallProofErrorCode::InvalidRequest,
                CallProofStage::Validate,
                transport,
            ));
        }
        let arguments_bytes = serde_json::to_vec(&arguments).map_err(|_| {
            failure(
                McpCallProofErrorCode::InvalidRequest,
                CallProofStage::Validate,
                transport,
            )
        })?;
        if arguments_bytes.len() > MCP_CALL_PROOF_MAX_ARGUMENT_BYTES {
            return Err(failure(
                McpCallProofErrorCode::ArgumentsTooLarge,
                CallProofStage::Validate,
                transport,
            ));
        }
        Ok(Self {
            server_name: name.to_owned(),
            tool: tool.to_owned(),
            arguments,
            arguments_bytes,
            authenticated_subject_sha256: sha256(authenticated_user_id.as_bytes()),
            runtime_scope_id: runtime_scope_id.map(str::to_owned),
        })
    }
}

struct ProofHttpResponse {
    rpc: JsonRpcResponse,
    session_id: Option<String>,
}

struct HttpSessionCleanupGuard {
    client: reqwest::Client,
    url: String,
    headers: reqwest::header::HeaderMap,
    armed: bool,
}

impl HttpSessionCleanupGuard {
    fn new(client: reqwest::Client, url: String, headers: reqwest::header::HeaderMap) -> Self {
        Self {
            client,
            url,
            headers,
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for HttpSessionCleanupGuard {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let client = self.client.clone();
        let url = self.url.clone();
        let headers = self.headers.clone();
        tokio::spawn(async move {
            const CLEANUP_TIMEOUT: Duration = Duration::from_secs(2);
            let _ = tokio::time::timeout(CLEANUP_TIMEOUT, client.delete(url).headers(headers).send()).await;
        });
    }
}

async fn run_stdio_call_protocol(
    mut stdin: tokio::process::ChildStdin,
    stdout: tokio::process::ChildStdout,
    prepared: &PreparedCall,
) -> CallProofResult<McpCallProofResult> {
    let mut reader = BufReader::new(stdout);
    write_bounded_json_line(&mut stdin, &build_initialize_request(1), CallProofStage::Initialize).await?;
    let init = read_bounded_jsonrpc_line(&mut reader, CallProofStage::Initialize).await?;
    expect_response_id(&init, 1, CallProofStage::Initialize, "stdio")?;
    let protocol_version = validate_initialize(&init, "stdio")?;
    write_bounded_json_line(
        &mut stdin,
        &build_initialized_notification(),
        CallProofStage::Initialized,
    )
    .await?;
    write_bounded_json_line(&mut stdin, &build_tools_list_request(2), CallProofStage::ToolsList).await?;
    let tools = read_bounded_jsonrpc_line(&mut reader, CallProofStage::ToolsList).await?;
    expect_response_id(&tools, 2, CallProofStage::ToolsList, "stdio")?;
    let tools_value = validate_tools_list(&tools, &prepared.tool, "stdio")?;
    write_bounded_json_line(
        &mut stdin,
        &build_tools_call_request(TOOL_CALL_REQUEST_ID, &prepared.tool, prepared.arguments.clone()),
        CallProofStage::ToolsCall,
    )
    .await?;
    let call = read_bounded_jsonrpc_line(&mut reader, CallProofStage::ToolsCall).await?;
    expect_response_id(&call, TOOL_CALL_REQUEST_ID, CallProofStage::ToolsCall, "stdio")?;
    build_proof(
        prepared,
        protocol_version,
        tools_value,
        validate_tool_call(&call, "stdio")?,
    )
}

async fn write_bounded_json_line<T: Serialize>(
    stdin: &mut tokio::process::ChildStdin,
    message: &T,
    stage: CallProofStage,
) -> CallProofResult<()> {
    let bytes =
        serde_json::to_vec(message).map_err(|_| failure(McpCallProofErrorCode::ProtocolError, stage, "stdio"))?;
    stdin
        .write_all(&bytes)
        .await
        .map_err(|_| failure(McpCallProofErrorCode::ConnectionFailed, stage, "stdio"))?;
    stdin
        .write_all(b"\n")
        .await
        .map_err(|_| failure(McpCallProofErrorCode::ConnectionFailed, stage, "stdio"))?;
    stdin
        .flush()
        .await
        .map_err(|_| failure(McpCallProofErrorCode::ConnectionFailed, stage, "stdio"))
}

async fn read_bounded_jsonrpc_line<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    stage: CallProofStage,
) -> CallProofResult<JsonRpcResponse> {
    loop {
        let mut line = Vec::new();
        let read = tokio::io::AsyncReadExt::take(&mut *reader, (MCP_CALL_PROOF_MAX_RESPONSE_BYTES + 1) as u64)
            .read_until(b'\n', &mut line)
            .await
            .map_err(|_| failure(McpCallProofErrorCode::ConnectionFailed, stage, "stdio"))?;
        if read == 0 {
            return Err(failure(McpCallProofErrorCode::ProtocolError, stage, "stdio"));
        }
        if line.len() > MCP_CALL_PROOF_MAX_RESPONSE_BYTES || !line.ends_with(b"\n") {
            return Err(failure(McpCallProofErrorCode::ResponseTooLarge, stage, "stdio"));
        }
        if let Ok(response) = serde_json::from_slice::<JsonRpcResponse>(&line)
            && response.id.is_some()
        {
            return Ok(response);
        }
    }
}

fn build_tools_call_request(id: u64, tool: &str, arguments: Value) -> JsonRpcRequest {
    JsonRpcRequest {
        jsonrpc: "2.0",
        id,
        method: "tools/call".to_owned(),
        params: Some(json!({ "name": tool, "arguments": arguments })),
    }
}

fn expect_response_id(
    response: &JsonRpcResponse,
    expected: u64,
    stage: CallProofStage,
    transport: &'static str,
) -> CallProofResult<()> {
    if response.id != Some(expected) {
        return Err(failure(McpCallProofErrorCode::ProtocolError, stage, transport));
    }
    Ok(())
}

fn validate_initialize(response: &JsonRpcResponse, transport: &'static str) -> CallProofResult<String> {
    reject_rpc_error(response, CallProofStage::Initialize, transport)?;
    let version = response
        .result
        .as_ref()
        .and_then(|value| value.get("protocolVersion"))
        .and_then(Value::as_str)
        .filter(|value| *value == PROTOCOL_VERSION)
        .ok_or_else(|| {
            failure(
                McpCallProofErrorCode::ProtocolError,
                CallProofStage::Initialize,
                transport,
            )
        })?;
    Ok(version.to_owned())
}

fn validate_tools_list(
    response: &JsonRpcResponse,
    requested_tool: &str,
    transport: &'static str,
) -> CallProofResult<Value> {
    reject_rpc_error(response, CallProofStage::ToolsList, transport)?;
    let result = response.result.as_ref().ok_or_else(|| {
        failure(
            McpCallProofErrorCode::ProtocolError,
            CallProofStage::ToolsList,
            transport,
        )
    })?;
    let bytes = serde_json::to_vec(result).map_err(|_| {
        failure(
            McpCallProofErrorCode::ProtocolError,
            CallProofStage::ToolsList,
            transport,
        )
    })?;
    if bytes.len() > MCP_CALL_PROOF_MAX_RESPONSE_BYTES {
        return Err(failure(
            McpCallProofErrorCode::ResponseTooLarge,
            CallProofStage::ToolsList,
            transport,
        ));
    }
    let tools = result.get("tools").and_then(Value::as_array).ok_or_else(|| {
        failure(
            McpCallProofErrorCode::ProtocolError,
            CallProofStage::ToolsList,
            transport,
        )
    })?;
    let matches: Vec<&Value> = tools
        .iter()
        .filter(|tool| tool.get("name").and_then(Value::as_str) == Some(requested_tool))
        .collect();
    if matches.is_empty() {
        return Err(failure(
            McpCallProofErrorCode::ToolNotFound,
            CallProofStage::ToolsList,
            transport,
        ));
    }
    if matches.len() != 1 {
        return Err(failure(
            McpCallProofErrorCode::ProtocolError,
            CallProofStage::ToolsList,
            transport,
        ));
    }
    let read_only = matches[0]
        .get("annotations")
        .and_then(|annotations| annotations.get("readOnlyHint"))
        .and_then(Value::as_bool)
        == Some(true);
    if !read_only {
        return Err(failure(
            McpCallProofErrorCode::ToolNotReadOnly,
            CallProofStage::ToolsList,
            transport,
        ));
    }
    Ok(result.clone())
}

fn validate_tool_call(response: &JsonRpcResponse, transport: &'static str) -> CallProofResult<Value> {
    reject_rpc_error(response, CallProofStage::ToolsCall, transport)?;
    let result = response.result.as_ref().ok_or_else(|| {
        failure(
            McpCallProofErrorCode::ProtocolError,
            CallProofStage::ToolsCall,
            transport,
        )
    })?;
    if !result.is_object()
        || !["content", "structuredContent", "isError", "_meta"]
            .iter()
            .any(|field| result.get(*field).is_some())
        || result.get("content").is_some_and(|content| !content.is_array())
        || result.get("isError").is_some_and(|is_error| !is_error.is_boolean())
    {
        return Err(failure(
            McpCallProofErrorCode::ProtocolError,
            CallProofStage::ToolsCall,
            transport,
        ));
    }
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        return Err(failure(
            McpCallProofErrorCode::ToolCallFailed,
            CallProofStage::ToolsCall,
            transport,
        ));
    }
    let bytes = serde_json::to_vec(result).map_err(|_| {
        failure(
            McpCallProofErrorCode::ProtocolError,
            CallProofStage::ToolsCall,
            transport,
        )
    })?;
    if bytes.len() > MCP_CALL_PROOF_MAX_RESPONSE_BYTES {
        return Err(failure(
            McpCallProofErrorCode::ResponseTooLarge,
            CallProofStage::ToolsCall,
            transport,
        ));
    }
    Ok(result.clone())
}

fn reject_rpc_error(response: &JsonRpcResponse, stage: CallProofStage, transport: &'static str) -> CallProofResult<()> {
    if let Some(error) = &response.error {
        return Err(McpCallProofFailure {
            code: McpCallProofErrorCode::RpcError,
            stage,
            transport,
            rpc_code: Some(error.code),
        });
    }
    Ok(())
}

fn build_proof(
    prepared: &PreparedCall,
    protocol_version: String,
    tools: Value,
    result: Value,
) -> CallProofResult<McpCallProofResult> {
    let tools_bytes = serde_json::to_vec(&tools).map_err(|_| {
        failure(
            McpCallProofErrorCode::ProtocolError,
            CallProofStage::ToolsList,
            "unknown",
        )
    })?;
    let result_bytes = serde_json::to_vec(&result).map_err(|_| {
        failure(
            McpCallProofErrorCode::ProtocolError,
            CallProofStage::ToolsCall,
            "unknown",
        )
    })?;
    let arguments_sha256 = sha256(&prepared.arguments_bytes);
    let tools_sha256 = sha256(&tools_bytes);
    let result_sha256 = sha256(&result_bytes);
    let proof_bytes = serde_json::to_vec(&json!({
        "protocol_version": protocol_version,
        "server_name": prepared.server_name,
        "runtime_scope_id": prepared.runtime_scope_id,
        "tool": prepared.tool,
        "authenticated_subject_sha256": prepared.authenticated_subject_sha256,
        "arguments_sha256": arguments_sha256,
        "tools_sha256": tools_sha256,
        "result_sha256": result_sha256,
    }))
    .expect("proof tuple should serialize");
    Ok(McpCallProofResult {
        protocol_version,
        server_name: prepared.server_name.clone(),
        runtime_scope_id: prepared.runtime_scope_id.clone(),
        tool: prepared.tool.clone(),
        authenticated_subject_sha256: prepared.authenticated_subject_sha256.clone(),
        arguments_sha256,
        tools_sha256,
        result_sha256,
        proof_sha256: sha256(&proof_bytes),
        result_bytes: result_bytes.len(),
        content_items: result.get("content").and_then(Value::as_array).map_or(0, Vec::len),
    })
}

fn sha256(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

async fn parse_bounded_http_response(
    response: reqwest::Response,
    stage: CallProofStage,
    transport: &'static str,
) -> CallProofResult<JsonRpcResponse> {
    let is_sse = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.contains("text/event-stream"));
    if is_sse {
        let (event_tx, mut event_rx) = mpsc::channel::<Result<SseEvent, ()>>(1);
        let reader = tokio::spawn(read_sse_events_bounded(
            response,
            event_tx,
            MCP_CALL_PROOF_MAX_RESPONSE_BYTES,
        ));
        let result = wait_for_bounded_sse_response(&mut event_rx, stage).await;
        reader.abort();
        return result.map_err(|mut error| {
            error.transport = transport;
            error
        });
    }

    let mut response = response;
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|_| failure(McpCallProofErrorCode::ConnectionFailed, stage, transport))?
    {
        if body.len().saturating_add(chunk.len()) > MCP_CALL_PROOF_MAX_RESPONSE_BYTES {
            return Err(failure(McpCallProofErrorCode::ResponseTooLarge, stage, transport));
        }
        body.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&body).map_err(|_| failure(McpCallProofErrorCode::ProtocolError, stage, transport))
}

fn validate_http_status(
    response: &reqwest::Response,
    stage: CallProofStage,
    transport: &'static str,
) -> CallProofResult<()> {
    if response.status().is_redirection() {
        return Err(failure(McpCallProofErrorCode::RedirectRejected, stage, transport));
    }
    if !response.status().is_success() {
        return Err(failure(McpCallProofErrorCode::HttpError, stage, transport));
    }
    Ok(())
}

fn validate_transport_url(url: &str, transport: &'static str) -> CallProofResult<reqwest::Url> {
    let parsed = reqwest::Url::parse(url).map_err(|_| {
        failure(
            McpCallProofErrorCode::InvalidRequest,
            CallProofStage::Validate,
            transport,
        )
    })?;
    if !matches!(parsed.scheme(), "http" | "https") || !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(failure(
            McpCallProofErrorCode::InvalidRequest,
            CallProofStage::Validate,
            transport,
        ));
    }
    Ok(parsed)
}

async fn wait_for_bounded_endpoint(
    events: &mut mpsc::Receiver<Result<SseEvent, ()>>,
    base: &reqwest::Url,
) -> CallProofResult<String> {
    while let Some(event) = events.recv().await {
        let event =
            event.map_err(|_| failure(McpCallProofErrorCode::ResponseTooLarge, CallProofStage::Endpoint, "sse"))?;
        if event.data.len() > MCP_CALL_PROOF_MAX_RESPONSE_BYTES {
            return Err(failure(
                McpCallProofErrorCode::ResponseTooLarge,
                CallProofStage::Endpoint,
                "sse",
            ));
        }
        if event.event_type == "endpoint" {
            let endpoint = base
                .join(&event.data)
                .map_err(|_| failure(McpCallProofErrorCode::ProtocolError, CallProofStage::Endpoint, "sse"))?;
            if endpoint.scheme() != base.scheme()
                || endpoint.host_str() != base.host_str()
                || endpoint.port_or_known_default() != base.port_or_known_default()
                || !endpoint.username().is_empty()
                || endpoint.password().is_some()
            {
                return Err(failure(
                    McpCallProofErrorCode::RedirectRejected,
                    CallProofStage::Endpoint,
                    "sse",
                ));
            }
            return Ok(endpoint.to_string());
        }
    }
    Err(failure(
        McpCallProofErrorCode::ProtocolError,
        CallProofStage::Endpoint,
        "sse",
    ))
}

async fn wait_for_bounded_sse_response(
    events: &mut mpsc::Receiver<Result<SseEvent, ()>>,
    stage: CallProofStage,
) -> CallProofResult<JsonRpcResponse> {
    while let Some(event) = events.recv().await {
        let event = event.map_err(|_| failure(McpCallProofErrorCode::ResponseTooLarge, stage, "sse"))?;
        if event.data.len() > MCP_CALL_PROOF_MAX_RESPONSE_BYTES {
            return Err(failure(McpCallProofErrorCode::ResponseTooLarge, stage, "sse"));
        }
        if (event.event_type.is_empty() || event.event_type == "message")
            && let Ok(response) = serde_json::from_str::<JsonRpcResponse>(&event.data)
            && response.id.is_some()
        {
            return Ok(response);
        }
    }
    Err(failure(McpCallProofErrorCode::ProtocolError, stage, "sse"))
}

fn failure(code: McpCallProofErrorCode, stage: CallProofStage, transport: &'static str) -> McpCallProofFailure {
    McpCallProofFailure {
        code,
        stage,
        transport,
        rpc_code: None,
    }
}

fn spawn_error(command: &str, error: &std::io::Error) -> McpCallProofFailure {
    let code = match error.kind() {
        std::io::ErrorKind::NotFound => McpCallProofErrorCode::CommandNotFound,
        std::io::ErrorKind::PermissionDenied => McpCallProofErrorCode::CommandPermissionDenied,
        _ => McpCallProofErrorCode::CommandStartFailed,
    };
    let _ = command;
    failure(code, CallProofStage::Spawn, "stdio")
}

fn resolve_stdio_command(command: &str) -> OsString {
    if !command.is_empty()
        && !command.contains('/')
        && !command.contains('\\')
        && let Some(path) = resolve_command_path(command)
    {
        return path.into_os_string();
    }
    OsString::from(command)
}

fn runtime_resolution_error(message: &str) -> std::io::Error {
    let lower = message.to_ascii_lowercase();
    if lower.contains("not found")
        || lower.contains("unsupported")
        || lower.contains("unavailable")
        || lower.contains("system node")
    {
        std::io::Error::new(std::io::ErrorKind::NotFound, "runtime unavailable")
    } else {
        std::io::Error::other("runtime preparation failed")
    }
}

fn transport_type(transport: &McpServerTransport) -> &'static str {
    match transport {
        McpServerTransport::Stdio { .. } => "stdio",
        McpServerTransport::Http { .. } => "http",
        McpServerTransport::Sse { .. } => "sse",
    }
}

#[cfg(test)]
mod tests;
