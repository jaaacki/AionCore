//! Integration tests for McpConnectionTestService.
//!
//! Tests from test-plan §2 (Connection Test):
//! - CT-3: Command not found (ENOENT)
//! - CT-4: URL not reachable
//! - CT-5: Needs OAuth authentication (401)
//! - CT-6: Timeout
//! - SSE auth probe (M-33 coverage)

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use aionui_mcp::McpConnectionTestService;
use aionui_mcp::McpServerTransport;
use aionui_realtime::BroadcastEventBus;

fn make_service() -> McpConnectionTestService {
    McpConnectionTestService::new(reqwest::Client::new(), Arc::new(BroadcastEventBus::new(16)))
}

fn make_service_with_timeout(timeout: Duration) -> McpConnectionTestService {
    McpConnectionTestService::new(reqwest::Client::new(), Arc::new(BroadcastEventBus::new(16))).with_timeout(timeout)
}

// ---------------------------------------------------------------------------
// CT-3: Command not found (ENOENT)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stdio_nonexistent_command_returns_not_found_error() {
    let svc = make_service();
    let transport = McpServerTransport::Stdio {
        command: "nonexistent-mcp-cmd-xyz-12345".into(),
        args: vec![],
        env: HashMap::new(),
    };

    let result = svc.test_connection("test-server", &transport).await;

    assert!(!result.success);
    let error = result.error.as_deref().unwrap();
    assert!(
        error.contains("Command not found"),
        "expected 'Command not found' in: {error}"
    );
    assert!(result.tools.is_none());
    assert!(result.needs_auth.is_none());
}

// ---------------------------------------------------------------------------
// CT-4: URL not reachable
// ---------------------------------------------------------------------------

#[tokio::test]
async fn http_unreachable_url_returns_connection_error() {
    let svc = make_service_with_timeout(Duration::from_secs(5));
    let transport = McpServerTransport::Http {
        url: "http://127.0.0.1:1/mcp-unreachable".into(),
        headers: HashMap::new(),
    };

    let result = svc.test_connection("test-http", &transport).await;

    assert!(!result.success);
    let error = result.error.as_deref().unwrap();
    assert!(
        error.contains("Connection failed"),
        "expected connection failure in: {error}"
    );
}

#[tokio::test]
async fn sse_unreachable_url_returns_connection_error() {
    let svc = make_service_with_timeout(Duration::from_secs(5));
    let transport = McpServerTransport::Sse {
        url: "http://127.0.0.1:1/sse-unreachable".into(),
        headers: HashMap::new(),
    };

    let result = svc.test_connection("test-sse", &transport).await;

    assert!(!result.success);
    let error = result.error.as_deref().unwrap();
    assert!(
        error.contains("Connection failed"),
        "expected connection failure in: {error}"
    );
}

// ---------------------------------------------------------------------------
// CT-5: HTTP 401 Unauthorized -> needsAuth
// ---------------------------------------------------------------------------

#[tokio::test]
async fn http_401_returns_needs_auth() {
    // Spin up a mock server that returns 401 with WWW-Authenticate
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        let app = axum::Router::new().route(
            "/mcp",
            axum::routing::post(|| async {
                (
                    axum::http::StatusCode::UNAUTHORIZED,
                    [(axum::http::header::WWW_AUTHENTICATE, "Bearer realm=\"mcp-server\"")],
                    "",
                )
            }),
        );
        axum::serve(listener, app).await.unwrap();
    });

    let svc = make_service();
    let transport = McpServerTransport::Http {
        url: format!("http://{}/mcp", addr),
        headers: HashMap::new(),
    };

    let result = svc.test_connection("auth-server", &transport).await;

    assert!(!result.success);
    assert_eq!(result.needs_auth, Some(true));
    assert!(result.auth_method.is_some());
    assert!(result.www_authenticate.is_some());
    assert!(result.error.is_none());

    server_handle.abort();
}

#[tokio::test]
async fn sse_401_returns_needs_auth() {
    // Spin up a mock server that returns 401 for GET
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        let app = axum::Router::new().route(
            "/sse",
            axum::routing::get(|| async {
                (
                    axum::http::StatusCode::UNAUTHORIZED,
                    [(axum::http::header::WWW_AUTHENTICATE, "Bearer realm=\"mcp-sse\"")],
                    "",
                )
            }),
        );
        axum::serve(listener, app).await.unwrap();
    });

    let svc = make_service();
    let transport = McpServerTransport::Sse {
        url: format!("http://{}/sse", addr),
        headers: HashMap::new(),
    };

    let result = svc.test_connection("sse-auth", &transport).await;

    assert!(!result.success);
    assert_eq!(result.needs_auth, Some(true));
    assert!(result.www_authenticate.is_some());

    server_handle.abort();
}

// ---------------------------------------------------------------------------
// CT-6: Timeout
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stdio_timeout_returns_timeout_error() {
    // Use `sleep` which produces no stdout — our protocol read will block
    let svc = make_service_with_timeout(Duration::from_secs(1));
    let transport = McpServerTransport::Stdio {
        command: "sleep".into(),
        args: vec!["60".into()],
        env: HashMap::new(),
    };

    let result = svc.test_connection("timeout-server", &transport).await;

    assert!(!result.success);
    let error = result.error.as_deref().unwrap();
    assert!(error.contains("timed out"), "expected timeout in: {error}");
}

// ---------------------------------------------------------------------------
// HTTP non-success status
// ---------------------------------------------------------------------------

#[tokio::test]
async fn http_500_returns_error_with_status() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        let app = axum::Router::new().route(
            "/mcp",
            axum::routing::post(|| async { axum::http::StatusCode::INTERNAL_SERVER_ERROR }),
        );
        axum::serve(listener, app).await.unwrap();
    });

    let svc = make_service();
    let transport = McpServerTransport::Http {
        url: format!("http://{}/mcp", addr),
        headers: HashMap::new(),
    };

    let result = svc.test_connection("error-server", &transport).await;

    assert!(!result.success);
    let error = result.error.as_deref().unwrap();
    assert!(error.contains("500"), "expected HTTP 500 in: {error}");

    server_handle.abort();
}

// ---------------------------------------------------------------------------
// HTTP transport with custom headers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn http_custom_headers_are_sent() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let server_handle = tokio::spawn(async move {
        let app = axum::Router::new().route(
            "/mcp",
            axum::routing::post(|headers: axum::http::HeaderMap| async move {
                // Verify the custom header was received
                if headers.get("x-api-key").and_then(|v| v.to_str().ok()) == Some("secret") {
                    // Return a valid initialize response
                    axum::Json(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "serverInfo": { "name": "test", "version": "1.0" }
                        }
                    }))
                } else {
                    // Return error if header missing
                    axum::Json(serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "error": { "code": -1, "message": "Missing API key" }
                    }))
                }
            }),
        );
        axum::serve(listener, app).await.unwrap();
    });

    let svc = make_service();
    let mut headers = HashMap::new();
    headers.insert("X-Api-Key".into(), "secret".into());
    let transport = McpServerTransport::Http {
        url: format!("http://{}/mcp", addr),
        headers,
    };

    let result = svc.test_connection("header-server", &transport).await;

    // The server returns a valid initialize response for request id=1,
    // but the subsequent tools/list (id=2) will also hit the same handler.
    // Either way, the first request should succeed (no initialize error).
    // The tools/list might succeed or fail depending on how the mock handles id=2.
    // For this test, we just verify the custom header was sent (no "Missing API key" error).
    if let Some(ref error) = result.error {
        assert!(
            !error.contains("Missing API key"),
            "Custom header should have been sent"
        );
    }

    server_handle.abort();
}

// ---------------------------------------------------------------------------
// Stdio with args and env
// ---------------------------------------------------------------------------

#[tokio::test]
async fn stdio_with_args_spawns_correctly() {
    // Use echo as a simple command that exits immediately
    // Since echo doesn't speak MCP, we expect a protocol error (not a spawn error)
    let svc = make_service_with_timeout(Duration::from_secs(3));
    let transport = McpServerTransport::Stdio {
        command: "echo".into(),
        args: vec!["hello".into()],
        env: HashMap::new(),
    };

    let result = svc.test_connection("echo-server", &transport).await;

    // echo outputs "hello\n" then exits — not valid JSON-RPC
    assert!(!result.success);
    let error = result.error.as_deref().unwrap();
    // Should be a protocol error, not a spawn error
    assert!(!error.contains("Command not found"), "echo should be found");
}

// ---------------------------------------------------------------------------
// Exact call proof: genuine protocol lifecycles and hostile transport cases
// ---------------------------------------------------------------------------

use aionui_api_types::{MCP_CALL_PROOF_MAX_RESPONSE_BYTES, McpCallProofErrorCode};
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, Response, StatusCode};
use axum::response::sse::{Event, Sse};
use futures_util::stream::{self, StreamExt};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::broadcast;

#[derive(Clone, Default)]
struct HttpCallState {
    calls: Arc<std::sync::Mutex<Vec<Value>>>,
    saw_session: Arc<std::sync::atomic::AtomicBool>,
    saw_delete: Arc<std::sync::atomic::AtomicBool>,
}

async fn http_delete_handler(State(state): State<HttpCallState>, headers: HeaderMap) -> StatusCode {
    if headers.get("mcp-session-id").and_then(|value| value.to_str().ok()) == Some("session-1") {
        state.saw_delete.store(true, Ordering::SeqCst);
    }
    StatusCode::NO_CONTENT
}

async fn http_call_handler(
    State(state): State<HttpCallState>,
    headers: HeaderMap,
    axum::Json(body): axum::Json<Value>,
) -> Response<Body> {
    state.calls.lock().unwrap().push(body.clone());
    if body["method"] != "initialize"
        && headers.get("mcp-session-id").and_then(|v| v.to_str().ok()) == Some("session-1")
    {
        state.saw_session.store(true, Ordering::SeqCst);
    }
    let method = body["method"].as_str().unwrap_or_default();
    if method == "notifications/initialized" {
        return Response::builder()
            .status(StatusCode::ACCEPTED)
            .body(Body::empty())
            .unwrap();
    }
    let id = body["id"].clone();
    let result = match method {
        "initialize" => json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "fixture", "version": "1"}
        }),
        "tools/list" => json!({
            "tools": [{
                "name": "read_exact",
                "annotations": {"readOnlyHint": true},
                "inputSchema": {"type": "object"}
            }]
        }),
        "tools/call" => {
            assert_eq!(body["params"]["name"], "read_exact");
            assert_eq!(body["params"]["arguments"], json!({"record_id": "7"}));
            json!({
                "content": [{"type": "text", "text": "HTTP_RESULT_SECRET"}],
                "isError": false
            })
        }
        _ => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::empty())
                .unwrap();
        }
    };
    let mut response = Response::new(Body::from(
        serde_json::to_vec(&json!({"jsonrpc": "2.0", "id": id, "result": result})).unwrap(),
    ));
    response
        .headers_mut()
        .insert("content-type", "application/json".parse().unwrap());
    if method == "initialize" {
        response
            .headers_mut()
            .insert("mcp-session-id", "session-1".parse().unwrap());
    }
    response
}

#[tokio::test]
async fn http_call_proof_runs_exact_lifecycle_and_returns_payload_free_proof() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let state = HttpCallState::default();
    let server_state = state.clone();
    let server = tokio::spawn(async move {
        let app = axum::Router::new()
            .route(
                "/mcp",
                axum::routing::post(http_call_handler).delete(http_delete_handler),
            )
            .with_state(server_state);
        axum::serve(listener, app).await.unwrap();
    });

    let transport = McpServerTransport::Http {
        url: format!("http://{addr}/mcp"),
        headers: HashMap::from([("Authorization".to_owned(), "Bearer HTTP_HEADER_SECRET".to_owned())]),
    };
    let proof = make_service()
        .call_proof(
            "fixture",
            &transport,
            "read_exact",
            json!({"record_id": "7"}),
            "user-1",
            Some("scope-1"),
        )
        .await
        .unwrap();

    server.abort();
    let methods: Vec<String> = state
        .calls
        .lock()
        .unwrap()
        .iter()
        .filter_map(|call| call["method"].as_str().map(str::to_owned))
        .collect();
    assert_eq!(
        methods,
        ["initialize", "notifications/initialized", "tools/list", "tools/call"]
    );
    assert!(state.saw_session.load(Ordering::SeqCst));
    assert!(state.saw_delete.load(Ordering::SeqCst));
    assert_eq!(proof.tool, "read_exact");
    assert_eq!(proof.protocol_version, "2024-11-05");
    assert_eq!(proof.content_items, 1);
    let serialized = serde_json::to_string(&proof).unwrap();
    assert!(!serialized.contains("HTTP_RESULT_SECRET"));
    assert!(!serialized.contains("HTTP_HEADER_SECRET"));
    assert!(!serialized.contains("record_id"));
}

#[tokio::test]
async fn http_call_proof_accepts_streaming_sse_response_without_waiting_for_eof() {
    use axum::body::Bytes;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let app = axum::Router::new().route(
            "/mcp",
            axum::routing::post(|axum::Json(body): axum::Json<Value>| async move {
                let method = body["method"].as_str().unwrap_or_default();
                if method == "notifications/initialized" {
                    return Response::builder()
                        .status(StatusCode::ACCEPTED)
                        .body(Body::empty())
                        .unwrap();
                }
                let result = match method {
                    "initialize" => json!({
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {"name": "fixture", "version": "1"}
                    }),
                    "tools/list" => json!({
                        "tools": [{"name": "read_exact", "annotations": {"readOnlyHint": true}}]
                    }),
                    "tools/call" => json!({"content": [], "isError": false}),
                    _ => unreachable!(),
                };
                let payload = format!(
                    "data: {}\n\n",
                    json!({"jsonrpc": "2.0", "id": body["id"], "result": result})
                );
                let stream =
                    stream::once(async move { Ok::<_, Infallible>(Bytes::from(payload)) }).chain(stream::pending());
                Response::builder()
                    .header("content-type", "text/event-stream")
                    .body(Body::from_stream(stream))
                    .unwrap()
            }),
        );
        axum::serve(listener, app).await.unwrap();
    });

    let transport = McpServerTransport::Http {
        url: format!("http://{addr}/mcp"),
        headers: HashMap::new(),
    };
    let proof = make_service_with_timeout(Duration::from_secs(3))
        .call_proof("streaming", &transport, "read_exact", json!({}), "user-1", None)
        .await
        .unwrap();

    server.abort();
    assert_eq!(proof.tool, "read_exact");
}

#[tokio::test]
async fn http_call_proof_rejects_redirect_without_forwarding_authorization() {
    let target_hits = Arc::new(AtomicUsize::new(0));
    let target_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let target_addr = target_listener.local_addr().unwrap();
    let target_hits_server = target_hits.clone();
    let target_server = tokio::spawn(async move {
        let app = axum::Router::new().route(
            "/capture",
            axum::routing::post(move |headers: HeaderMap| {
                let target_hits = target_hits_server.clone();
                async move {
                    if headers.contains_key("authorization") {
                        target_hits.fetch_add(1, Ordering::SeqCst);
                    }
                    StatusCode::OK
                }
            }),
        );
        axum::serve(target_listener, app).await.unwrap();
    });

    let redirect_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let redirect_addr = redirect_listener.local_addr().unwrap();
    let location = format!("http://{target_addr}/capture");
    let redirect_server = tokio::spawn(async move {
        let app = axum::Router::new().route(
            "/mcp",
            axum::routing::post(move || {
                let location = location.clone();
                async move { (StatusCode::TEMPORARY_REDIRECT, [("location", location)]) }
            }),
        );
        axum::serve(redirect_listener, app).await.unwrap();
    });

    let transport = McpServerTransport::Http {
        url: format!("http://{redirect_addr}/mcp"),
        headers: HashMap::from([("Authorization".to_owned(), "Bearer DO_NOT_FORWARD".to_owned())]),
    };
    let error = make_service()
        .call_proof("redirect", &transport, "read_exact", json!({}), "user-1", None)
        .await
        .unwrap_err();

    redirect_server.abort();
    target_server.abort();
    assert_eq!(error.code(), McpCallProofErrorCode::RedirectRejected);
    assert_eq!(target_hits.load(Ordering::SeqCst), 0);
    let serialized = serde_json::to_string(&error.details()).unwrap();
    assert!(!serialized.contains("DO_NOT_FORWARD"));
    assert!(!serialized.contains(&target_addr.to_string()));
}

#[derive(Clone)]
struct SseCallState {
    events: broadcast::Sender<String>,
}

async fn sse_open(
    State(state): State<SseCallState>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let receiver = state.events.subscribe();
    let initial = stream::once(async { Ok(Event::default().event("endpoint").data("/messages")) });
    let updates = stream::unfold(receiver, |mut receiver| async move {
        loop {
            match receiver.recv().await {
                Ok(data) => return Some((Ok(Event::default().event("message").data(data)), receiver)),
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(initial.chain(updates))
}

async fn sse_message(State(state): State<SseCallState>, axum::Json(body): axum::Json<Value>) -> StatusCode {
    let method = body["method"].as_str().unwrap_or_default();
    if method == "notifications/initialized" {
        return StatusCode::ACCEPTED;
    }
    let id = body["id"].clone();
    let result = match method {
        "initialize" => json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "sse-fixture", "version": "1"}
        }),
        "tools/list" => json!({
            "tools": [{"name": "read_sse", "annotations": {"readOnlyHint": true}}]
        }),
        "tools/call" => {
            assert_eq!(body["params"], json!({"name": "read_sse", "arguments": {"limit": 1}}));
            json!({"content": [{"type": "text", "text": "SSE_RESULT_SECRET"}]})
        }
        _ => return StatusCode::BAD_REQUEST,
    };
    state
        .events
        .send(json!({"jsonrpc": "2.0", "id": id, "result": result}).to_string())
        .unwrap();
    StatusCode::ACCEPTED
}

#[tokio::test]
async fn sse_call_proof_runs_exact_lifecycle_and_keeps_result_redacted() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (events, _) = broadcast::channel(16);
    let server = tokio::spawn(async move {
        let app = axum::Router::new()
            .route("/sse", axum::routing::get(sse_open))
            .route("/messages", axum::routing::post(sse_message))
            .with_state(SseCallState { events });
        axum::serve(listener, app).await.unwrap();
    });

    let transport = McpServerTransport::Sse {
        url: format!("http://{addr}/sse"),
        headers: HashMap::new(),
    };
    let proof = make_service()
        .call_proof(
            "sse-fixture",
            &transport,
            "read_sse",
            json!({"limit": 1}),
            "user-1",
            None,
        )
        .await
        .unwrap();

    server.abort();
    assert_eq!(proof.tool, "read_sse");
    assert_eq!(proof.content_items, 1);
    assert!(!serde_json::to_string(&proof).unwrap().contains("SSE_RESULT_SECRET"));
}

#[tokio::test]
async fn http_call_proof_rejects_oversized_result_without_returning_it() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let app = axum::Router::new().route(
            "/mcp",
            axum::routing::post(|axum::Json(body): axum::Json<Value>| async move {
                let method = body["method"].as_str().unwrap_or_default();
                if method == "notifications/initialized" {
                    return Response::builder()
                        .status(StatusCode::ACCEPTED)
                        .body(Body::empty())
                        .unwrap();
                }
                let result = match method {
                    "initialize" => json!({"protocolVersion": "2024-11-05"}),
                    "tools/list" => json!({"tools": [{"name": "read", "annotations": {"readOnlyHint": true}}]}),
                    "tools/call" => {
                        json!({"content": [{"type": "text", "text": "x".repeat(MCP_CALL_PROOF_MAX_RESPONSE_BYTES)}]})
                    }
                    _ => json!({}),
                };
                Response::builder()
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({"jsonrpc": "2.0", "id": body["id"], "result": result}).to_string(),
                    ))
                    .unwrap()
            }),
        );
        axum::serve(listener, app).await.unwrap();
    });
    let transport = McpServerTransport::Http {
        url: format!("http://{addr}/mcp"),
        headers: HashMap::new(),
    };
    let error = make_service()
        .call_proof("oversized", &transport, "read", json!({}), "user-1", None)
        .await
        .unwrap_err();
    server.abort();
    assert_eq!(error.code(), McpCallProofErrorCode::ResponseTooLarge);
}

#[cfg(unix)]
#[tokio::test]
async fn stdio_call_proof_uses_real_child_protocol_and_exact_call() {
    let script = r#"
while IFS= read -r line; do
  case "$line" in
    *'"method":"initialize"'*) printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05"}}' ;;
    *'"method":"notifications/initialized"'*) ;;
    *'"method":"tools/list"'*) printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"read_stdio","annotations":{"readOnlyHint":true}}]}}' ;;
    *'"method":"tools/call"'*)
      if printf '%s' "$line" | grep -q '"name":"read_stdio"' && printf '%s' "$line" | grep -q '"id":9'; then
        printf '%s\n' '{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"STDIO_RESULT_SECRET"}]}}'
      else
        printf '%s\n' '{"jsonrpc":"2.0","id":3,"error":{"code":-32602,"message":"wrong exact call"}}'
      fi ;;
  esac
done
"#;
    let transport = McpServerTransport::Stdio {
        command: "sh".to_owned(),
        args: vec!["-c".to_owned(), script.to_owned()],
        env: HashMap::new(),
    };
    let proof = make_service()
        .call_proof(
            "stdio-fixture",
            &transport,
            "read_stdio",
            json!({"id": 9}),
            "user-1",
            None,
        )
        .await
        .unwrap();
    assert_eq!(proof.tool, "read_stdio");
    assert!(!serde_json::to_string(&proof).unwrap().contains("STDIO_RESULT_SECRET"));
}

#[tokio::test]
async fn http_call_proof_stops_when_exact_tool_is_absent() {
    let calls = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server_calls = calls.clone();
    let server = tokio::spawn(async move {
        let app = axum::Router::new().route(
            "/mcp",
            axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
                let calls = server_calls.clone();
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    if body["method"] == "notifications/initialized" {
                        return Response::builder()
                            .status(StatusCode::ACCEPTED)
                            .body(Body::empty())
                            .unwrap();
                    }
                    let result = if body["method"] == "initialize" {
                        json!({"protocolVersion": "2024-11-05"})
                    } else {
                        json!({"tools": [{"name": "different", "annotations": {"readOnlyHint": true}}]})
                    };
                    Response::builder()
                        .header("content-type", "application/json")
                        .body(Body::from(
                            json!({"jsonrpc": "2.0", "id": body["id"], "result": result}).to_string(),
                        ))
                        .unwrap()
                }
            }),
        );
        axum::serve(listener, app).await.unwrap();
    });
    let transport = McpServerTransport::Http {
        url: format!("http://{addr}/mcp"),
        headers: HashMap::new(),
    };
    let error = make_service()
        .call_proof("absent", &transport, "read_exact", json!({}), "user-1", None)
        .await
        .unwrap_err();
    server.abort();
    assert_eq!(error.code(), McpCallProofErrorCode::ToolNotFound);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        3,
        "tools/call must not run after an absent listing"
    );
}

#[tokio::test]
async fn http_call_proof_redacts_upstream_rpc_error_message() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let app = axum::Router::new().route(
            "/mcp",
            axum::routing::post(|axum::Json(body): axum::Json<Value>| async move {
                if body["method"] == "notifications/initialized" {
                    return Response::builder().status(StatusCode::ACCEPTED).body(Body::empty()).unwrap();
                }
                let envelope = match body["method"].as_str().unwrap_or_default() {
                    "initialize" => json!({"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2024-11-05"}}),
                    "tools/list" => json!({"jsonrpc": "2.0", "id": 2, "result": {"tools": [{"name": "read", "annotations": {"readOnlyHint": true}}]}}),
                    "tools/call" => json!({"jsonrpc": "2.0", "id": 3, "error": {"code": -32042, "message": "UPSTREAM_RPC_SECRET"}}),
                    _ => json!({}),
                };
                Response::builder()
                    .header("content-type", "application/json")
                    .body(Body::from(envelope.to_string()))
                    .unwrap()
            }),
        );
        axum::serve(listener, app).await.unwrap();
    });
    let transport = McpServerTransport::Http {
        url: format!("http://{addr}/mcp"),
        headers: HashMap::new(),
    };
    let error = make_service()
        .call_proof("rpc", &transport, "read", json!({}), "user-1", None)
        .await
        .unwrap_err();
    server.abort();
    assert_eq!(error.code(), McpCallProofErrorCode::RpcError);
    assert_eq!(error.details()["rpc_code"], -32042);
    let serialized = serde_json::to_string(&error.details()).unwrap();
    assert!(!serialized.contains("UPSTREAM_RPC_SECRET"));
}
