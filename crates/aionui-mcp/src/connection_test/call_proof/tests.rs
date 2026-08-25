use super::*;

fn response(result: Value) -> JsonRpcResponse {
    JsonRpcResponse {
        jsonrpc: "2.0".to_owned(),
        id: Some(2),
        result: Some(result),
        error: None,
    }
}

#[test]
fn exact_call_request_contains_only_requested_tool_and_arguments() {
    let request = build_tools_call_request(3, "search", json!({"query": "safe"}));
    let value = serde_json::to_value(request).unwrap();
    assert_eq!(value["method"], "tools/call");
    assert_eq!(
        value["params"],
        json!({"name": "search", "arguments": {"query": "safe"}})
    );
}

#[test]
fn missing_and_non_read_only_tools_fail_closed() {
    let listed = response(json!({"tools": [{"name": "write"}, {"name": "read", "annotations": {}}]}));
    assert_eq!(
        validate_tools_list(&listed, "absent", "stdio").unwrap_err().code,
        McpCallProofErrorCode::ToolNotFound
    );
    assert_eq!(
        validate_tools_list(&listed, "read", "stdio").unwrap_err().code,
        McpCallProofErrorCode::ToolNotReadOnly
    );
}

#[test]
fn rpc_error_and_tool_error_do_not_expose_payload() {
    let secret = "RPC_SECRET_42";
    let rpc = JsonRpcResponse {
        jsonrpc: "2.0".to_owned(),
        id: Some(3),
        result: None,
        error: Some(super::super::protocol::JsonRpcError {
            code: -32603,
            message: secret.to_owned(),
        }),
    };
    let error = validate_tool_call(&rpc, "http").unwrap_err();
    let serialized = serde_json::to_string(&error.details()).unwrap();
    assert_eq!(error.code, McpCallProofErrorCode::RpcError);
    assert!(!serialized.contains(secret));

    let failed = response(json!({"isError": true, "content": [{"type":"text", "text": secret}]}));
    let error = validate_tool_call(&failed, "stdio").unwrap_err();
    assert_eq!(error.code, McpCallProofErrorCode::ToolCallFailed);
    assert!(!serde_json::to_string(&error.details()).unwrap().contains(secret));
}

#[test]
fn oversized_arguments_are_rejected_before_transport() {
    let arguments = json!({"value": "x".repeat(MCP_CALL_PROOF_MAX_ARGUMENT_BYTES)});
    let error = PreparedCall::new("server", "read", arguments, "user-1", "http").unwrap_err();
    assert_eq!(error.code, McpCallProofErrorCode::ArgumentsTooLarge);
    assert_eq!(error.stage.as_str(), "validate");
}

#[test]
fn proof_is_stable_and_contains_no_raw_result() {
    let prepared = PreparedCall::new("server", "read", json!({"id": 7}), "user-1", "http").unwrap();
    let tools = json!({"tools": [{"name": "read", "annotations": {"readOnlyHint": true}}]});
    let result = json!({"content": [{"type": "text", "text": "RESULT_SECRET_42"}]});
    let first = build_proof(&prepared, "2024-11-05".to_owned(), tools.clone(), result.clone()).unwrap();
    let second = build_proof(&prepared, "2024-11-05".to_owned(), tools, result).unwrap();
    assert_eq!(first, second);
    let serialized = serde_json::to_string(&first).unwrap();
    assert!(!serialized.contains("RESULT_SECRET_42"));
    assert_eq!(first.content_items, 1);
}

#[test]
fn unsupported_protocol_version_and_malformed_call_result_fail_closed() {
    let init = JsonRpcResponse {
        jsonrpc: "2.0".to_owned(),
        id: Some(1),
        result: Some(json!({"protocolVersion": "2099-01-01"})),
        error: None,
    };
    assert_eq!(
        validate_initialize(&init, "http").unwrap_err().code,
        McpCallProofErrorCode::ProtocolError
    );

    let malformed = response(json!({"unexpected": true}));
    assert_eq!(
        validate_tool_call(&malformed, "http").unwrap_err().code,
        McpCallProofErrorCode::ProtocolError
    );
}

#[test]
fn proof_digest_binds_the_requested_server_name() {
    let one = PreparedCall::new("server-one", "read", json!({"id": 7}), "user-1", "http").unwrap();
    let two = PreparedCall::new("server-two", "read", json!({"id": 7}), "user-1", "http").unwrap();
    let tools = json!({"tools": [{"name": "read", "annotations": {"readOnlyHint": true}}]});
    let result = json!({"content": []});
    let first = build_proof(&one, "2024-11-05".to_owned(), tools.clone(), result.clone()).unwrap();
    let second = build_proof(&two, "2024-11-05".to_owned(), tools, result).unwrap();
    assert_ne!(first.proof_sha256, second.proof_sha256);
}

#[tokio::test]
async fn cross_origin_sse_endpoint_is_rejected() {
    let base = reqwest::Url::parse("https://one.example/sse").unwrap();
    let (sender, mut receiver) = mpsc::channel(1);
    sender
        .send(Ok(SseEvent {
            event_type: "endpoint".to_owned(),
            data: "https://two.example/messages".to_owned(),
        }))
        .await
        .unwrap();
    drop(sender);
    let error = wait_for_bounded_endpoint(&mut receiver, &base).await.unwrap_err();
    assert_eq!(error.code, McpCallProofErrorCode::RedirectRejected);
    assert_eq!(error.stage.as_str(), "endpoint");
}
