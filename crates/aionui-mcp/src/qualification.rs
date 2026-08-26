use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use aionui_api_types::{
    McpCallProofResult, McpQualificationBinding, McpQualificationCapabilityResult, McpQualificationRequest,
};
use aionui_db::IConversationRepository;
use aionui_process::read_process_start_time;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::connection_test::{McpCallProofFailure, McpConnectionTestService};
use crate::types::McpServerTransport;

const CAPABILITY_TTL: Duration = Duration::from_secs(30);
const MAX_BINDING_BYTES: usize = 1024;
const MAX_PENDING_CAPABILITIES: usize = 256;

#[derive(Clone)]
pub struct McpQualificationService {
    pending: Arc<Mutex<HashMap<String, PendingQualification>>>,
    conversations: Arc<dyn IConversationRepository>,
    calls: McpConnectionTestService,
}

#[derive(Clone)]
struct PendingQualification {
    user_id: String,
    conversation_id: String,
    request: McpQualificationRequest,
    binding: McpQualificationBinding,
}

#[derive(Debug)]
pub enum QualificationError {
    Invalid,
    Unauthorized,
    Expired,
    Internal,
    Call(McpCallProofFailure),
}

impl McpQualificationService {
    pub fn new(conversations: Arc<dyn IConversationRepository>, calls: McpConnectionTestService) -> Self {
        Self {
            pending: Arc::new(Mutex::new(HashMap::new())),
            conversations,
            calls,
        }
    }

    pub async fn mint(
        &self,
        user_id: &str,
        conversation_id: &str,
        request: McpQualificationRequest,
    ) -> Result<McpQualificationCapabilityResult, QualificationError> {
        validate_binding_input(&request)?;
        self.require_owner(user_id, conversation_id).await?;
        let identity = backend_identity()?;
        let issued_at_ms = now_ms()?;
        let expires_at_ms = issued_at_ms.saturating_add(CAPABILITY_TTL.as_millis() as u64);
        let capability = Uuid::new_v4().to_string();
        let arguments = serde_json::to_vec(&request.arguments).map_err(|_| QualificationError::Invalid)?;
        let mut binding = McpQualificationBinding {
            schema_version: 1,
            capability_sha256: sha256(capability.as_bytes()),
            authenticated_subject_sha256: sha256(user_id.as_bytes()),
            conversation_sha256: sha256(conversation_id.as_bytes()),
            runtime_id_sha256: identity.runtime_id_sha256,
            backend_pid: identity.pid,
            backend_start_time: identity.start_time,
            backend_executable_sha256: identity.executable_sha256,
            operation: format!("tools/call:{}", request.tool),
            arguments_sha256: sha256(&arguments),
            run_id: request.run_id.clone(),
            machine_challenge_sha256: sha256(request.machine_challenge.as_bytes()),
            issued_at_ms,
            expires_at_ms,
            binding_sha256: String::new(),
        };
        binding.binding_sha256 = qualification_binding_sha256(&binding);
        insert_pending(
            &self.pending,
            capability.clone(),
            PendingQualification {
                user_id: user_id.to_owned(),
                conversation_id: conversation_id.to_owned(),
                request,
                binding,
            },
            issued_at_ms,
        )?;
        Ok(McpQualificationCapabilityResult {
            capability,
            expires_at_ms,
        })
    }

    pub async fn consume(
        &self,
        user_id: &str,
        conversation_id: &str,
        capability: &str,
    ) -> Result<McpCallProofResult, QualificationError> {
        if capability.trim().is_empty() || capability.len() > 128 {
            return Err(QualificationError::Invalid);
        }
        // Removal is the single atomic consume decision. Every failure after it
        // burns the capability, so concurrent replay cannot race a read/write pair.
        let pending = take_pending(&self.pending, capability)?;
        validate_consume_context(&pending, user_id, conversation_id, now_ms()?)?;
        self.require_owner(user_id, conversation_id).await?;
        require_same_backend(&pending.binding)?;
        let transport = McpServerTransport::from(pending.request.transport);
        let mut proof = self
            .calls
            .call_proof(
                &pending.request.name,
                &transport,
                &pending.request.tool,
                pending.request.arguments,
                user_id,
                Some(&pending.request.run_id),
            )
            .await
            .map_err(QualificationError::Call)?;
        require_same_backend(&pending.binding)?;
        // Ownership is live authority, not a mint-time assertion. Re-read it
        // after the potentially long MCP operation before emitting proof.
        self.require_owner(user_id, conversation_id).await?;
        if proof.arguments_sha256 != pending.binding.arguments_sha256
            || proof.authenticated_subject_sha256 != pending.binding.authenticated_subject_sha256
        {
            return Err(QualificationError::Internal);
        }
        proof.qualification = Some(pending.binding);
        Ok(proof)
    }

    async fn require_owner(&self, user_id: &str, conversation_id: &str) -> Result<(), QualificationError> {
        match self.conversations.owner_user_id(conversation_id).await {
            Ok(Some(owner)) if owner == user_id => Ok(()),
            Ok(_) => Err(QualificationError::Unauthorized),
            Err(_) => Err(QualificationError::Internal),
        }
    }
}

struct BackendIdentity {
    pid: u32,
    start_time: u64,
    executable_sha256: String,
    runtime_id_sha256: String,
}

fn backend_identity() -> Result<BackendIdentity, QualificationError> {
    let pid = std::process::id();
    let start_time = read_process_start_time(pid).ok_or(QualificationError::Internal)?;
    let exe = std::env::current_exe().map_err(|_| QualificationError::Internal)?;
    let bytes = std::fs::read(&exe).map_err(|_| QualificationError::Internal)?;
    let executable_sha256 = sha256(&bytes);
    let runtime_id_sha256 = sha256(format!("{pid}:{start_time}:{}", executable_sha256).as_bytes());
    Ok(BackendIdentity {
        pid,
        start_time,
        executable_sha256,
        runtime_id_sha256,
    })
}

fn require_same_backend(binding: &McpQualificationBinding) -> Result<(), QualificationError> {
    let identity = backend_identity()?;
    if identity.pid != binding.backend_pid
        || identity.start_time != binding.backend_start_time
        || identity.executable_sha256 != binding.backend_executable_sha256
        || identity.runtime_id_sha256 != binding.runtime_id_sha256
    {
        return Err(QualificationError::Internal);
    }
    Ok(())
}

fn qualification_binding_sha256(binding: &McpQualificationBinding) -> String {
    let bytes = serde_json::to_vec(&serde_json::json!({
        "arguments_sha256": binding.arguments_sha256,
        "authenticated_subject_sha256": binding.authenticated_subject_sha256,
        "backend_executable_sha256": binding.backend_executable_sha256,
        "backend_pid": binding.backend_pid,
        "backend_start_time": binding.backend_start_time,
        "capability_sha256": binding.capability_sha256,
        "conversation_sha256": binding.conversation_sha256,
        "expires_at_ms": binding.expires_at_ms,
        "issued_at_ms": binding.issued_at_ms,
        "machine_challenge_sha256": binding.machine_challenge_sha256,
        "operation": binding.operation,
        "run_id": binding.run_id,
        "runtime_id_sha256": binding.runtime_id_sha256,
        "schema_version": binding.schema_version,
    }))
    .expect("qualification binding should serialize");
    sha256(&bytes)
}

fn validate_binding_input(request: &McpQualificationRequest) -> Result<(), QualificationError> {
    for value in [
        &request.name,
        &request.tool,
        &request.run_id,
        &request.machine_challenge,
    ] {
        if value.trim().is_empty() || value.len() > MAX_BINDING_BYTES {
            return Err(QualificationError::Invalid);
        }
    }
    if !request.arguments.is_object() {
        return Err(QualificationError::Invalid);
    }
    Ok(())
}

fn now_ms() -> Result<u64, QualificationError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| QualificationError::Internal)?
        .as_millis() as u64)
}

fn sha256(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn validate_consume_context(
    pending: &PendingQualification,
    user_id: &str,
    conversation_id: &str,
    now_ms: u64,
) -> Result<(), QualificationError> {
    if pending.user_id != user_id || pending.conversation_id != conversation_id {
        return Err(QualificationError::Unauthorized);
    }
    if now_ms >= pending.binding.expires_at_ms {
        return Err(QualificationError::Expired);
    }
    Ok(())
}

fn insert_pending(
    pending: &Arc<Mutex<HashMap<String, PendingQualification>>>,
    capability: String,
    item: PendingQualification,
    now_ms: u64,
) -> Result<(), QualificationError> {
    let mut pending = pending.lock().map_err(|_| QualificationError::Internal)?;
    pending.retain(|_, entry| entry.binding.expires_at_ms > now_ms);
    if pending.len() >= MAX_PENDING_CAPABILITIES {
        return Err(QualificationError::Internal);
    }
    pending.insert(capability, item);
    Ok(())
}

fn take_pending(
    pending: &Arc<Mutex<HashMap<String, PendingQualification>>>,
    capability: &str,
) -> Result<PendingQualification, QualificationError> {
    pending
        .lock()
        .map_err(|_| QualificationError::Internal)?
        .remove(capability)
        .ok_or(QualificationError::Unauthorized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aionui_api_types::McpTransport;
    use std::thread;

    fn request() -> McpQualificationRequest {
        McpQualificationRequest {
            name: "qualification".into(),
            transport: McpTransport::Stdio {
                command: "x".into(),
                args: vec![],
                env: HashMap::new(),
            },
            tool: "read".into(),
            arguments: serde_json::json!({}),
            run_id: "run-1".into(),
            machine_challenge: "challenge-1".into(),
        }
    }

    fn pending() -> PendingQualification {
        PendingQualification {
            user_id: "user-1".into(),
            conversation_id: "conv-1".into(),
            request: request(),
            binding: McpQualificationBinding {
                schema_version: 1,
                capability_sha256: sha256(b"cap"),
                authenticated_subject_sha256: sha256(b"user-1"),
                conversation_sha256: sha256(b"conv-1"),
                runtime_id_sha256: sha256(b"runtime"),
                backend_pid: 1,
                backend_start_time: 1,
                backend_executable_sha256: sha256(b"exe"),
                operation: "tools/call:read".into(),
                arguments_sha256: sha256(b"{}"),
                run_id: "run-1".into(),
                machine_challenge_sha256: sha256(b"challenge-1"),
                issued_at_ms: 1,
                expires_at_ms: u64::MAX,
                binding_sha256: sha256(b"binding"),
            },
        }
    }

    #[test]
    fn capability_consume_is_atomic_and_single_use() {
        let store = Arc::new(Mutex::new(HashMap::from([("cap".into(), pending())])));
        let mut joins = Vec::new();
        for _ in 0..16 {
            let store = store.clone();
            joins.push(thread::spawn(move || take_pending(&store, "cap").is_ok()));
        }
        assert_eq!(joins.into_iter().map(|j| j.join().unwrap()).filter(|ok| *ok).count(), 1);
    }

    #[test]
    fn binding_input_rejects_missing_run_challenge_and_non_object_args() {
        let mut req = request();
        req.run_id.clear();
        assert!(matches!(validate_binding_input(&req), Err(QualificationError::Invalid)));
        let mut req = request();
        req.machine_challenge.clear();
        assert!(matches!(validate_binding_input(&req), Err(QualificationError::Invalid)));
        let mut req = request();
        req.arguments = serde_json::json!([]);
        assert!(matches!(validate_binding_input(&req), Err(QualificationError::Invalid)));
    }

    #[test]
    fn consume_context_rejects_cross_subject_conversation_and_expiry() {
        let item = pending();
        assert!(matches!(
            validate_consume_context(&item, "other", "conv-1", 2),
            Err(QualificationError::Unauthorized)
        ));
        assert!(matches!(
            validate_consume_context(&item, "user-1", "other", 2),
            Err(QualificationError::Unauthorized)
        ));
        let mut expired = pending();
        expired.binding.expires_at_ms = 10;
        assert!(matches!(
            validate_consume_context(&expired, "user-1", "conv-1", 10),
            Err(QualificationError::Expired)
        ));
    }

    #[test]
    fn pending_store_enforces_capacity_after_exact_expiry_cleanup() {
        let store = Arc::new(Mutex::new(HashMap::new()));
        let now = 10;
        for index in 0..MAX_PENDING_CAPABILITIES {
            let mut item = pending();
            item.binding.expires_at_ms = now + 1;
            assert!(insert_pending(&store, format!("cap-{index}"), item, now).is_ok());
        }
        assert_eq!(store.lock().unwrap().len(), MAX_PENDING_CAPABILITIES);

        let mut overflow = pending();
        overflow.binding.expires_at_ms = now + 1;
        assert!(matches!(
            insert_pending(&store, "cap-overflow".into(), overflow, now),
            Err(QualificationError::Internal)
        ));

        for item in store.lock().unwrap().values_mut() {
            item.binding.expires_at_ms = now;
        }
        let mut replacement = pending();
        replacement.binding.expires_at_ms = now + 1;
        assert!(insert_pending(&store, "cap-replacement".into(), replacement, now).is_ok());
        let store = store.lock().unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.contains_key("cap-replacement"));
    }

    #[test]
    fn backend_identity_is_stable_for_current_process() {
        let first = backend_identity().unwrap();
        let second = backend_identity().unwrap();
        assert_eq!(first.pid, second.pid);
        assert_eq!(first.start_time, second.start_time);
        assert_eq!(first.executable_sha256, second.executable_sha256);
        assert_eq!(first.runtime_id_sha256, second.runtime_id_sha256);
    }
}
