use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
/// Token 精确计费模块
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

/// Token 计数记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub session_id: String,
    pub model_id: String,
    pub user_id: Option<String>,
    pub tokens_generated: u64,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
}

/// Token 计数器（线程安全）
#[derive(Clone)]
pub struct TokenCounter {
    sessions: Arc<Mutex<HashMap<String, SessionCounter>>>,
}

#[derive(Debug)]
struct SessionCounter {
    session_id: String,
    model_id: String,
    user_id: Option<String>,
    token_count: u64,
    start_time: DateTime<Utc>,
}

impl TokenCounter {
    /// 创建新的计数器
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// 开始新的会话
    pub fn start_session(&self, model_id: String, user_id: Option<String>) -> String {
        let session_id = Uuid::new_v4().to_string();
        let mut sessions = self.sessions.lock().unwrap();

        sessions.insert(
            session_id.clone(),
            SessionCounter {
                session_id: session_id.clone(),
                model_id,
                user_id,
                token_count: 0,
                start_time: Utc::now(),
            },
        );

        session_id
    }

    /// 增加 Token 计数
    pub fn add_tokens(&self, session_id: &str, count: u64) {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(session) = sessions.get_mut(session_id) {
            session.token_count += count;
        }
    }

    /// 结束会话并返回统计
    pub fn end_session(&self, session_id: &str) -> Option<TokenUsage> {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.remove(session_id).map(|session| TokenUsage {
            session_id: session.session_id,
            model_id: session.model_id,
            user_id: session.user_id,
            tokens_generated: session.token_count,
            start_time: session.start_time,
            end_time: Utc::now(),
        })
    }

    /// 获取会话当前统计
    pub fn get_session_stats(&self, session_id: &str) -> Option<(String, u64)> {
        let sessions = self.sessions.lock().unwrap();
        sessions
            .get(session_id)
            .map(|s| (s.model_id.clone(), s.token_count))
    }

    /// 清理过期会话（超过 1 小时未更新）
    pub fn cleanup_stale_sessions(&self) {
        let mut sessions = self.sessions.lock().unwrap();
        let now = Utc::now();
        sessions.retain(|_, session| (now - session.start_time).num_hours() < 1);
    }
}

impl Default for TokenCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_counting() {
        let counter = TokenCounter::new();

        let session_id = counter.start_session("qwen-7b".to_string(), Some("user123".to_string()));

        counter.add_tokens(&session_id, 10);
        counter.add_tokens(&session_id, 20);
        counter.add_tokens(&session_id, 30);

        let stats = counter.get_session_stats(&session_id).unwrap();
        assert_eq!(stats.0, "qwen-7b");
        assert_eq!(stats.1, 60);

        let usage = counter.end_session(&session_id).unwrap();
        assert_eq!(usage.tokens_generated, 60);
        assert_eq!(usage.model_id, "qwen-7b");
    }

    #[test]
    fn test_multiple_sessions() {
        let counter = TokenCounter::new();

        let session1 = counter.start_session("model-a".to_string(), None);
        let session2 = counter.start_session("model-b".to_string(), None);

        counter.add_tokens(&session1, 100);
        counter.add_tokens(&session2, 200);

        assert_eq!(counter.get_session_stats(&session1).unwrap().1, 100);
        assert_eq!(counter.get_session_stats(&session2).unwrap().1, 200);
    }
}
