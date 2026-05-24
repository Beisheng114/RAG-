"""
优化后的上下文管理模块

核心优化：
1. 每3次对话自动总结并缓存
2. 近2次对话直接添加
3. 本地缓存管理
4. 对话结束后总结加速
"""
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConversationSummary:
    """对话摘要"""
    summary_id: str
    timestamp: str
    keywords: List[str]
    summary: str
    key_topics: List[str]
    message_count: int
    start_idx: int
    end_idx: int


@dataclass
class ContextWindow:
    """上下文窗口"""
    recent_messages: List[Dict]
    summary: Optional[str]
    keywords: List[str]
    total_tokens: int
    cached_summaries: List[str]


class ContextManager:
    """
    优化的上下文管理器
    
    核心特性：
    1. 每3次对话自动总结并缓存
    2. 保留最近2次对话的完整内容
    3. 本地缓存管理，支持持久化
    4. 智能上下文压缩
    """
    
    def __init__(self, 
                 embedding_model_path: str = "./models/bge-small-zh-v1.5",
                 cache_dir: str = "./context_cache",
                 summary_interval: int = 5,
                 keep_recent_messages: int = 2):
        """
        初始化优化的上下文管理器
        
        Args:
            embedding_model_path: 本地embedding模型路径
            cache_dir: 缓存目录
            summary_interval: 总结间隔（默认3次对话）
            keep_recent_messages: 保留的最近消息数（默认2条）
        """
        try:
            self.embedding_model = SentenceTransformer(embedding_model_path)
            logger.info(f"上下文管理器初始化完成，模型: {embedding_model_path}")
        except Exception as e:
            logger.error(f"初始化embedding模型失败: {e}")
            self.embedding_model = None
        
        self.cache_dir = cache_dir
        self.summary_interval = summary_interval
        self.keep_recent_messages = keep_recent_messages
        
        self.max_context_tokens = 2048
        self.keyword_threshold = 0.4
        
        self.conversation_history: List[Dict] = []
        self.cached_summaries: List[ConversationSummary] = []
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self._setup_cache_dir()
        self._load_cached_summaries()
        
        logger.info(f"优化上下文管理器初始化完成 - 总结间隔: {summary_interval}次, 保留消息: {keep_recent_messages}条")
    
    def _setup_cache_dir(self):
        """设置缓存目录"""
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"缓存目录设置完成: {self.cache_dir}")
    
    def _load_cached_summaries(self):
        """加载缓存的摘要"""
        cache_file = os.path.join(self.cache_dir, "conversation_summaries.pkl")
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.cached_summaries = pickle.load(f)
                logger.info(f"加载了 {len(self.cached_summaries)} 个缓存的摘要")
            except Exception as e:
                logger.error(f"加载缓存摘要失败: {e}")
                self.cached_summaries = []
        else:
            self.cached_summaries = []
    
    def _save_cached_summaries(self):
        """保存摘要到缓存"""
        cache_file = os.path.join(self.cache_dir, "conversation_summaries.pkl")
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cached_summaries, f)
            logger.info(f"保存了 {len(self.cached_summaries)} 个摘要到缓存")
        except Exception as e:
            logger.error(f"保存缓存摘要失败: {e}")
    
    def add_message(self, role: str, content: str) -> Dict[str, Any]:
        """
        添加消息并触发总结机制
        
        Args:
            role: 消息角色（user/assistant）
            content: 消息内容
            
        Returns:
            处理结果信息
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.conversation_history.append(message)
        
        result = {
            "message_added": True,
            "total_messages": len(self.conversation_history),
            "should_summarize": False,
            "summary": None
        }
        
        if role == "user":
            user_message_count = sum(1 for msg in self.conversation_history if msg["role"] == "user")
            
            if user_message_count % self.summary_interval == 0:
                logger.info(f"达到总结间隔 ({self.summary_interval}次)，触发总结")
                summary = self._create_and_cache_summary()
                result["should_summarize"] = True
                result["summary"] = summary
        
        return result
    
    def _create_and_cache_summary(self) -> ConversationSummary:
        """
        创建并缓存对话摘要
        
        Returns:
            对话摘要对象
        """
        if len(self.conversation_history) < self.summary_interval:
            logger.warning("对话历史不足，无法生成摘要")
            return None
        
        messages_to_summarize = self.conversation_history[:-self.keep_recent_messages]
        
        if not messages_to_summarize:
            logger.warning("没有可总结的消息")
            return None
        
        keywords = self._extract_keywords_from_messages(messages_to_summarize)
        summary_text = self._generate_summary_text(messages_to_summarize, keywords)
        key_topics = self._extract_key_topics(messages_to_summarize)
        
        summary = ConversationSummary(
            summary_id=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.cached_summaries)}",
            timestamp=datetime.now().isoformat(),
            keywords=keywords,
            summary=summary_text,
            key_topics=key_topics,
            message_count=len(messages_to_summarize),
            start_idx=0,
            end_idx=len(messages_to_summarize) - 1
        )
        
        self.cached_summaries.append(summary)
        self._save_cached_summaries()
        
        logger.info(f"创建并缓存摘要: {summary.summary_id}")
        return summary
    
    def _extract_keywords_from_messages(self, messages: List[Dict], top_k: int = 5) -> List[str]:
        """从消息中提取关键词"""
        if not self.embedding_model:
            return self._simple_keyword_extraction(messages, top_k)
        
        try:
            all_text = " ".join([msg["content"] for msg in messages if msg["role"] == "user"])
            
            sentences = [s.strip() for s in all_text.split('。') if s.strip()]
            if not sentences:
                return []
            
            sentence_embeddings = self.embedding_model.encode(sentences)
            text_embedding = self.embedding_model.encode([all_text])
            
            similarities = np.dot(sentence_embeddings, text_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            keywords = [sentences[i] for i in top_indices]
            keywords = [self._extract_key_phrase(kw) for kw in keywords]
            keywords = [kw for kw in keywords if kw and len(kw) > 2]
            
            return keywords[:top_k]
            
        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            return self._simple_keyword_extraction(messages, top_k)
    
    def _extract_key_phrase(self, sentence: str) -> str:
        """从句子中提取关键短语"""
        words = sentence.split()
        if len(words) <= 2:
            return sentence
        return ' '.join(words[:3])
    
    def _simple_keyword_extraction(self, messages: List[Dict], top_k: int = 5) -> List[str]:
        """简单的关键词提取"""
        all_text = " ".join([msg["content"] for msg in messages if msg["role"] == "user"])
        words = all_text.split()
        word_freq = {}
        
        for word in words:
            if len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_k]]
    
    def _generate_summary_text(self, messages: List[Dict], keywords: List[str]) -> str:
        """生成摘要文本"""
        user_messages = [msg["content"] for msg in messages if msg["role"] == "user"]
        
        if not user_messages:
            return "无用户消息"
        
        summary_parts = []
        
        if keywords:
            summary_parts.append(f"讨论主题: {', '.join(keywords[:3])}")
        
        if len(user_messages) > 0:
            first_question = user_messages[0][:50] + "..." if len(user_messages[0]) > 50 else user_messages[0]
            summary_parts.append(f"首个问题: {first_question}")
        
        if len(user_messages) > 1:
            last_question = user_messages[-1][:50] + "..." if len(user_messages[-1]) > 50 else user_messages[-1]
            summary_parts.append(f"最后问题: {last_question}")
        
        summary_parts.append(f"共 {len(user_messages)} 个问题")
        
        return " | ".join(summary_parts)
    
    def _extract_key_topics(self, messages: List[Dict]) -> List[str]:
        """提取关键主题"""
        topics = []
        
        topic_keywords = {
            "故障": "故障诊断",
            "维修": "维修操作",
            "检查": "检查流程",
            "更换": "部件更换",
            "调试": "设备调试",
            "安全": "安全注意事项",
            "主机": "主机系统",
            "发电机": "发电设备",
            "冷却": "冷却系统",
            "润滑": "润滑系统"
        }
        
        all_text = " ".join([msg["content"] for msg in messages])
        
        for keyword, topic in topic_keywords.items():
            if keyword in all_text:
                topics.append(topic)
        
        return list(set(topics))[:5]
    
    def build_context_for_query(self, current_question: str) -> ContextWindow:
        """
        构建查询上下文
        
        Args:
            current_question: 当前问题
            
        Returns:
            上下文窗口对象
        """
        recent_messages = self.conversation_history[-self.keep_recent_messages:] if len(self.conversation_history) > self.keep_recent_messages else self.conversation_history
        
        summary_texts = [s.summary for s in self.cached_summaries[-3:]] if self.cached_summaries else []
        
        all_keywords = []
        for summary in self.cached_summaries[-3:]:
            all_keywords.extend(summary.keywords)
        all_keywords = list(set(all_keywords))[:5]
        
        total_tokens = self._estimate_tokens(recent_messages)
        
        return ContextWindow(
            recent_messages=recent_messages,
            summary=" | ".join(summary_texts) if summary_texts else None,
            keywords=all_keywords,
            total_tokens=total_tokens,
            cached_summaries=summary_texts
        )
    
    def format_context_for_llm(self, context: ContextWindow, current_question: str) -> str:
        """
        格式化上下文用于LLM
        
        Args:
            context: 上下文窗口
            current_question: 当前问题
            
        Returns:
            格式化的上下文字符串
        """
        context_parts = []
        
        if context.cached_summaries:
            context_parts.append("【历史对话摘要】")
            for i, summary in enumerate(context.cached_summaries, 1):
                context_parts.append(f"{i}. {summary}")
        
        if context.keywords:
            context_parts.append(f"\n【关键主题】: {', '.join(context.keywords)}")
        
        if context.recent_messages:
            context_parts.append("\n【最近对话】")
            for msg in context.recent_messages:
                role = "用户" if msg["role"] == "user" else "助手"
                content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
                context_parts.append(f"{role}: {content}")
        
        context_parts.append(f"\n【当前问题】: {current_question}")
        
        return "\n".join(context_parts)
    
    def _estimate_tokens(self, messages: List[Dict]) -> int:
        """估算token数"""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        return int(total_chars * 0.7)
    
    def generate_final_summary(self) -> str:
        """
        生成对话结束后的最终总结
        
        Returns:
            最终总结字符串
        """
        if not self.conversation_history:
            return "无对话历史"
        
        all_keywords = []
        all_topics = []
        
        for summary in self.cached_summaries:
            all_keywords.extend(summary.keywords)
            all_topics.extend(summary.key_topics)
        
        all_keywords = list(set(all_keywords))[:10]
        all_topics = list(set(all_topics))[:5]
        
        user_message_count = sum(1 for msg in self.conversation_history if msg["role"] == "user")
        
        final_summary = f"""
对话总结报告
==================
对话时间: {self.current_session_id}
总消息数: {len(self.conversation_history)}
用户提问: {user_message_count} 次
关键主题: {', '.join(all_topics) if all_topics else '无'}
关键词: {', '.join(all_keywords) if all_keywords else '无'}
摘要次数: {len(self.cached_summaries)}
"""
        
        return final_summary
    
    def clear_session(self):
        """清空当前会话"""
        self.conversation_history = []
        self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("会话已清空")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": sum(1 for msg in self.conversation_history if msg["role"] == "user"),
            "assistant_messages": sum(1 for msg in self.conversation_history if msg["role"] == "assistant"),
            "cached_summaries": len(self.cached_summaries),
            "current_session_id": self.current_session_id,
            "summary_interval": self.summary_interval,
            "keep_recent_messages": self.keep_recent_messages
        }
    
    def export_session(self, filepath: Optional[str] = None) -> str:
        """
        导出当前会话
        
        Args:
            filepath: 导出文件路径（可选）
            
        Returns:
            导出文件路径
        """
        if filepath is None:
            filepath = os.path.join(self.cache_dir, f"session_{self.current_session_id}.json")
        
        session_data = {
            "session_id": self.current_session_id,
            "conversation_history": self.conversation_history,
            "cached_summaries": [asdict(s) for s in self.cached_summaries],
            "statistics": self.get_statistics()
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"会话已导出到: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"导出会话失败: {e}")
            return ""
    
    def import_session(self, filepath: str) -> bool:
        """
        导入会话
        
        Args:
            filepath: 会话文件路径
            
        Returns:
            是否导入成功
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            self.current_session_id = session_data["session_id"]
            self.conversation_history = session_data["conversation_history"]
            
            self.cached_summaries = [
                ConversationSummary(**s) for s in session_data["cached_summaries"]
            ]
            
            logger.info(f"会话已导入: {filepath}")
            return True
        except Exception as e:
            logger.error(f"导入会话失败: {e}")
            return False
    
    def build_context_window(self, messages: List[Dict], max_tokens: Optional[int] = None) -> ContextWindow:
        """
        构建上下文窗口（兼容旧接口）
        
        Args:
            messages: 消息列表（此参数在优化版本中被忽略，使用内部历史）
            max_tokens: 最大token数（可选）
            
        Returns:
            上下文窗口对象
        """
        if messages:
            self.conversation_history = messages
        
        if not self.conversation_history:
            return ContextWindow(
                recent_messages=[],
                summary=None,
                keywords=[],
                total_tokens=0,
                cached_summaries=[]
            )
        
        recent_messages = self.conversation_history[-self.keep_recent_messages:] if len(self.conversation_history) > self.keep_recent_messages else self.conversation_history
        
        summary_texts = [s.summary for s in self.cached_summaries[-3:]] if self.cached_summaries else []
        
        all_keywords = []
        for summary in self.cached_summaries[-3:]:
            all_keywords.extend(summary.keywords)
        all_keywords = list(set(all_keywords))[:5]
        
        total_tokens = self._estimate_tokens(recent_messages)
        
        return ContextWindow(
            recent_messages=recent_messages,
            summary=" | ".join(summary_texts) if summary_texts else None,
            keywords=all_keywords,
            total_tokens=total_tokens,
            cached_summaries=summary_texts
        )
    
    def format_context_for_query(self, context: ContextWindow, current_question: str) -> str:
        """
        格式化上下文用于查询（兼容旧接口）
        
        Args:
            context: 上下文窗口
            current_question: 当前问题
            
        Returns:
            格式化的上下文字符串
        """
        return self.format_context_for_llm(context, current_question)
