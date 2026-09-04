"""
生成集成模块

动态计算max_tokens以适应模型上下文长度限制
"""
import logging
import os
import time
import json
from typing import List, Optional, Dict
import requests

from openai import OpenAI
from langchain_core.documents import Document
from .context_manager import ContextManager


logger = logging.getLogger(__name__)


class GenerationIntegrationModule:
    """生成集成模块 - 负责答案生成"""
    
    MODEL_MAX_CONTEXT = 8192
    DEFAULT_MAX_TOKENS = 1024
    MIN_MAX_TOKENS = 512
    
    def __init__(self, model_name: str = "/models/Qwen3-4B-Instruct-2507-AWQ", temperature: float = 0.3, max_tokens: int = 2048, embedding_model_path: str = "./models/bge-small-zh-v1.5", llm_provider: str = "vllm", ollama_base_url: str = "http://localhost:11434", ollama_model: str = "qwen2.5:7b", vllm_base_url: str = "http://127.0.0.1:8000/v1"):
        """
        初始化生成集成模块
        
        Args:
            model_name: 模型名称（vLLM使用）
            temperature: 温度参数
            max_tokens: 最大token数
            embedding_model_path: 嵌入模型路径
            llm_provider: LLM提供者，可选 "vllm" 或 "ollama"
            ollama_base_url: Ollama的base_url（不包含/v1）
            ollama_model: Ollama中的模型名称
            vllm_base_url: vLLM的base_url
        """ 
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # 初始化上下文管理器
        self.context_manager = ContextManager(embedding_model_path)
        
        # 根据LLM提供者选择不同的客户端
        if llm_provider == "ollama":
            self.ollama_base_url = ollama_base_url.rstrip('/')
            self.ollama_model = ollama_model
            self.client = None  # Ollama使用原生API，不使用OpenAI客户端
            logger.info(f"使用Ollama原生API: {self.ollama_base_url}, 模型: {ollama_model}")
        else:
            base_url = vllm_base_url
            self.model_name = model_name
            self.client = OpenAI(api_key='dummy',
                                base_url=base_url,
                                timeout=30.0)
            logger.info(f"使用vLLM OpenAI兼容API: {base_url}, 模型: {model_name}")

        logger.info(f"生成模块初始化完成，LLM提供者: {llm_provider}")

    def _calculate_max_tokens(self, prompt: str) -> int:
        """
        动态计算max_tokens，确保不超过模型上下文限制
        
        Args:
            prompt: 输入提示词
            
        Returns:
            计算后的max_tokens值
        """
        estimated_input_tokens = int(len(prompt) * 0.7)
        
        available_tokens = self.MODEL_MAX_CONTEXT - estimated_input_tokens - 100
        
        if available_tokens < self.MIN_MAX_TOKENS:
            available_tokens = self.MIN_MAX_TOKENS
        
        max_output_tokens = min(available_tokens, self.max_tokens)
        
        logger.info(f"输入估计tokens: {estimated_input_tokens}, 可用tokens: {available_tokens}, 输出max_tokens: {max_output_tokens}")
        
        return max_output_tokens

    def generate_adaptive_answer(self, question: str, documents: List[Document], conversation_history: Optional[List[Dict]] = None) -> str:
        """
        智能统一答案生成
        自动适应不同类型的查询，无需预先分类
        Args:
            question: 用户问题
            documents: 检索到的文档
            conversation_history: 对话历史（可选）
            
        Returns:
            生成的回答
        """
        context_parts = []
        all_sources = []

        for idx, doc in enumerate(documents):
            content = doc.page_content.strip()
            if content:
                sources = doc.metadata.get('sources', [])
                if sources:
                    source_strs = []
                    for s in sources:
                        src = f"{s.get('name', '未知')}"
                        if s.get('chapter'):
                            src += f" {s.get('chapter')}"
                        if s.get('section'):
                            src += f" {s.get('section')}"
                        source_strs.append(src)
                    source_label = f"[来源: {'; '.join(source_strs)}]"
                    context_parts.append(f"{content}\n{source_label}")
                    all_sources.extend(sources)
                else:
                    level = doc.metadata.get('retrieval_level', '')
                    if level:
                        context_parts.append(f"[{level.upper()}] {content}")
                    else:
                        context_parts.append(content)

        context = "\n\n".join(context_parts)
        
        unique_sources = []
        seen = set()
        for s in all_sources:
            key = s.get('source_id', '')
            if key and key not in seen:
                seen.add(key)
                unique_sources.append(s)
        
        source_info = ""
        if unique_sources:
            source_lines = []
            for s in unique_sources:
                src_line = f"- {s.get('name', '未知来源')} (ID: {s.get('source_id', 'N/A')})"
                if s.get('chapter'):
                    src_line += f", 第{s.get('chapter')}章"
                if s.get('section'):
                    src_line += f" 第{s.get('section')}节"
                if s.get('type'):
                    src_line += f", 类型: {s.get('type')}"
                if s.get('reliability'):
                    src_line += f", 可靠性: {s.get('reliability')}"
                source_lines.append(src_line)
            source_info = "\n\n## 参考来源:\n" + "\n".join(source_lines)
        
        # 处理对话历史
        context_info = ""
        if conversation_history:
            context_window = self.context_manager.build_context_window(conversation_history)
            context_info = self.context_manager.format_context_for_query(context_window, question)
            logger.info(f"使用对话上下文: {len(conversation_history)}条消息, 关键词: {context_window.keywords}")

        prompt = f"""
        作为一位专业的船舶维修工程师，请基于以下信息回答用户的问题。
        {context_info}
        检索到的相关信息：
        {context}
        {source_info}
        
        用户问题：{question}
        
        请提供准确、实用的回答。根据问题的性质：
        - 如果是询问多个问题，请提供清晰的列表
        - 若问题涉及多种原因、多因素或综合故障，请逐条列出各原因及对应依据，不要合并成单一原因
        - 如果是询问具体维修方法，请提供详细步骤
        - 如果是一般性咨询，请提供综合性回答
        - 请严格遵守船舶维修相关的法律和规定
        - 如果使用了相关的检索信息，请在回答中用括号引用来源
        - 如果有对话历史，请参考历史上下文进行回答
        - **重要**：请在回答中尽可能引用上述参考来源信息，包括文档名称、章节等
        - 如果没有足够的信息来回答问题，请说“根据检索到的信息，我无法回答这个问题。”
        回答：
        """

        calculated_max_tokens = self._calculate_max_tokens(prompt)

        try:
            if self.llm_provider == "ollama":
                # 使用Ollama原生API
                url = f"{self.ollama_base_url}/api/chat"
                payload = {
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": calculated_max_tokens
                    }
                }
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result["message"]["content"].strip()
            else:
                # 使用vLLM OpenAI兼容API
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=calculated_max_tokens
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LightRAG答案生成失败: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"

    def rewrite_query(self, question: str, conversation_history: Optional[List[Dict]] = None, max_history_turns: int = 3) -> str:
        """
        多轮对话查询改写（指代消解），用于检索前的查询补全

        将"它怎么修？""这个设备还有什么故障？"等依赖上文的问题改写为
        独立完整的问题，避免直接拿原文检索导致召回丢失。

        任何失败（LLM不可用/超时/输出异常）均降级返回原问题，不影响主链路。

        Args:
            question: 当前用户问题
            conversation_history: 对话历史 [{"role": "user"/"assistant", "content": ...}, ...]
            max_history_turns: 参与改写的最近历史轮数

        Returns:
            改写后的独立问题；无需改写或失败时返回原问题
        """
        if not conversation_history:
            return question

        # 取最近的几轮历史
        recent = conversation_history[-max_history_turns * 2:]
        history_lines = []
        for msg in recent:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))[:300]
            history_lines.append(f"{'用户' if role == 'user' else '助手'}: {content}")
        history_text = "\n".join(history_lines)

        prompt = f"""以下是一段关于船舶故障维修的多轮对话历史：

{history_text}

用户的新问题是："{question}"

请把新问题改写成一个不依赖对话历史、可以独立理解的完整问题（指代消解）。
要求：
- 只输出改写后的问题本身，不要任何解释、前缀或引号
- 保留原问题中的设备/系统/部件等关键实体
- 如果新问题本身已经独立完整，直接原样输出该问题
改写后的问题："""

        try:
            if self.llm_provider == "ollama":
                url = f"{self.ollama_base_url}/api/chat"
                payload = {
                    "model": self.ollama_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 256}
                }
                response = requests.post(url, json=payload, timeout=15)
                response.raise_for_status()
                rewritten = response.json()["message"]["content"].strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=256
                )
                rewritten = response.choices[0].message.content.strip()

            # 清洗：取首行、去包裹引号
            rewritten = rewritten.splitlines()[0].strip().strip('"“”').strip()

            # 合理性校验：长度异常视为改写失败
            if not rewritten or len(rewritten) > 200:
                return question
            logger.info(f"查询改写: {question} -> {rewritten}")
            return rewritten
        except Exception as e:
            logger.warning(f"查询改写失败，使用原问题检索: {e}")
            return question

    def generate_adaptive_answer_stream(self, question: str, documents: List[Document], conversation_history: Optional[List[Dict]] = None, max_retries: int = 3):
        """
        LightRAG风格的流式答案生成（带重试机制）
        
        Args:
            question: 用户问题
            documents: 检索到的文档
            conversation_history: 对话历史（可选）
            max_retries: 最大重试次数
            
        Yields:
            生成的文本片段
        """
        context_parts = []
        all_sources = []

        for doc in documents:
            content = doc.page_content.strip()
            if content:
                sources = doc.metadata.get('sources', [])
                if sources:
                    source_strs = []
                    for s in sources:
                        src = f"{s.get('name', '未知')}"
                        if s.get('chapter'):
                            src += f" {s.get('chapter')}"
                        if s.get('section'):
                            src += f" {s.get('section')}"
                        source_strs.append(src)
                    source_label = f"[来源: {'; '.join(source_strs)}]"
                    context_parts.append(f"{content}\n{source_label}")
                    all_sources.extend(sources)
                else:
                    level = doc.metadata.get('retrieval_level', '')
                    if level:
                        context_parts.append(f"[{level.upper()}] {content}")
                    else:
                        context_parts.append(content)

        context = "\n\n".join(context_parts)
        
        unique_sources = []
        seen = set()
        for s in all_sources:
            key = s.get('source_id', '')
            if key and key not in seen:
                seen.add(key)
                unique_sources.append(s)
        
        source_info = ""
        if unique_sources:
            source_lines = []
            for s in unique_sources:
                src_line = f"- {s.get('name', '未知来源')} (ID: {s.get('source_id', 'N/A')})"
                if s.get('chapter'):
                    src_line += f", 第{s.get('chapter')}章"
                if s.get('section'):
                    src_line += f" 第{s.get('section')}节"
                if s.get('type'):
                    src_line += f", 类型: {s.get('type')}"
                if s.get('reliability'):
                    src_line += f", 可靠性: {s.get('reliability')}"
                source_lines.append(src_line)
            source_info = "\n\n## 参考来源:\n" + "\n".join(source_lines)
        
        # 处理对话历史
        context_info = ""
        if conversation_history:
            context_window = self.context_manager.build_context_window(conversation_history)
            context_info = self.context_manager.format_context_for_query(context_window, question)
            logger.info(f"使用对话上下文: {len(conversation_history)}条消息, 关键词: {context_window.keywords}")

        prompt = f"""
        作为一位专业的船舶维修工程师，请基于以下信息回答用户的问题。
        
        {context_info}
        
        检索到的相关信息：
        {context}
        {source_info}
        
        用户问题：{question}
        
        请提供准确、实用的回答。根据问题的性质：
        - 如果是询问多个问题，请提供清晰的列表
        - 若问题涉及多种原因、多因素或综合故障，请逐条列出各原因及对应依据，不要合并成单一原因
        - 如果是询问具体维修方法，请提供详细步骤
        - 如果是一般性咨询，请提供综合性回答
        - 如果有对话历史，请参考历史上下文进行回答
        - **重要**：请在回答中尽可能引用上述参考来源信息，包括文档名称、章节等
        - 请严格按照提供的信息进行回答，不能编造或假设信息
        回答：
        """

        calculated_max_tokens = self._calculate_max_tokens(prompt)

        for attempt in range(max_retries):
            try:
                if self.llm_provider == "ollama":
                    # 使用Ollama原生流式API
                    url = f"{self.ollama_base_url}/api/chat"
                    payload = {
                        "model": self.ollama_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": True,
                        "options": {
                            "temperature": self.temperature,
                            "num_predict": calculated_max_tokens
                        }
                    }
                    response = requests.post(url, json=payload, stream=True, timeout=60)
                    response.raise_for_status()

                    if attempt == 0:
                        print("开始流式生成回答...\n")
                    else:
                        print(f"第{attempt + 1}次尝试流式生成...\n")

                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                data = line.decode('utf-8')
                                chunk = json.loads(data)
                                if not chunk.get("done", False) and "message" in chunk:
                                    content = chunk["message"].get("content", "")
                                    if content:
                                        full_response += content
                                        yield content
                            except json.JSONDecodeError:
                                continue
                    return
                else:
                    # 使用vLLM OpenAI兼容流式API
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=calculated_max_tokens,
                        stream=True,
                        timeout=60
                    )

                    if attempt == 0:
                        print("开始流式生成回答...\n")
                    else:
                        print(f"第{attempt + 1}次尝试流式生成...\n")

                    full_response = ""
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            yield content
                    return

            except Exception as e:
                logger.warning(f"流式生成第{attempt + 1}次尝试失败: {e}")

                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"⚠️ 连接中断，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"流式生成完全失败，尝试非流式后备方案")
                    print("⚠️ 流式生成失败，切换到标准模式...")

                    try:
                        fallback_response = self.generate_adaptive_answer(question, documents)
                        yield fallback_response
                        return
                    except Exception as fallback_error:
                        logger.error(f"后备生成也失败: {fallback_error}")
                        error_msg = f"抱歉，生成回答时出现网络错误，请稍后重试。错误信息：{str(e)}"
                        yield error_msg
                        return 