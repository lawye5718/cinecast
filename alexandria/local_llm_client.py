#!/usr/bin/env python3
"""
本地化LLM客户端 - 集成CineCast中测试通过的Ollama Qwen14B-Pro模型
"""

import json
import requests
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class LocalLLMClient:
    """本地Ollama Qwen14B-Pro模型客户端"""
    
    def __init__(self, config: Dict):
        self.config = config.get("llm", {})
        self.api_url = self.config.get("api_url", "http://localhost:11434/api/chat")
        self.model_name = self.config.get("model", "qwen14b-pro")
        self.temperature = self.config.get("temperature", 0.0)
        self.num_ctx = self.config.get("num_ctx", 8192)
        
        # 验证Ollama服务连接
        self._check_connection()
    
    def _check_connection(self) -> bool:
        """检查Ollama服务连接状态"""
        try:
            response = requests.get(f"{self.api_url.replace('/api/chat', '')}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Ollama服务连接正常")
                return True
            else:
                logger.warning("❌ Ollama服务响应异常")
                return False
        except Exception as e:
            logger.warning(f"❌ 无法连接到Ollama服务: {e}")
            return False
    
    def generate_script(self, text_chunk: str, context: str = "") -> List[Dict]:
        """生成有声书剧本 - 使用CineCast中验证的System Prompt
        
        🌟 修改2：强制按字符数将超长文本切分为 1500 字左右的块
        这样能保证大模型每次输出的 JSON 不会超过 2000 个 Token，从而在合理时间内完成
        """
        
        # 🌟 如果文本过长，进行智能分块处理
        if len(text_chunk) > 2000:
            logger.info(f"📝 检测到长文本 ({len(text_chunk)} 字符)，进行智能分块处理...")
            return self._process_long_text_with_chunking(text_chunk, context)
        
        # 短文本直接处理
        return self._process_single_chunk(text_chunk, context)
        system_prompt = """
你是一位顶级的有声书导演兼数据清洗专家，负责将原始小说文本转换为标准化的录音剧本。
你必须严格遵守以下四大纪律，任何违反都将导致系统崩溃：

【一、 绝对忠实原则（Iron Rule）】
- 必须 100% 逐字保留原文内容！
- 严禁任何形式的概括、改写、缩写、续写或润色！
- 严禁自行添加原文中不存在的台词或动作描写！

【二、 字符净化原则】
- 剔除所有不可发音的特殊符号（如 Emoji表情、Markdown标记 * _ ~ #、制表符 \t、不可见控制字符）。
- 仅保留基础标点符号（，。！？：；、“”‘’（））。
- 数字、英文字母允许保留，但禁止出现复杂的数学公式符号。

【三、 粒度拆分原则】
- 必须将"对白"和"旁白/动作描写"严格剥离为独立的对象！
- 例如原文："你好，"老渔夫笑着说。
  必须拆分为两个对象：1. 角色对白("你好，") 2. 旁白描述("老渔夫笑着说。")

【四、 JSON 格式规范】
必须且只能输出合法的 JSON 数组，禁止任何解释性前言或后缀（如"好的，以下是..."），禁止输出 Markdown 代码块标记（```json）。
数组元素字段要求：
- "type": 仅限 "title"(章节名), "subtitle"(小标题), "narration"(旁白), "dialogue"(对白)。
- "speaker": 对白填具体的角色名（需根据上下文推断并保持全书统一）；旁白和标题统一填 "narrator"。
- "gender": 仅限 "male"、"female" 或 "unknown"。对白请推测性别；旁白固定为 "male"。
- "emotion": 情感标签（如"平静"、"激动"、"沧桑/叹息"、"愤怒"、"悲伤"等），用于未来语音合成的情感控制。
- "content": 纯净的文本内容。如果 type 是 "dialogue"，必须去掉最外层的引号（如""或""）。

【输出格式示例（One-Shot）】
[
  {
    "type": "narration",
    "speaker": "narrator",
    "gender": "male",
    "emotion": "平静",
    "content": "夜幕降临，港口的灯火开始闪烁。"
  },
  {
    "type": "dialogue",
    "speaker": "老渔夫",
    "gender": "male",
    "emotion": "沧桑/叹息",
    "content": "你相信命运吗？"
  },
  {
    "type": "narration",
    "speaker": "narrator",
    "gender": "male",
    "emotion": "平静",
    "content": "老渔夫说道。"
  }
]
"""
        
        # 构建用户提示
        user_prompt = f"请严格按照规范，将以下文本拆解为纯净的 JSON 剧本（绝不改写原意）：\n\n{text_chunk}"
        if context:
            user_prompt = f"上下文信息：{context}\n\n{user_prompt}"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "top_p": 0.1
            }
        }
        
        # 🌟 增加最大尝试次数
        max_retries = 2 
        
        for attempt in range(max_retries + 1):
            try:
                # 🌟 修改1：将 timeout 放宽到 300 秒（5分钟）
                response = requests.post(self.api_url, json=payload, timeout=300)
                response.raise_for_status()
                content = response.json().get('message', {}).get('content', '[]')
                
                # 清理Markdown代码块
                content = self._clean_markdown(content)
                
                # 解析JSON
                script = json.loads(content)
                
                # 验证并修复数据结构
                if isinstance(script, list):
                    return self._validate_script_elements(script)
                elif isinstance(script, dict):
                    # 处理包装格式
                    for value in script.values():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            return self._validate_script_elements(value)
                
                # 如果解析成功，直接返回
                return script
                
            except requests.exceptions.Timeout:
                logger.error(f"❌ Ollama 响应超时 (尝试 {attempt+1}/{max_retries+1})，模型可能输出过长。")
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON解析失败 (尝试 {attempt+1}/{max_retries+1}): {e}")
            except Exception as e:
                logger.error(f"❌ Ollama 请求异常 (尝试 {attempt+1}/{max_retries+1}): {e}")
            
            # 如果是最后一次尝试依然失败，则使用降级方案
            if attempt == max_retries:
                logger.error("🚨 达到最大重试次数，使用正则降级方案。")
                return self._fallback_regex_parse(text_chunk)
            
            # 重试前短暂等待
            import time
            time.sleep(1)
    
    def _process_long_text_with_chunking(self, text: str, context: str = "") -> List[Dict]:
        """处理长文本的智能分块逻辑"""
        full_script = []
        
        # 🌟 强制按换行符将超长文本切分为 1500 字左右的块
        max_chunk_chars = 1500
        paragraphs = text.split('\n')
        current_chunk = ""
        chunks = []
        
        for p in paragraphs:
            if len(current_chunk) + len(p) > max_chunk_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n"
            else:
                current_chunk += p + "\n"
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # 分块交给 Ollama 处理并合并结果
        for idx, chunk in enumerate(chunks):
            if not chunk: 
                continue
            logger.info(f"  -> 正在让大模型解析文本块 {idx+1}/{len(chunks)} (字数: {len(chunk)})")
            
            # 递归调用自身处理每个小块
            script_part = self._process_single_chunk(chunk, context)
            if script_part:
                full_script.extend(script_part)
            
            # 添加小的延迟避免过于频繁的请求
            import time
            time.sleep(0.5)
        
        return full_script
    
    def _process_single_chunk(self, text_chunk: str, context: str = "") -> List[Dict]:
        """处理单个文本块的核心逻辑"""
        # 使用CineCast中测试通过的强化System Prompt
        system_prompt = """
你是一位顶级的有声书导演兼数据清洗专家，负责将原始小说文本转换为标准化的录音剧本。
你必须严格遵守以下四大纪律，任何违反都将导致系统崩溃：

【一、 绝对忠实原则（Iron Rule）】
- 必须 100% 逐字保留原文内容！
- 严禁任何形式的概括、改写、缩写、续写或润色！
- 严禁自行添加原文中不存在的台词或动作描写！

【二、 字符净化原则】
- 剔除所有不可发音的特殊符号（如 Emoji表情、Markdown标记 * _ ~ #、制表符 \t、不可见控制字符）。
- 仅保留基础标点符号（，。！？：；、“”‘’（））。
- 数字、英文字母允许保留，但禁止出现复杂的数学公式符号。

【三、 粒度拆分原则】
- 必须将"对白"和"旁白/动作描写"严格剥离为独立的对象！
- 例如原文："你好，"老渔夫笑着说。
  必须拆分为两个对象：1. 角色对白("你好，") 2. 旁白描述("老渔夫笑着说。")

【四、 JSON 格式规范】
必须且只能输出合法的 JSON 数组，禁止任何解释性前言或后缀（如"好的，以下是..."），禁止输出 Markdown 代码块标记（```json）。
数组元素字段要求：
- "type": 仅限 "title"(章节名), "subtitle"(小标题), "narration"(旁白), "dialogue"(对白)。
- "speaker": 对白填具体的角色名（需根据上下文推断并保持全书统一）；旁白和标题统一填 "narrator"。
- "gender": 仅限 "male"、"female" 或 "unknown"。对白请推测性别；旁白固定为 "male"。
- "emotion": 情感标签（如"平静"、"激动"、"沧桑/叹息"、"愤怒"、"悲伤"等），用于未来语音合成的情感控制。
- "content": 纯净的文本内容。如果 type 是 "dialogue"，必须去掉最外层的引号（如""或""）。

【输出格式示例（One-Shot）】
[
  {
    "type": "narration",
    "speaker": "narrator",
    "gender": "male",
    "emotion": "平静",
    "content": "夜幕降临，港口的灯火开始闪烁。"
  },
  {
    "type": "dialogue",
    "speaker": "老渔夫",
    "gender": "male",
    "emotion": "沧桑/叹息",
    "content": "你相信命运吗？"
  },
  {
    "type": "narration",
    "speaker": "narrator",
    "gender": "male",
    "emotion": "平静",
    "content": "老渔夫说道。"
  }
]
"""
        
        # 构建用户提示
        user_prompt = f"请严格按照规范，将以下文本拆解为纯净的 JSON 剧本（绝不改写原意）：\n\n{text_chunk}"
        if context:
            user_prompt = f"上下文信息：{context}\n\n{user_prompt}"
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "10m",
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
                "top_p": 0.1
            }
        }
        
        # 🌟 增加最大尝试次数
        max_retries = 2 
        
        for attempt in range(max_retries + 1):
            try:
                # 🌟 修改1：将 timeout 放宽到 300 秒（5分钟）
                response = requests.post(self.api_url, json=payload, timeout=300)
                response.raise_for_status()
                content = response.json().get('message', {}).get('content', '[]')
                
                # 清理Markdown代码块
                content = self._clean_markdown(content)
                
                # 解析JSON
                script = json.loads(content)
                
                # 验证并修复数据结构
                if isinstance(script, list):
                    return self._validate_script_elements(script)
                elif isinstance(script, dict):
                    # 处理包装格式
                    for value in script.values():
                        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                            return self._validate_script_elements(value)
                
                # 如果都不是期望的格式，使用降级方案
                logger.warning("⚠️ 返回格式不符合预期，使用正则降级方案")
                return self._fallback_regex_parse(text_chunk)
                
            except requests.exceptions.Timeout:
                logger.error(f"❌ Ollama 响应超时 (尝试 {attempt+1}/{max_retries+1})，模型可能输出过长。")
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON解析失败 (尝试 {attempt+1}/{max_retries+1}): {e}")
            except Exception as e:
                logger.error(f"❌ Ollama 请求异常 (尝试 {attempt+1}/{max_retries+1}): {e}")
            
            # 如果是最后一次尝试依然失败，则使用降级方案
            if attempt == max_retries:
                logger.error("🚨 达到最大重试次数，使用正则降级方案。")
                return self._fallback_regex_parse(text_chunk)
            
            # 重试前短暂等待
            import time
            time.sleep(1)
    
    def _validate_script_elements(self, script: List[Dict]) -> List[Dict]:
        """验证并修复脚本元素"""
        required_fields = ['type', 'speaker', 'content']
        validated_script = []
        
        for i, element in enumerate(script):
            if not isinstance(element, dict):
                logger.warning(f"⚠️ 脚本元素 {i} 不是字典类型，跳过")
                continue
                
            fixed_element = element.copy()
            
            # 补充缺失字段
            for field in required_fields:
                if field not in fixed_element:
                    if field == 'type':
                        fixed_element['type'] = 'narration'
                    elif field == 'speaker':
                        fixed_element['speaker'] = 'narrator'
                    elif field == 'content':
                        fixed_element['content'] = ''
                    logger.warning(f"⚠️ 补充缺失字段 '{field}'")
            
            # 确保其他必需字段
            if 'gender' not in fixed_element:
                fixed_element['gender'] = 'unknown'
            if 'emotion' not in fixed_element:
                fixed_element['emotion'] = '平静'
                
            validated_script.append(fixed_element)
        
        return validated_script
    
    def _fallback_regex_parse(self, text: str) -> List[Dict]:
        """正则降级解析方案"""
        import re
        
        units = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检测标题
            if self._is_title(line):
                units.append({
                    "type": "title", 
                    "speaker": "narrator", 
                    "gender": "unknown",
                    "emotion": "平静",
                    "content": line
                })
            # 检测对话
            elif self._is_dialogue(line):
                speaker, content = self._extract_dialogue_components(line)
                gender = self._predict_gender(speaker)
                units.append({
                    "type": "dialogue", 
                    "speaker": speaker, 
                    "gender": gender, 
                    "emotion": "平静",
                    "content": content
                })
            # 默认为旁白
            else:
                units.append({
                    "type": "narration", 
                    "speaker": "narrator", 
                    "gender": "unknown",
                    "emotion": "平静",
                    "content": line
                })
        
        return units
    
    def _is_title(self, text: str) -> bool:
        """判断是否为标题"""
        import re
        if len(text) < 30 and re.search(r'[第章节卷部集]', text):
            return True
        if text.isupper() and len(text) < 50:
            return True
        return False
    
    def _is_dialogue(self, text: str) -> bool:
        """判断是否为对话"""
        return ('"' in text or '“' in text or '”' in text)
    
    def _extract_dialogue_components(self, line: str) -> tuple:
        """提取对话组件"""
        import re
        # 简单的对话提取逻辑
        match = re.search(r'^(.*?)["“](.*?)["”]?(?:\s*(.*))?$', line)
        if match:
            speaker = match.group(1).strip().rstrip('：:')
            content = match.group(2).strip()
            return speaker if speaker else "未知角色", content
        return "未知角色", line
    
    def _clean_markdown(self, content: str) -> str:
        """清理Markdown代码块标记和多余内容"""
        import re
        # 移除开头的代码块标记
        content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
        # 移除结尾的代码块标记
        content = re.sub(r'\s*```$', '', content.strip())
        # 移除可能的前导说明文字
        content = re.sub(r'^[^\[]*?(?=\[)', '', content, flags=re.DOTALL)
        # 移除可能的尾随说明文字
        content = re.sub(r'\][^\]]*?$', ']', content, flags=re.DOTALL)
        return content.strip()
    
    def _predict_gender(self, speaker_name: str) -> str:
        """简单性别预测"""
        female_indicators = ['女士', '小姐', '夫人', '妈妈', '姐姐', '妹妹', '女儿']
        male_indicators = ['先生', '少爷', '老爷', '爸爸', '哥哥', '弟弟', '儿子']
        
        for indicator in female_indicators:
            if indicator in speaker_name:
                return "female"
        for indicator in male_indicators:
            if indicator in speaker_name:
                return "male"
        
        return "unknown"

# 兼容性函数
def create_local_llm_client(config: Dict):
    """创建本地LLM客户端实例"""
    return LocalLLMClient(config)