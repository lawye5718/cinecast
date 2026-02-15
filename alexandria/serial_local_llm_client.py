import threading
import time
import requests
import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SerialLocalLLMClient:
    """串行本地LLM客户端 - 确保一次只处理一个请求以避免内存冲突"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("llm", {})
        self.api_url = self.config.get("api_url", "http://localhost:11434/api/chat")
        self.model_name = self.config.get("model", "qwen14b-pro")
        self.temperature = self.config.get("temperature", 0.0)
        self.num_ctx = self.config.get("num_ctx", 8192)
        
        # 串行锁，确保一次只处理一个请求
        self._request_lock = threading.Lock()
        
        # 验证连接
        self._verify_connection()
    
    def _verify_connection(self):
        """验证与本地LLM服务的连接"""
        try:
            # 测试请求
            test_payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "options": {"temperature": 0.0, "num_ctx": 512}
            }
            
            response = requests.post(self.api_url, json=test_payload, timeout=10)
            if response.status_code == 200:
                logger.info(f"✅ 本地LLM服务连接正常: {self.model_name}")
            else:
                logger.warning(f"⚠️ 本地LLM服务响应异常: {response.status_code}")
        except Exception as e:
            logger.error(f"❌ 无法连接到本地LLM服务: {e}")
    
    def generate_script(self, text_chunk: str, context: str = "") -> List[Dict]:
        """串行生成脚本 - 一次只处理一个请求"""
        with self._request_lock:  # 串行执行
            logger.info(f"🔒 串行锁已获取，开始处理LLM请求...")
            start_time = time.time()
            
            try:
                result = self._generate_script_internal(text_chunk, context)
                end_time = time.time()
                logger.info(f"✅ LLM请求处理完成，耗时: {end_time - start_time:.2f}秒")
                return result
            except Exception as e:
                end_time = time.time()
                logger.error(f"❌ LLM请求处理失败，耗时: {end_time - start_time:.2f}秒, 错误: {e}")
                raise
            finally:
                logger.info("🔓 串行锁已释放")
    
    def _generate_script_internal(self, text_chunk: str, context: str = "") -> List[Dict]:
        """内部生成方法 - 基于CineCast中验证的实现"""
        system_prompt = """
你是一位顶级的有声书导演兼数据清洗专家，负责将原始小说文本转换为标准化的录音剧本。
你必须严格遵守以下四大纪律，任何违反都将导致系统崩溃：

【一、 绝对忠实原则（Iron Rule）】
- 必须 100% 逐字保留原文内容！
- 严禁任何形式的概括、改写、缩写、续写或润色！
- 严禁自行添加原文中不存在的台词或动作描写！
- 严禁在 content 中保留归属标签（如"他说"、"她叫道"），归属信息只能出现在 speaker 字段！

【二、 字符净化原则】
- 剔除所有不可发音的特殊符号（如 Emoji表情、Markdown标记 * _ ~ #、制表符 	、不可见控制字符）。
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

        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            content = response.json().get('message', {}).get('content', '[]')

            # 清理Markdown代码块
            import re
            content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content.strip())

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

            # 降级到正则解析
            logger.warning("⚠️ JSON解析失败，使用正则降级方案")
            return self._fallback_regex_parse(text_chunk)

        except Exception as e:
            logger.error(f"❌ LLM剧本生成失败: {e}")
            return self._fallback_regex_parse(text_chunk)
    
    def _validate_script_elements(self, script: List[Dict]) -> List[Dict]:
        """验证并修复脚本元素"""
        required_fields = ['type', 'speaker', 'content']
        validated_script = []

        for i, element in enumerate(script):
            if not isinstance(element, dict):
                logger.warning(f"⚠️ 脚本元素 {i} 不是字典类型，跳过: {element}")
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
        return ('"' in text or '"' in text or
                '""' in text or '""' in text)

    def _extract_dialogue_components(self, line: str) -> tuple:
        """提取对话组件"""
        import re
        # 简单的对话提取逻辑
        match = re.search(r'^(.*?)[""""](.*)["""]', line)
        if match:
            speaker = match.group(1).strip().rstrip(':：')
            content = match.group(2).strip()
            return speaker if speaker else "未知角色", content
        return "未知角色", line

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
def create_serial_local_llm_client(config: Dict[str, Any]):
    """创建串行本地LLM客户端实例"""
    return SerialLocalLLMClient(config)
