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
        """生成有声书剧本 - 使用CineCast中验证的System Prompt"""
        
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
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
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
            
            # 降级到正则解析
            logger.warning("⚠️ JSON解析失败，使用正则降级方案")
            return self._fallback_regex_parse(text_chunk)
            
        except Exception as e:
            logger.error(f"❌ LLM剧本生成失败: {e}")
            return self._fallback_regex_parse(text_chunk)
    
    def _clean_markdown(self, content: str) -> str:
        """清理Markdown代码块标记"""
        import re
        content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
        content = re.sub(r'\s*```$', '', content.strip())
        return content
    
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