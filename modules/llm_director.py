#!/usr/bin/env python3
"""
CineCast 大模型剧本预处理器
利用本地Qwen模型将小说文本转化为结构化剧本
"""

import json
import re
import logging
import requests
from typing import List, Dict

logger = logging.getLogger(__name__)

class LLMScriptDirector:
    def __init__(self, ollama_host="http://localhost:11434"):
        self.api_url = f"{ollama_host}/api/chat"
        # 请确保你在 ollama 中运行的模型名称与此一致，例如 "qwen2.5:14b"
        self.model_name = "qwen14b-pro" 
        self.use_local = True  # 默认使用Ollama
        
        # 测试Ollama连接
        self._test_ollama_connection()
    
    def _test_ollama_connection(self):
        """测试Ollama服务连接"""
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
    
    def _try_ollama_qwen(self) -> bool:
        """尝试使用Ollama的Qwen14B模型"""
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0 and "qwen14b-pro" in result.stdout:
                logger.info("✅ 成功检测到本地Ollama Qwen14B模型")
                self.model_type = "ollama"
                self.model_name = "qwen14b-pro"
                return True
            else:
                logger.info("未找到Ollama Qwen14B模型")
                return False
                
        except Exception as e:
            logger.warning(f"检查Ollama模型时出错: {e}")
            return False
    
    def parse_text_to_script(self, text: str) -> List[Dict]:
        """调用本地 Ollama 14B 进行专业剧本拆解"""
        logger.info(f"🧠 请求 Ollama ({self.model_name}) 拆解剧本...")
        
        system_prompt = """
        你是一位顶级的有声书导演。请将提供的小说文本拆解为专业的广播剧JSON剧本。
        
        【角色规则】
        1. type 必须是 "title"(章节标题), "subtitle"(小标题), "narration"(旁白), "dialogue"(对白) 之一。
        2. 对于 dialogue，必须推断出具体的 speaker（人名）和 gender（male/female）。
        3. speaker 字段必须统一，如果同一个人说话，名字必须完全一致。
        
        【输出要求】
        必须且只能输出一个合法的 JSON 数组，格式如下：
        [
          {"type": "title", "speaker": "narrator", "content": "第一章 风雪"},
          {"type": "subtitle", "speaker": "narrator", "content": "1976年"},
          {"type": "narration", "speaker": "narrator", "content": "夜幕降临。"},
          {"type": "dialogue", "speaker": "老渔夫", "gender": "male", "content": "你相信命运吗？"}
        ]
        """
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请拆解以下文本：\n{text[:2500]}"}
            ],
            "format": "json",       # 强制 Ollama 输出 JSON
            "stream": False,
            "keep_alive": 0         # 🌟 核心防冲突：生成完毕后，立即将 14B 模型从 M4 内存中卸载！
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            # 提取并解析 JSON
            content = result.get('message', {}).get('content', '[]')
            
            # 🌟 幻觉防御：强力剥离 Markdown 代码块
            content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content.strip())
            
            script = json.loads(content)
            
            # 兜底校验
            if not isinstance(script, list):
                raise ValueError("Ollama 返回的不是 JSON 数组")
            return script
            
        except Exception as e:
            logger.error(f"❌ Ollama 解析失败，触发降级方案: {e}")
            return self._fallback_regex_parse(text)
    
    def _fallback_regex_parse(self, text: str) -> List[Dict]:
        """🌟 降级正则方案：当大模型解析失败时的保底方案"""
        units = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检测标题（章节标题通常较短且有特定格式）
            if self._is_title(line):
                units.append({
                    "type": "title", 
                    "speaker": "narrator", 
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
                    "content": content
                })
            # 默认为旁白
            else:
                units.append({
                    "type": "narration", 
                    "speaker": "narrator", 
                    "content": line
                })
        
        return units
    
    def _is_title(self, text: str) -> bool:
        """判断是否为标题"""
        # 标题特征：较短、可能包含"第"、"章"等字样
        if len(text) < 30 and re.search(r'[第章节卷部集]', text):
            return True
        # 或者全是大写字母（英文标题）
        if text.isupper() and len(text) < 50:
            return True
        return False
        
    def _is_dialogue(self, text: str) -> bool:
        """判断是否为对话"""
        # 包含引号的文本
        if ('"' in text or '"' in text or 
            '“' in text or '”' in text or
            '『' in text or '』' in text):
            return True
        return False
        
    def _extract_dialogue_components(self, text: str) -> tuple:
        """提取对话的说话人和内容"""
        # 处理常见的对话格式
        patterns = [
            r'^(.*?)\s*[:："“「『]\s*(.*?)\s*[:："“」』]$',
            r'^(.*?)\s*[:："“]\s*(.*?)(?=\s*[:："“]|$)',
            r'^["“](.*?)["”]\s*[—\-]\s*(.*)$',
        ]
            
        for pattern in patterns:
            match = re.match(pattern, text.strip())
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    speaker = groups[0].strip()
                    content = groups[1].strip()
                    # 清理内容中的引号
                    content = re.sub(r'^["“”『』「」]|["“”『』「」]$', '', content)
                    return speaker, content
            
        # 如果无法解析，返回默认值
        return "未知角色", text
        
    def _predict_gender(self, speaker_name: str) -> str:
        """
        简单的性别预测（可根据需要扩展）
        """
        # 常见的女性名字特征
        female_indicators = ['女士', '小姐', '夫人', '妈妈', '姐姐', '妹妹', '女儿']
        male_indicators = ['先生', '少爷', '老爷', '爸爸', '哥哥', '弟弟', '儿子']
            
        # 基于称谓判断
        for indicator in female_indicators:
            if indicator in speaker_name:
                return "female"
        for indicator in male_indicators:
            if indicator in speaker_name:
                return "male"
            
        # 基于常见姓名库判断（简化版）
        female_names = ['玛丽', '琳达', '芭芭拉', '伊丽莎白', '珍妮弗', '李娜', '王芳', '张丽']
        male_names = ['约翰', '迈克尔', '大卫', '罗伯特', '詹姆斯', '李明', '王强', '张伟']
            
        for name in female_names:
            if name in speaker_name:
                return "female"
        for name in male_names:
            if name in speaker_name:
                return "male"
            
        # 默认返回男性（可根据统计数据调整）
        return "male"

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    director = LLMScriptDirector()
    
    # 测试文本
    test_text = """
第一章 凯夫拉维克的风雪

夜幕降临，港口的灯火开始闪烁。

"你相信命运吗？"老渔夫说道。

年轻人摇摇头："我只相信海。"

远处传来汽笛声，划破了寂静的夜空。
"""
    
    script = director.parse_text_to_script(test_text)
    print("解析结果:")
    for i, unit in enumerate(script, 1):
        print(f"{i}. {unit}")