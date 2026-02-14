#!/usr/bin/env python3
"""
CineCast 大模型剧本预处理器
阶段一：剧本化与微切片 (Script & Micro-chunking)
实现宏观剧本解析 -> 自动展开为微切片剧本
"""

import json
import re
import logging
import requests
import os
from typing import List, Dict

logger = logging.getLogger(__name__)

class LLMScriptDirector:
    def __init__(self, ollama_url="http://127.0.0.1:11434", use_local_mlx_lm=False):
        self.api_url = f"{ollama_url}/api/chat"
        self.model_name = "qwen14b-pro"
        self.max_chars_per_chunk = 60 # 微切片红线
        self.use_local_mlx_lm = use_local_mlx_lm
        
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
    
    def _chunk_text_for_llm(self, text: str, max_length: int = 1500) -> List[str]:
        """🌟 防止章节过长，按段落切分为安全大小给 LLM 处理"""
        paragraphs = text.split('\n')
        chunks, current_chunk = [], ""
        for para in paragraphs:
            if not para.strip(): continue
            if len(current_chunk) + len(para) > max_length and current_chunk:
                chunks.append(current_chunk)
                current_chunk = para + "\n"
            else:
                current_chunk += para + "\n"
        if current_chunk:
            chunks.append(current_chunk)
        return chunks
    
    def parse_and_micro_chunk(self, text: str, chapter_prefix: str = "chunk") -> List[Dict]:
        """宏观剧本解析 -> 自动展开为微切片剧本
        
        Args:
            text: 待处理的章节文本
            chapter_prefix: 章节名称前缀，用于避免文件名冲突
        """
        # 第一步：生成宏观剧本
        macro_script = self.parse_text_to_script(text)
        micro_script = []
        chunk_id = 1
        
        for unit in macro_script:
            # 实施微切片
            raw_sentences = re.split(r'([。！？；，、：])', unit["content"])
            chunks, temp = [], ""
            for part in raw_sentences:
                if not part.strip(): continue
                if re.match(r'^[。！？；，、：]$', part.strip()):
                    chunks.append(temp + part)
                    temp = ""
                else:
                    temp += part
                    if len(temp) >= self.max_chars_per_chunk:
                        chunks.append(temp)
                        temp = ""
            if temp: chunks.append(temp)
            
            # 清理空块并计算停顿
            valid_chunks = [c.strip() for c in chunks if c.strip()]
            for idx, chunk in enumerate(valid_chunks):
                is_para_end = (idx == len(valid_chunks) - 1)
                pause_ms = self._calculate_pause(chunk, is_para_end)
                
                # 🌟 修复：将章节名称前缀加入ID，杜绝文件覆盖！
                micro_script.append({
                    "chunk_id": f"{chapter_prefix}_{chunk_id:05d}",
                    "type": unit["type"],
                    "speaker": unit["speaker"],
                    "gender": unit.get("gender", "male"),
                    "content": chunk,
                    "pause_ms": pause_ms
                })
                chunk_id += 1
                
        return micro_script

    def _calculate_pause(self, chunk_text: str, is_para_end: bool) -> int:
        """提前计算好物理停顿时间"""
        if is_para_end: return 1000
        if chunk_text.endswith(('。', '！', '？', '.', '!', '?')): return 600
        elif chunk_text.endswith(('；', ';')): return 400
        elif chunk_text.endswith(('，', '、', ',', '：', ':')): return 250
        else: return 100

    def parse_text_to_script(self, text: str) -> List[Dict]:
        """阶段一：宏观剧本解析（保持原有逻辑）"""
        # 🌟 修复截断漏洞：按段落切分长章节
        text_chunks = self._chunk_text_for_llm(text)
        full_script = []
        
        for i, chunk in enumerate(text_chunks):
            logger.info(f"   🧠 正在解析剧情片段 {i+1}/{len(text_chunks)}...")
            chunk_script = self._request_ollama(chunk)
            full_script.extend(chunk_script)
            
        return full_script
    
    def _request_ollama(self, text_chunk: str) -> List[Dict]:
        """向Ollama发送单个文本块请求"""
        system_prompt = """
        你是一位顶级的有声书导演。请将提供的小说文本拆解为严格的 JSON 数组。
        【字段要求】
        - "type": "title"(标题), "subtitle"(小标题), "narration"(旁白), "dialogue"(对白)
        - "speaker": 旁白和标题填 "narrator"，对白填具体人名（需推断，且上下文统一）。
        - "gender": "male" 或 "female" 或 "unknown"。
        - "content": 台词或描述，去除外层引号。
        【输出格式】只输出合法的 JSON 数组，不包含任何 Markdown 标记。
        """
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"请拆解以下剧情：\n\n{text_chunk}"}
            ],
            "format": "json",
            "stream": False,
            "keep_alive": "10m",  # 🌟 修复潮汐漏洞：保持模型在内存中 10 分钟
            "options": {
                "num_ctx": 8192,  # 🌟 修复截断漏洞：扩大上下文窗口
                "temperature": 0.1 # 降低温度，确保 JSON 格式稳定
            }
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=180)
            response.raise_for_status()
            content = response.json().get('message', {}).get('content', '[]')
            
            # 🌟 强力剥离 Markdown 代码块（防止 LLM 幻觉）
            content = re.sub(r'^```(?:json)?\s*', '', content.strip(), flags=re.IGNORECASE)
            content = re.sub(r'\s*```$', '', content.strip())
            
            script = json.loads(content)
            if isinstance(script, list):
                return script
            # Handle case where model returns {"result": [...]} or similar wrapper.
            # The first list value found is used since the prompt requests a single array.
            if isinstance(script, dict):
                for value in script.values():
                    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                        return value
            return self._fallback_regex_parse(text_chunk)
            
        except Exception as e:
            logger.error(f"❌ Ollama 解析失败，触发正则降级: {e}")
            return self._fallback_regex_parse(text_chunk)
    
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