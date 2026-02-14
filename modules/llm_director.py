#!/usr/bin/env python3
"""
CineCast 大模型剧本预处理器
利用本地Qwen模型将小说文本转化为结构化剧本
"""

import json
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class LLMScriptDirector:
    def __init__(self, use_local_mlx_lm=True):
        self.use_local = use_local_mlx_lm
        self.model = None
        self.tokenizer = None
        
        if self.use_local:
            self._initialize_local_model()
    
    def _initialize_local_model(self):
        """初始化本地模型（优先使用Ollama Qwen14B）"""
        # 首先尝试使用本地Ollama Qwen14B模型
        if self._try_ollama_qwen():
            return
            
        # 如果Ollama不可用，则尝试MLX模型
        try:
            from mlx_lm import load
            model_names = [
                "Qwen/Qwen1.5-4B-Chat-MLX",
                "qwen/Qwen1.5-4B-Chat-MLX",
                "mlx-community/Qwen1.5-4B-Chat-MLX"
            ]
            
            for model_name in model_names:
                try:
                    logger.info(f"尝试加载MLX模型: {model_name}")
                    self.model, self.tokenizer = load(model_name)
                    logger.info(f"✅ 成功加载MLX模型: {model_name}")
                    return
                except Exception as e:
                    logger.warning(f"加载MLX模型 {model_name} 失败: {e}")
                    continue
            
            logger.error("❌ 未能加载任何本地Qwen模型")
            self.use_local = False
            
        except ImportError:
            logger.warning("mlx_lm 未安装或不可用，将使用正则降级方案")
            self.use_local = False
        except Exception as e:
            logger.error(f"初始化本地模型失败: {e}")
            self.use_local = False
    
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
        """
        核心方法：调用大模型进行剧本杀级解析
        """
        if self.use_local and self.model and self.tokenizer:
            return self._parse_with_local_llm(text)
        else:
            logger.info("使用正则降级方案解析文本")
            return self._fallback_regex_parse(text)
    
    def _parse_with_local_llm(self, text: str) -> List[Dict]:
        """
        使用本地模型进行智能解析（优先Ollama Qwen14B）
        """
        try:
            if hasattr(self, 'model_type') and self.model_type == "ollama":
                return self._parse_with_ollama(text)
            else:
                return self._parse_with_mlx(text)
                
        except Exception as e:
            logger.error(f"本地LLM解析失败: {e}")
            return self._fallback_regex_parse(text)
    
    def _parse_with_ollama(self, text: str) -> List[Dict]:
        """使用Ollama Qwen14B模型解析"""
        import subprocess
        import json
        
        prompt = f"""
请作为一个专业的广播剧导演，将以下小说文本转化为JSON格式的剧本。
必须识别出：旁白(narration)、对白(dialogue)、标题(title)。
对于对白，必须推断出说话人(speaker)和性别(gender)。

文本：{text[:2000]}  # 限制长度避免模型负担

返回格式示例：
[
  {{"type": "narration", "speaker": "narrator", "content": "夜幕降临..."}},
  {{"type": "dialogue", "speaker": "老渔夫", "gender": "male", "content": "你相信命运吗？"}}
]
只返回JSON数组，不要其他废话。
"""
        
        result = subprocess.run(
            ["ollama", "run", self.model_name, prompt],
            capture_output=True,
            text=True,
            timeout=120  # 2分钟超时
        )
        
        if result.returncode == 0:
            response = result.stdout.strip()
            # 提取JSON内容
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logger.warning("Ollama模型输出中未找到有效JSON，使用降级方案")
                return self._fallback_regex_parse(text)
        else:
            error_msg = result.stderr.strip() if result.stderr else "未知错误"
            logger.error(f"Ollama模型调用失败: {error_msg}")
            return self._fallback_regex_parse(text)
    
    def _parse_with_mlx(self, text: str) -> List[Dict]:
        """使用MLX模型解析"""
        from mlx_lm import generate
        
        prompt = f"""
请作为一个专业的广播剧导演，将以下小说文本转化为JSON格式的剧本。
必须识别出：旁白(narration)、对白(dialogue)、标题(title)。
对于对白，必须推断出说话人(speaker)和性别(gender)。

文本：{text[:2000]}  # 限制长度避免模型负担

返回格式示例：
[
  {{"type": "narration", "speaker": "narrator", "content": "夜幕降临..."}},
  {{"type": "dialogue", "speaker": "老渔夫", "gender": "male", "content": "你相信命运吗？"}}
]
只返回JSON数组，不要其他废话。
"""
        
        response = generate(self.model, self.tokenizer, prompt, max_tokens=1000)
        
        # 提取JSON内容
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            logger.warning("MLX模型输出中未找到有效JSON，使用降级方案")
            return self._fallback_regex_parse(text)
    
    def _fallback_regex_parse(self, text: str) -> List[Dict]:
        """
        当大模型解析失败或未启用时的保底方案
        """
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
            r'^(.*?)\s*[："“「『]\s*(.*?)\s*[："“」』]$',
            r'^(.*?)\s*[："“]\s*(.*?)(?=\s*[："“]|$)',
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