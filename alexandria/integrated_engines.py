#!/usr/bin/env python3
"""
Alexandria项目 - 集成CineCast成功实现的MLX TTS引擎
使用本地MLX Qwen3-TTS模型进行音频生成
"""

import os
import gc
import json
import logging
import mlx.core as mx
import numpy as np
import soundfile as sf
from typing import List, Dict, Optional
from pathlib import Path

# 尝试导入MLX TTS相关模块
try:
    from mlx_audio.tts.utils import load_model
    MLX_AVAILABLE = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"MLX TTS模块不可用: {e}")
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)

class MLXTTSEngine:
    """基于CineCast成功实现的MLX TTS引擎"""

    def __init__(self, config: Dict):
        self.config = config.get("tts", {})
        # 使用CineCast中验证的模型路径
        self.model_path = self.config.get("model_path", "./models/Qwen3-TTS-MLX-0.6B")
        self.device = self.config.get("device", "metal")
        self.language = self.config.get("language", "Chinese")

        # MLX模型相关
        self.model = None
        self.sample_rate = 22050

        # 初始化模型
        if MLX_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("⚠️ MLX框架不可用，TTS功能将受限")

    def _initialize_model(self):
        """初始化MLX TTS模型 - 基于CineCast中验证的实现"""
        try:
            logger.info(f"🚀 初始化MLX TTS引擎: {self.model_path}")

            # 直接使用CineCast中验证的模型加载方式
            self.model = load_model(self.model_path)
            logger.info("✅ MLX TTS模型加载成功")

        except Exception as e:
            logger.error(f"❌ MLX TTS模型初始化失败: {e}")
            self.model = None
            raise

    def generate_voice(self, text: str, instruct_text: str, speaker: str, voice_config: Dict, output_path: str) -> bool:
        """
        生成语音 - 基于CineCast中验证的实现
        """
        if not MLX_AVAILABLE or self.model is None:
            logger.error("❌ MLX TTS引擎未初始化")
            return False

        try:
            # 文本预处理（基于CineCast的清洗规则）
            cleaned_text = self._clean_text(text)
            if len(cleaned_text) < 3:
                logger.warning(f"⚠️ 文本过短，跳过渲染: {text}")
                return self._insert_silence(output_path)

            # 获取语音配置
            voice_data = voice_config.get(speaker)
            if not voice_data:
                logger.warning(f"⚠️ 未找到说话人配置: {speaker}")
                return False

            # 获取参考音频和文本
            ref_audio_path = voice_data.get("ref_audio")
            ref_text = voice_data.get("ref_text", "参考音频文本")

            if not ref_audio_path or not os.path.exists(ref_audio_path):
                logger.error(f"❌ 参考音频不存在: {ref_audio_path}")
                return False

            # 加载参考音频
            import librosa
            ref_audio, ref_sr = librosa.load(ref_audio_path, sr=22050, mono=True)
            ref_audio = ref_audio.astype(np.float32)

            # MLX推理生成音频
            return self._generate_audio_with_mlx(cleaned_text, ref_audio, ref_sr, output_path)

        except Exception as e:
            logger.error(f"❌ TTS生成失败: {e}")
            return False
        finally:
            # 清理内存（基于CineCast的优化策略）
            self._cleanup_memory()

    def _clean_text(self, text: str) -> str:
        """文本清洗 - 基于CineCast的规则"""
        import re

        # 移除不可发音字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）]', ' ', text)

        # 标准化标点符号
        text = re.sub(r'[,.!?;:]', lambda m: {'!': '！', '?': '？', ';': '；', ':': '：',
                                             ',': '，', '.': '。'}[m.group()], text)

        # 清理多余空白
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _generate_audio_with_mlx(self, text: str, ref_audio: np.ndarray, ref_sr: int, output_path: str) -> bool:
        """使用MLX生成音频"""
        try:
            logger.debug(f"🎵 MLX渲染: {text[:50]}... -> {output_path}")

            # 使用MLX模型生成音频
            results = list(self.model.generate(
                text=text,
                ref_audio=(ref_audio, ref_sr),
                ref_text="参考音频文本"  # 使用固定的参考文本
            ))

            if not results or len(results) == 0:
                logger.error(f"❌ MLX未生成音频结果")
                return False

            audio_array = results[0].audio
            mx.eval(audio_array)  # 强制执行
            audio_data = np.array(audio_array)

            # 直接写入磁盘，避免内存积压
            sf.write(output_path, audio_data, self.sample_rate, format='WAV')
            logger.debug(f"✅ MLX音频渲染完成: {output_path}, 大小: {os.path.getsize(output_path)} bytes")
            return True

        except Exception as e:
            logger.error(f"❌ MLX音频生成失败: {e}")
            return False

    def _insert_silence(self, save_path: str) -> bool:
        """插入静音文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 生成1秒静音
            silence = np.zeros(22050, dtype=np.float32)  # 1秒 @ 22050Hz
            sf.write(save_path, silence, 22050, subtype='FLOAT')
            logger.debug(f"✅ 静音文件创建成功: {save_path}")
            return True
        except Exception as e:
            logger.error(f"❌ 静音文件创建失败: {e}")
            return False

    def _cleanup_memory(self):
        """内存清理 - 基于CineCast的优化策略"""
        try:
            # MLX显存清理
            if hasattr(mx, 'metal') and hasattr(mx.metal, 'clear_cache'):
                mx.metal.clear_cache()
            elif hasattr(mx, 'clear_cache'):
                mx.clear_cache()

            # Python垃圾回收（适度使用）
            if 'gc' not in globals():
                import gc
            gc.collect()

        except Exception as e:
            logger.debug(f"内存清理小错误（可忽略）: {e}")

    def is_available(self) -> bool:
        """检查TTS引擎是否可用"""
        return MLX_AVAILABLE and self.model is not None


class SerialLocalLLMClient:
    """串行本地LLM客户端 - 基于CineCast中验证的实现"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("llm", {})
        self.api_url = self.config.get("api_url", "http://localhost:11434/api/chat")
        self.model_name = self.config.get("model", "qwen14b-pro")
        self.temperature = self.config.get("temperature", 0.0)
        self.num_ctx = self.config.get("num_ctx", 8192)
        
        # 串行锁，确保一次只处理一个请求
        self._lock = threading.Lock()
        
        # 导入requests
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("需要安装requests库: pip install requests")
    
    def generate_script(self, text_chunk: str, context: str = "") -> list:
        """串行生成脚本 - 一次只处理一个请求以避免内存冲突"""
        with self._lock:
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
    
    def _generate_script_internal(self, text_chunk: str, context: str = "") -> list:
        """内部生成方法 - 基于CineCast中验证的实现"""
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
            response = self.requests.post(self.api_url, json=payload, timeout=180)
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
def create_mlx_tts_engine(config: Dict):
    """创建MLX TTS引擎实例"""
    return MLXTTSEngine(config)


def create_serial_local_llm_client(config: Dict):
    """创建串行本地LLM客户端实例"""
    import threading
    import time
    from typing import Dict, Any
    
    return SerialLocalLLMClient(config)