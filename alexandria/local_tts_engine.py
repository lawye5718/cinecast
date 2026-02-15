#!/usr/bin/env python3
"""
本地化TTS引擎 - 集成CineCast中测试通过的MLX Qwen-TTS模型
"""

import os
import gc
import json
import logging
import mlx.core as mx
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

class LocalTTSEngine:
    """本地MLX Qwen-TTS引擎 - 基于CineCast中验证的实现"""
    
    def __init__(self, config: Dict):
        self.config = config.get("tts", {})
        self.model_path = self.config.get("model_path", "../qwentts/models/Qwen3-TTS-MLX-0.6B")
        self.device = self.config.get("device", "metal")
        self.language = self.config.get("language", "Chinese")
        
        # MLX模型相关
        self.model = None
        self.tokenizer = None
        self.speech_tokenizer = None
        
        # 初始化模型
        if MLX_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("⚠️ MLX框架不可用，TTS功能将受限")
    
    def _initialize_model(self):
        """初始化MLX TTS模型 - 直接使用CineCast的load_model方式"""
        try:
            logger.info(f"🚀 初始化MLX TTS引擎: {self.model_path}")
            
            # 直接使用CineCast中验证的模型加载方式
            self.model = load_model(self.model_path)
            logger.info("✅ MLX TTS模型加载成功")
                
        except Exception as e:
            logger.error(f"❌ MLX TTS模型初始化失败: {e}")
            self.model = None
    
    def render_dry_chunk(self, text: str, voice_config: Dict, save_path: str, emotion: str = "平静") -> bool:
        """
        纯净干音渲染 - 基于CineCast中验证的实现
        只负责将文本变成WAV文件，绝不维护状态
        """
        if not MLX_AVAILABLE or self.model is None:
            logger.error("❌ MLX TTS引擎未初始化")
            return False
        
        try:
            # 文本预处理（基于CineCast的清洗规则）
            cleaned_text = self._clean_text(text)
            if len(cleaned_text) < 3:
                logger.warning(f"⚠️ 文本过短，跳过渲染: {text}")
                return self._insert_silence(save_path)
            
            # 应用情感参数（预留接口）
            # TODO: 未来版本支持情感控制
            processed_text = cleaned_text
            
            # MLX推理生成音频
            audio_array = self._generate_audio(processed_text, voice_config)
            
            # 保存WAV文件
            return self._save_wav(audio_array, save_path)
            
        except Exception as e:
            logger.error(f"❌ TTS渲染失败: {e}")
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
    
    def _generate_audio(self, text: str, voice_config: Dict):
        """生成音频数组 - 基于CineCast的正确实现"""
        try:
            logger.debug(f"🎵 开始生成音频: {text[:50]}...")
            
            # 直接使用模型的generate方法（基于CineCast的实现）
            # 注意：这里需要根据实际的voice_config结构调整参数
            if 'audio' in voice_config and 'text' in voice_config:
                # 如果提供了参考音频和文本
                results = list(self.model.generate(
                    text=text,
                    ref_audio=voice_config['audio'],
                    ref_text=voice_config['text']
                ))
            else:
                # 默认生成方式
                results = list(self.model.generate(text=text))
            
            # 提取音频数据
            if results and hasattr(results[0], 'audio'):
                audio_array = results[0].audio
                mx.eval(audio_array)  # 强制执行计算
                logger.debug(f"✅ 音频生成成功: {audio_array.shape}")
                return audio_array
            else:
                logger.warning("⚠️ 生成结果为空，返回静音")
                import numpy as np
                return np.zeros(24000, dtype=np.float32)  # 1秒静音
                
        except Exception as e:
            logger.error(f"❌ 音频生成失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            import numpy as np
            return np.zeros(24000, dtype=np.float32)
    
    def _save_wav(self, audio_array, save_path: str) -> bool:
        """保存WAV文件"""
        try:
            import soundfile as sf
            import numpy as np
            
            logger.debug(f"💾 开始保存音频文件: {save_path}")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 转换为适当的格式
            if isinstance(audio_array, mx.array):
                audio_data = np.array(audio_array.astype(mx.float32))
            else:
                audio_data = np.array(audio_array, dtype=np.float32)
            
            logger.debug(f"📊 音频数据信息: shape={audio_data.shape}, dtype={audio_data.dtype}, min={audio_data.min():.6f}, max={audio_data.max():.6f}")
            
            # 检查是否为有效的音频数据
            if audio_data.size == 0 or (audio_data.max() == 0.0 and audio_data.min() == 0.0):
                logger.warning(f"⚠️ 检测到静音或空音频数据: {save_path}")
            
            # 保存为WAV文件
            sf.write(save_path, audio_data, 24000, subtype='FLOAT')
            logger.debug(f"✅ 音频保存成功: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 音频保存失败: {e}")
            return False
    
    def _insert_silence(self, save_path: str) -> bool:
        """插入静音文件"""
        try:
            import soundfile as sf
            import numpy as np
            
            # 生成1秒静音
            silence = np.zeros(24000, dtype=np.float32)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            sf.write(save_path, silence, 24000, subtype='FLOAT')
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
            gc.collect()
            
        except Exception as e:
            logger.debug(f"内存清理小错误（可忽略）: {e}")
    
    def is_available(self) -> bool:
        """检查TTS引擎是否可用"""
        return MLX_AVAILABLE and self.model is not None

# 兼容性函数
def create_local_tts_engine(config: Dict):
    """创建本地TTS引擎实例"""
    return LocalTTSEngine(config)