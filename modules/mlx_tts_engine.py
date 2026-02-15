#!/usr/bin/env python3
"""
CineCast MLX底层渲染引擎
阶段二：纯净干音渲染 (Dry Voice Rendering)
只负责将文本变成 WAV 文件，绝不维护状态
基于qwentts项目的成熟实现

Supports an optional "group-by-voice" rendering strategy: instead of
rendering chunks in script order (which forces frequent voice-embedding
switches), callers can use ``group_indices_by_voice_type`` to cluster all
chunks that share the same voice first, render each cluster in one pass,
and then reassemble in the original order during Stage 3.
"""

import gc
import os
import numpy as np
import soundfile as sf
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

try:
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    _MLX_AVAILABLE = True
except (ImportError, OSError):
    mx = None
    load_model = None
    _MLX_AVAILABLE = False

logger = logging.getLogger(__name__)


def group_indices_by_voice_type(
    micro_script: List[Dict],
) -> Dict[str, List[int]]:
    """Group script indices by their effective voice type.

    Returns a dict mapping voice-type keys (e.g. ``"narrator"``,
    ``"dialogue:老渔夫"``) to the list of indices in *micro_script* that
    should be rendered with that voice.  This allows the caller to render
    all chunks for a single voice consecutively, minimising MLX
    embedding switches and potentially improving throughput by 2-3×.
    """
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, item in enumerate(micro_script):
        item_type = item.get("type", "narration")
        speaker = item.get("speaker", "narrator")
        if item_type in ("title", "subtitle", "narration", "recap"):
            key = item_type
        else:
            key = f"dialogue:{speaker}"
        groups[key].append(idx)
    return dict(groups)

class MLXRenderEngine:
    def __init__(self, model_path="./models/Qwen3-TTS-MLX-0.6B"):
        """
        初始化MLX纯净干音渲染引擎
        
        Args:
            model_path: Qwen3-TTS-MLX模型路径
        """
        if not _MLX_AVAILABLE:
            raise ImportError(
                "MLX is not available in this environment. "
                "MLX requires Apple Silicon (macOS with M-series chips)."
            )
        logger.info("🚀 启动 MLX 纯净干音渲染引擎...")
        try:
            self.model = load_model(model_path)
            self.sample_rate = 22050
            self.max_chars = 60  # 微切片安全长度上限
            logger.info("✅ MLX渲染引擎初始化成功")
        except Exception as e:
            logger.error(f"❌ MLX渲染引擎初始化失败: {e}")
            raise
    
    def render_dry_chunk(self, content: str, voice_cfg: dict, save_path: str, emotion: str = "平静") -> bool:
        """
        只负责将文本变成 WAV 文件，绝不维护状态
        🌟 断点续传核心：已存在则直接跳过！
        
        Args:
            content: 要渲染的文本内容
            voice_cfg: 音色配置
            save_path: 保存路径
            emotion: 情感标签（预留参数，当前版本暂不使用）
        """
        # TODO: [CineCast 2.0 预留] 当前 Qwen3-TTS 暂不支持细粒度情感参数
        # 未来接入 CosyVoice/ChatTTS 时，将 emotion 传入模型 prompt
        # current_prompt = f"<{emotion}> {content}"
        if os.path.exists(save_path):
            logger.debug(f"⏭️  文件已存在，跳过渲染: {save_path}")
            return True # 🌟 断点续传核心：已存在则直接跳过！
            
        try:
            logger.debug(f"🎵 渲染干音: {content[:50]}... -> {save_path}")
            
            # MLX 极速推理
            results = list(self.model.generate(
                text=content,
                ref_audio=voice_cfg["audio"],
                ref_text=voice_cfg["text"]
            ))
            
            audio_array = results[0].audio
            mx.eval(audio_array) # 强制执行
            audio_data = np.array(audio_array)
            
            # 直接写入磁盘，绝不在内存中积压
            sf.write(save_path, audio_data, self.sample_rate, format='WAV')
            logger.debug(f"✅ 干音渲染完成: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 干音渲染失败 [{content[:10]}...]: {e}")
            return False
            
        finally:
            # 清理内存 (保留局部变量删除和 mx 的缓存清理)
            if 'results' in locals(): del results
            if 'audio_array' in locals(): del audio_array
            if 'audio_data' in locals(): del audio_data
            if mx is not None and hasattr(mx, 'metal'):
                mx.metal.clear_cache()
            
            # 🌟 优化：移除全局的 gc.collect()。
            # Python 的引用计数已经能自动清理大部分局部变量，
            # mx.metal.clear_cache() 足以防止 MLX 显存泄漏。
            # 如果不放心，可以引入一个计数器，每处理 50 个 chunk 才调用一次 gc.collect()。

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.DEBUG)
    
    # 注意：这里需要确保模型路径正确
    try:
        engine = MLXRenderEngine()
        
        # 测试音色配置
        test_voice_cfg = {
            "audio": "reference_for_production.wav",
            "text": "测试参考文本",
            "speed": 1.0
        }
        
        # 测试渲染（使用三段式架构的 render_dry_chunk）
        test_content = "这是一个测试文本，用来验证MLX渲染引擎是否正常工作。"
        test_save_path = "/tmp/cinecast_test_dry.wav"
        success = engine.render_dry_chunk(test_content, test_voice_cfg, test_save_path)
        
        if success:
            print(f"✅ 渲染成功，干音文件已写入: {test_save_path}")
        else:
            print("❌ 渲染失败")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")