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
import re
import numpy as np
import soundfile as sf
import mlx.core as mx
from mlx_audio.tts.utils import load_model
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

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
            # 🌟 核心修复与优化：防止自回归TTS短文本复读与不停止幻觉
            render_text = content.strip()
            # 将省略号、破折号替换为普通的逗号或句号，防止模型卡死
            render_text = re.sub(r'[…]+', '。', render_text)
            render_text = re.sub(r'[—]+', '，', render_text)
            render_text = re.sub(r'\.{3,}', '。', render_text)
            
            # 如果结尾没有标准的中文或英文闭合标点，强制补全句号
            if not re.search(r'[。！？；.!?;]$', render_text):
                render_text += "。"

            logger.debug(f"🎵 渲染干音: {render_text[:50]}... -> {save_path}")
            
            # MLX 极速推理 (传入处理后的 render_text)
            results = list(self.model.generate(
                text=render_text,
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
            raise RuntimeError(f"❌ MLX 干音渲染失败 [{content[:10]}...]: {e}") from e
            
        finally:
            # 清理内存
            if 'results' in locals(): del results
            if 'audio_array' in locals(): del audio_array
            if 'audio_data' in locals(): del audio_data
            
            # MLX 缓存清理
            mx.clear_cache()
            
            # 🌟 强制召回：在长时间循环中，必须依靠强硬的 gc 介入来防御碎片化
            # 我们引入一个微小的开销，强制 Python 每处理完一个切片就回收废弃对象
            gc.collect()

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