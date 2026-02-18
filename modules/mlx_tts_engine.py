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
    def __init__(self, model_path="./models/Qwen3-TTS-MLX-0.6B", config=None):
        """
        初始化MLX纯净干音渲染引擎 (支持 Qwen3-TTS 1.7B Model Pool)
        
        Args:
            model_path: 默认模型路径 (兼容旧版单模型模式)
            config: 可选配置字典，支持多模型路径：
                - model_path_base: 1.7B Base (克隆用)
                - model_path_design: 1.7B VoiceDesign (设计用)
                - model_path_custom: 1.7B CustomVoice (内置角色用)
                - model_path_fallback: 0.6B 回退路径
        """
        logger.info("🚀 启动 MLX 纯净干音渲染引擎...")
        self.config = config or {}
        self.current_mode = None
        self.model = None
        self._fallback_path = self.config.get(
            "model_path_fallback", model_path
        )
        self._model_paths = {
            "clone": self.config.get("model_path_base"),
            "design": self.config.get("model_path_design"),
            "preset": self.config.get("model_path_custom"),
        }
        try:
            # 默认加载：如果配置了 preset 路径则用 preset，否则用传入的 model_path
            default_path = self._model_paths.get("preset") or model_path
            self._do_load(default_path, mode="preset")
            self.sample_rate = 24000  # Qwen3-TTS 1.7B 高保真采样率
            self.max_chars = 60  # 微切片安全长度上限
            logger.info("✅ MLX渲染引擎初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ 首选模型加载失败 ({e})，尝试回退到 0.6B...")
            try:
                self._do_load(self._fallback_path, mode="preset")
                self.sample_rate = 22050  # 0.6B 模型使用旧采样率
                self.max_chars = 60
                logger.info("✅ MLX渲染引擎初始化成功 (回退到 0.6B)")
            except Exception as e2:
                logger.error(f"❌ MLX渲染引擎初始化失败: {e2}")
                raise

    def _do_load(self, path, mode="preset"):
        """实际加载模型到内存"""
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            mx.clear_cache()
        self.model = load_model(path)
        self.current_mode = mode
        logger.info(f"✅ 已加载模型 [{mode}]: {path}")

    def _load_mode(self, mode):
        """根据任务类型切换模型 (Model Pool 模式)"""
        if mode == self.current_mode:
            return
        target_path = self._model_paths.get(mode)
        if not target_path:
            # 没有配置对应模式的路径，保持当前模型
            logger.debug(f"⏭️ 未配置 [{mode}] 模型路径，保持当前模型")
            return
        try:
            mx.clear_cache()
            self._do_load(target_path, mode=mode)
        except Exception as e:
            logger.warning(f"⚠️ 切换到 [{mode}] 模型失败 ({e})，保持当前模型")

    def warmup(self, modes=None):
        """预热指定模式的模型，验证路径可用性

        Args:
            modes: 要预热的模式列表，如 ["preset", "clone"]。
                   默认预热 preset 模式。
        """
        if modes is None:
            modes = ["preset"]
        for mode in modes:
            path = self._model_paths.get(mode)
            if path:
                logger.info(f"🔥 预热模型 [{mode}]: {path}")
                try:
                    self._do_load(path, mode=mode)
                except Exception as e:
                    logger.warning(f"⚠️ 预热 [{mode}] 失败: {e}")

    def destroy(self):
        """显式清理 MLX 模型资源，释放显存"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        self.current_mode = None
        mx.clear_cache()
        logger.info("🧹 MLX 渲染引擎资源已显式释放")
    
    def render_dry_chunk(self, content: str, voice_cfg: dict, save_path: str, emotion: str = "平静") -> bool:
        """
        只负责将文本变成 WAV 文件，绝不维护状态
        🌟 断点续传核心：已存在则直接跳过！
        
        支持三种 voice_cfg 模式 (通过 "mode" 字段区分)：
          - preset (默认): 传统参考音频克隆 {"mode": "preset", "audio": "...", "text": "..."}
          - clone: 用户上传音频克隆 {"mode": "clone", "ref_audio": "...", "ref_text": "..."}
          - design: 文字驱动设计 {"mode": "design", "instruct": "Deep male voice..."}
        
        Args:
            content: 要渲染的文本内容
            voice_cfg: 音色配置 (支持 preset/clone/design 三种模式)
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
            render_text = content.strip()
            
            # 🌟 终极暴力清洗：消灭一切导致复读的特殊符号
            render_text = re.sub(r'[…]+', '。', render_text)       # 中文省略号
            render_text = re.sub(r'\.{2,}', '。', render_text)     # 英文省略号（含双点）
            render_text = re.sub(r'[—]+', '，', render_text)       # 中文破折号
            render_text = re.sub(r'[-]{2,}', '，', render_text)    # 英文破折号
            render_text = re.sub(r'[~～]+', '。', render_text)     # 波浪号
            # 清洗所有内部换行和异常空白
            render_text = re.sub(r'\s+', ' ', render_text).strip()
            # 强制防卡死长度截断
            if len(render_text) > self.max_chars:
                render_text = render_text[:self.max_chars] + "。"
            
            if not re.search(r'[。！？；.!?;]$', render_text):
                render_text += "。"

            # 🌟 绝杀防御：检查清理后是否只剩下标点符号（无实际文字）
            pure_text = re.sub(r'[。，！？；、\u201c\u201d\u2018\u2019（）《》,.!?;:\'\"()\s-]', '', render_text)
            if not pure_text:
                logger.warning(f"⚠️ 切片无有效文字，跳过大模型渲染，生成 0.5s 空白音频: {save_path}")
                # 强行生成 0.5 秒的静音，避免后续混音时找不到文件报错
                audio_data = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
                sf.write(save_path, audio_data, self.sample_rate, format='WAV')
                return True

            logger.debug(f"🎵 渲染干音: {render_text[:50]}... -> {save_path}")
            
            # 🌟 根据 voice_cfg 中的 mode 字段选择渲染策略
            mode = voice_cfg.get("mode", "preset")
            self._load_mode(mode)

            if mode == "clone":
                # 克隆模式：使用用户上传的参考音频
                results = list(self.model.generate(
                    text=render_text,
                    ref_audio=voice_cfg["ref_audio"],
                    ref_text=voice_cfg.get("ref_text", "")
                ))
            elif mode == "design":
                # 设计模式：使用文字描述驱动音色
                results = list(self.model.generate(
                    text=render_text,
                    instruct=voice_cfg["instruct"]
                ))
            else:
                # 传统 Preset 模式 (兼容旧版)
                generate_kwargs = {
                    "text": render_text,
                    "ref_audio": voice_cfg["audio"],
                    "ref_text": voice_cfg["text"],
                }
                # 如果 voice_cfg 包含 speaker 字段 (CustomVoice 内置角色,
                # 如 "Male_01", "Female_03" 等 Qwen3-TTS 预设角色 ID)
                if "speaker" in voice_cfg:
                    generate_kwargs["speaker"] = voice_cfg["speaker"]
                results = list(self.model.generate(**generate_kwargs))
            
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
        
        # 测试音色配置 (传统 preset 模式)
        test_voice_cfg = {
            "mode": "preset",
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