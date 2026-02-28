#!/usr/bin/env python3
"""
CineCast 有声书编排器 (Audiobook Orchestrator)
整合 MLX 推理引擎、韵律控制器和角色管理器，
支持多角色、多语言和高效批量推理。
"""

import gc
import logging
import re
from typing import List, Dict, Tuple, Optional

import numpy as np

from modules.rhythm_manager import RhythmManager
from modules.role_manager import RoleManager

logger = logging.getLogger(__name__)

# 支持的语言映射
LANGUAGE_MAP = {
    "Chinese": "zh", "中文": "zh", "zh": "zh",
    "English": "en", "英文": "en", "en": "en",
    "Japanese": "jp", "日文": "jp", "jp": "jp",
    "Korean": "ko", "韩文": "ko", "ko": "ko",
    "French": "fr", "法文": "fr", "fr": "fr",
    "German": "de", "德文": "de", "de": "de",
    "Spanish": "es", "西班牙文": "es", "es": "es",
    "Italian": "it", "意大利文": "it", "it": "it",
    "Russian": "ru", "俄文": "ru", "ru": "ru",
    "Portuguese": "pt", "葡萄牙文": "pt", "pt": "pt",
}


def parse_script_line(line: str) -> Tuple[Optional[str], str]:
    """解析"角色名：文本内容"格式的剧本行。

    支持中文冒号（：）和英文冒号（:）。
    如果行中没有角色标记，则整行视为旁白内容。

    Args:
        line: 单行剧本文本

    Returns:
        (角色名, 文本内容) 元组。无角色时角色名为 None。
    """
    line = line.strip()
    if not line:
        return None, ""

    # 匹配 "角色名：内容" 或 "角色名: 内容"
    match = re.match(r'^([^：:]{1,20})[：:]\s*(.+)', line)
    if match:
        role_name = match.group(1).strip()
        content = match.group(2).strip()
        return role_name, content

    return None, line


def parse_script(text: str) -> List[Tuple[Optional[str], str]]:
    """解析多行剧本文本。

    Args:
        text: 完整剧本文本

    Returns:
        列表，每个元素为 (角色名, 文本内容) 元组
    """
    results = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            results.append(parse_script_line(line))
    return results


class AudiobookOrchestrator:
    """有声书编排器。

    整合所有功能模块，支持多角色、多语言有声书实时转换。
    """

    def __init__(self, engine=None, role_manager: Optional[RoleManager] = None,
                 rhythm_config: Optional[Dict[str, float]] = None,
                 sample_rate: int = 24000):
        """初始化编排器。

        Args:
            engine: MLX TTS 引擎实例（MLXRenderEngine 或 CinecastMLXEngine）
            role_manager: 角色管理器实例
            rhythm_config: 韵律控制器的自定义停顿配置
            sample_rate: 音频采样率
        """
        self.engine = engine
        self.rm = role_manager or RoleManager()
        self.rhythm = RhythmManager(rhythm_config)
        self.sample_rate = sample_rate

    def process_chapter(self, script: List[Tuple[str, str]],
                        role_names: Optional[List[str]] = None,
                        lang: str = "Chinese",
                        batch_size: int = 1,
                        paragraph_pause: float = 0.5) -> np.ndarray:
        """处理单个章节的多角色剧本。

        Args:
            script: 剧本列表 [("角色名", "文本内容"), ...]
            role_names: 需要加载的角色名列表（为 None 时自动扫描）
            lang: 语言名称
            batch_size: 批处理大小（Mac mini 建议 1-2）
            paragraph_pause: 段落间停顿时长（秒）

        Returns:
            合并后的音频 numpy 数组
        """
        # 1. 加载角色库
        role_bank = self.rm.load_role_bank(role_names)
        lang_code = LANGUAGE_MAP.get(lang, "zh")

        final_audio_segments = []

        for role, text in script:
            if not text.strip():
                continue

            # 2. 韵律处理
            segments = self.rhythm.process_text_with_metadata(text)

            for seg in segments:
                seg_text = seg["text"]
                seg_pause = seg["pause"]

                if not seg_text.strip():
                    continue

                # 3. 推理生成音频
                audio_segment = self._generate_for_role(
                    seg_text, role, role_bank, lang_code
                )
                if audio_segment is not None:
                    final_audio_segments.append(audio_segment)

                # 4. 注入片段内停顿
                if seg_pause > 0:
                    silence = self.rhythm.create_silence_frames(
                        seg_pause, self.sample_rate
                    )
                    final_audio_segments.append(silence)

            # 5. 注入段落停顿（角色发言之间）
            silence = self.rhythm.create_silence_frames(
                paragraph_pause, self.sample_rate
            )
            final_audio_segments.append(silence)

        # 6. 合并所有片段
        if not final_audio_segments:
            return np.array([], dtype=np.float32)

        return np.concatenate(final_audio_segments)

    def _generate_for_role(self, text: str, role: Optional[str],
                           role_bank: Dict, lang_code: str) -> Optional[np.ndarray]:
        """为指定角色生成音频。

        Args:
            text: 要生成的文本
            role: 角色名
            role_bank: 已加载的角色库
            lang_code: 语言代码

        Returns:
            音频 numpy 数组，或 None
        """
        if self.engine is None:
            logger.warning("⚠️ 引擎未初始化，跳过音频生成")
            return None

        try:
            if role and role in role_bank:
                # 使用角色库中的特征克隆
                feature = role_bank[role]
                if hasattr(self.engine, 'generate_voice_clone'):
                    audio, sr = self.engine.generate_voice_clone(text, feature)
                    return audio
                elif hasattr(self.engine, 'generate'):
                    audio, sr = self.engine.generate(
                        text, mode="clone",
                        prompt_npz=feature,
                        language=lang_code
                    )
                    return audio

            # 回退到基础模式
            if hasattr(self.engine, 'generate'):
                audio, sr = self.engine.generate(text, mode="base")
                return audio

        except Exception as e:
            logger.error(f"❌ 角色 [{role}] 音频生成失败: {e}")

        return None

    def process_chapter_from_text(self, text: str,
                                  lang: str = "Chinese",
                                  paragraph_pause: float = 0.5) -> np.ndarray:
        """从原始文本解析剧本并处理章节。

        自动解析"角色名：文本内容"格式。

        Args:
            text: 原始剧本文本
            lang: 语言名称
            paragraph_pause: 段落间停顿时长

        Returns:
            合并后的音频 numpy 数组
        """
        script = parse_script(text)
        return self.process_chapter(script, lang=lang,
                                    paragraph_pause=paragraph_pause)

    def clear_memory(self):
        """清理内存缓存。

        在章节处理间隙调用，防止统一内存持续膨胀。
        """
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except (ImportError, AttributeError):
            pass
        logger.info("🧹 内存缓存已清理")
