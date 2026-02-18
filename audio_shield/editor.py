#!/usr/bin/env python3
"""
CineCast Audio Shield — 播放与编辑器 (Editor)

载入音频流，支持精确到毫秒的定位、试听，以及基于内存的切片删除逻辑。
包含无损试听与滑动删除功能，通过 pydub 在内存中操作。
"""

import logging
from typing import Optional

from pydub import AudioSegment

logger = logging.getLogger(__name__)


class AudioBufferManager:
    """
    音频缓冲管理器。

    在内存中操作 pydub AudioSegment，支持：
    - 删除指定时间范围（带 crossfade 防止新噼啪声）
    - 撤销操作
    - 全局归一化（Limiter）
    - 导出保存
    """

    # 默认 crossfade 时长（毫秒），用于删除剪切点过渡
    DEFAULT_CROSSFADE_MS = 10

    def __init__(self, file_path: Optional[str] = None):
        """
        初始化缓冲管理器

        Args:
            file_path: 音频文件路径。如果为 None，则初始化为空音频。
        """
        if file_path is not None:
            self.audio = AudioSegment.from_file(file_path)
            self._original_path = file_path
        else:
            self.audio = AudioSegment.empty()
            self._original_path = None
        self.history: list = []  # 用于撤销操作

    @property
    def duration_seconds(self) -> float:
        """当前音频时长（秒）"""
        return len(self.audio) / 1000.0

    def load(self, file_path: str):
        """
        载入新的音频文件，清空历史。

        Args:
            file_path: 音频文件路径
        """
        self.audio = AudioSegment.from_file(file_path)
        self._original_path = file_path
        self.history.clear()

    def delete_range(
        self,
        start_sec: float,
        end_sec: float,
        crossfade_ms: int = DEFAULT_CROSSFADE_MS,
    ) -> bool:
        """
        删除指定范围的音频。

        保留 start_sec 之前和 end_sec 之后的部分，
        使用 crossfade 过渡以防止剪切处产生新噼啪声。

        Args:
            start_sec: 开始时间（秒）
            end_sec: 结束时间（秒）
            crossfade_ms: 交叉淡入淡出时长（毫秒），默认 10ms

        Returns:
            True 删除成功
        """
        if start_sec < 0 or end_sec < 0:
            raise ValueError("start_sec and end_sec must be non-negative")
        if start_sec >= end_sec:
            raise ValueError("start_sec must be less than end_sec")

        duration_ms = len(self.audio)
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)

        # 裁剪到有效范围
        start_ms = max(0, min(start_ms, duration_ms))
        end_ms = max(0, min(end_ms, duration_ms))

        if start_ms >= end_ms:
            return False

        # 保存副本用于撤销
        self.history.append(self.audio)

        before = self.audio[:start_ms]
        after = self.audio[end_ms:]

        # 当两段都有足够长度时使用 crossfade
        effective_crossfade = min(crossfade_ms, len(before), len(after))
        if effective_crossfade > 0 and len(before) > 0 and len(after) > 0:
            self.audio = before.append(after, crossfade=effective_crossfade)
        else:
            self.audio = before + after

        logger.info(
            f"✂️ 已删除 [{start_sec:.3f}s - {end_sec:.3f}s] "
            f"(crossfade={effective_crossfade}ms)"
        )
        return True

    def undo(self) -> bool:
        """
        撤销上一次编辑操作。

        Returns:
            True 如果撤销成功，False 如果没有历史记录
        """
        if not self.history:
            return False
        self.audio = self.history.pop()
        logger.info("↩️ 已撤销上一次编辑")
        return True

    def normalize(self, target_dbfs: float = -3.0):
        """
        全局归一化（Limiter），防止数字剪切爆音。

        将音频峰值归一化到目标 dBFS 水平。

        Args:
            target_dbfs: 目标 dBFS 水平，默认 -3.0 dBFS
        """
        if len(self.audio) == 0:
            return

        self.history.append(self.audio)
        change = target_dbfs - self.audio.max_dBFS
        self.audio = self.audio.apply_gain(change)
        logger.info(f"📊 已归一化到 {target_dbfs} dBFS (增益 {change:+.1f} dB)")

    def save_result(self, output_path: str, file_format: str = "mp3"):
        """
        导出音频到文件。

        Args:
            output_path: 输出文件路径
            file_format: 输出格式，默认 "mp3"
        """
        self.audio.export(output_path, format=file_format)
        logger.info(f"💾 已保存: {output_path}")

    def get_segment(self, start_sec: float, end_sec: float) -> AudioSegment:
        """
        获取指定时间范围的音频片段（用于试听）。

        Args:
            start_sec: 开始时间（秒）
            end_sec: 结束时间（秒）

        Returns:
            AudioSegment 片段
        """
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        return self.audio[start_ms:end_ms]
