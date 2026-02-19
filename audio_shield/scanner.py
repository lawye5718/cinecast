#!/usr/bin/env python3
"""
CineCast Audio Shield — 文件扫描器 (Scanner)

递归遍历文件夹，建立 MP3 待审队列，管理任务状态。
"""

import os
import logging
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class FileStatus(Enum):
    """文件审查状态"""
    PENDING = "pending"       # 待审
    PASSED = "passed"         # 通过（无异常）
    NEEDS_FIX = "needs_fix"   # 待修复（检测到异常）
    FIXED = "fixed"           # 已修复


class AudioFileInfo:
    """单个音频文件的信息与状态"""

    def __init__(self, file_path: str):
        self.file_path = str(Path(file_path).resolve())
        self.filename = os.path.basename(file_path)
        self.status = FileStatus.PENDING
        self.glitches: List[float] = []  # 检测到的噪音时间戳列表（秒）

    def __repr__(self):
        status_icon = {
            FileStatus.PENDING: "⏳",
            FileStatus.PASSED: "✅",
            FileStatus.NEEDS_FIX: "⚠️",
            FileStatus.FIXED: "🔧",
        }
        icon = status_icon.get(self.status, "?")
        glitch_info = f" ({len(self.glitches)}处异常)" if self.glitches else ""
        return f"[{icon}] {self.filename}{glitch_info}"


class AudioScanner:
    """
    文件扫描器：递归扫描目录，建立音频文件待审队列。

    支持的格式：.mp3, .wav, .flac, .ogg, .m4a
    """

    SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}

    def __init__(self, source_dir: str):
        """
        初始化扫描器

        Args:
            source_dir: 要扫描的根目录路径
        """
        self.source_dir = str(Path(source_dir).resolve())
        self.files: List[AudioFileInfo] = []
        self._index_map: Dict[str, int] = {}  # file_path -> index in self.files

    def scan(self) -> List[AudioFileInfo]:
        """
        递归扫描目录，建立待审队列。

        Returns:
            扫描到的音频文件信息列表
        """
        self.files.clear()
        self._index_map.clear()

        if not os.path.isdir(self.source_dir):
            logger.warning(f"扫描目录不存在: {self.source_dir}")
            return self.files

        for root, _dirs, filenames in os.walk(self.source_dir):
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS:
                    full_path = os.path.join(root, fname)
                    info = AudioFileInfo(full_path)
                    self._index_map[info.file_path] = len(self.files)
                    self.files.append(info)

        logger.info(f"🔍 扫描完成，发现 {len(self.files)} 个音频文件")
        return self.files

    def get_pending_files(self) -> List[AudioFileInfo]:
        """获取所有待审文件"""
        return [f for f in self.files if f.status == FileStatus.PENDING]

    def get_needs_fix_files(self) -> List[AudioFileInfo]:
        """获取所有待修复文件"""
        return [f for f in self.files if f.status == FileStatus.NEEDS_FIX]

    def update_status(self, file_path: str, status: FileStatus,
                      glitches: Optional[List[float]] = None):
        """
        更新文件状态

        Args:
            file_path: 文件路径
            status: 新状态
            glitches: 检测到的噪音时间戳列表
        """
        resolved = str(Path(file_path).resolve())
        idx = self._index_map.get(resolved)
        if idx is not None:
            self.files[idx].status = status
            if glitches is not None:
                self.files[idx].glitches = glitches

    def get_file_info(self, file_path: str) -> Optional[AudioFileInfo]:
        """根据路径获取文件信息"""
        resolved = str(Path(file_path).resolve())
        idx = self._index_map.get(resolved)
        if idx is not None:
            return self.files[idx]
        return None

    def get_progress_stats(self) -> tuple:
        """
        返回处理进度统计。

        Returns:
            (processed_count, total_count, percentage) 元组，
            其中 processed_count 包含状态为 PASSED 或 FIXED 的文件数量。
        """
        total = len(self.files)
        processed = sum(
            1 for f in self.files
            if f.status in (FileStatus.PASSED, FileStatus.FIXED)
        )
        percentage = int((processed / total) * 100) if total > 0 else 0
        return processed, total, percentage
