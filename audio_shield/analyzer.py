#!/usr/bin/env python3
"""
CineCast Audio Shield — 噪音分析引擎 (Analyzer)

核心算法模块：对音频进行分帧处理，通过一阶差分能量检测定位尖刺和噼啪声。
"""

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Threshold multiplier: controls how many standard deviations above the mean
# the first-order difference must exceed to be classified as a glitch.
_THRESHOLD_MULTIPLIER = 5

# ---------------------------------------------------------------------------
# 尝试导入 librosa；CI 等轻量环境可能没有安装
# ---------------------------------------------------------------------------
try:
    import librosa  # type: ignore
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


def detect_audio_glitches(
    file_path: str,
    sensitivity: float = 0.4,
    min_interval: float = 0.5,
) -> List[float]:
    """
    检测 AI 生成音频中的尖刺/噼啪声。

    算法：一阶差分能量检测
    如果相邻采样点之间的振幅差值突然超过周围平均水平的数倍，即判定为尖刺。

    Args:
        file_path: 音频文件路径
        sensitivity: 灵敏度 (0.1–1.0)，越小越灵敏
        min_interval: 两个噪音点之间的最小间隔（秒），防止重复报警

    Returns:
        包含时间戳的列表（单位: 秒）

    Raises:
        RuntimeError: 如果 librosa 不可用
    """
    if not _HAS_LIBROSA:
        raise RuntimeError(
            "librosa is required for audio analysis. "
            "Install it with: pip install librosa"
        )

    # 加载音频，sr=None 保持原始采样率
    y, sr = librosa.load(file_path, sr=None)

    glitches = detect_audio_glitches_pro(
        y, sr, sensitivity=sensitivity, min_interval=min_interval
    )

    logger.info(
        f"🔎 分析完成: {file_path} — "
        f"发现 {len(glitches)} 处疑似噪音 "
        f"(灵敏度={sensitivity})"
    )
    return glitches


def detect_audio_glitches_pro(
    y: np.ndarray,
    sr: int,
    window_size_sec: float = 1.0,
    sensitivity: float = 0.4,
    min_interval: float = 0.5,
) -> List[float]:
    """
    滑动窗口版噪音检测，防止长音频动态范围过大导致的漏检。

    使用局部窗口计算阈值，而非全局平均值和标准差。
    当音频开头有大动态音乐时，全局标准差会被拉高，导致后面安静
    片段中的微小"噼啪声"被掩盖。滑动窗口逐段计算局部阈值来解决此问题。

    Args:
        y: 音频采样数组（mono）
        sr: 采样率
        window_size_sec: 滑动窗口大小（秒），默认 1.0
        sensitivity: 灵敏度 (0.1–1.0)，越小越灵敏
        min_interval: 两个噪音点之间的最小间隔（秒），防止重复报警

    Returns:
        包含时间戳的列表（单位: 秒）
    """
    # 参数校验
    sensitivity = max(0.1, sensitivity)
    min_interval = max(0.01, min_interval)

    if len(y) < 2:
        return []

    win_length = int(window_size_sec * sr)
    # Minimum of 2 samples needed for np.diff calculation
    if win_length < 2:
        win_length = 2

    glitch_times_raw: List[float] = []

    # 使用滑动窗口计算局部阈值，防止长音频动态范围过大导致的漏检
    # 50% overlap 保证相邻窗口之间不会有盲区
    step = max(1, win_length // 2)
    for i in range(0, len(y) - 1, step):
        chunk = y[i : i + win_length]
        diff = np.abs(np.diff(chunk))

        if len(diff) == 0:
            continue

        local_mean = np.mean(diff)
        local_std = np.std(diff)

        if local_std == 0:
            continue

        local_threshold = local_mean + (
            local_std * (1 / sensitivity) * _THRESHOLD_MULTIPLIER
        )

        indices = np.where(diff > local_threshold)[0]
        for idx in indices:
            t = (i + idx) / sr
            glitch_times_raw.append(t)

    # 去重 + 排序
    glitch_times_raw = sorted(set(glitch_times_raw))

    # 聚类：min_interval 秒内的多个点视为同一个噪音区
    refined_glitches: List[float] = []
    if len(glitch_times_raw) > 0:
        last_added = -min_interval
        for t in glitch_times_raw:
            if t - last_added > min_interval:
                refined_glitches.append(round(float(t), 3))
                last_added = t

    # 过滤：如果报警点超过 50 个，可能为底噪而非爆音，自动调高阈值重试
    # sensitivity 上限为 5.0 以防止无限递归
    if len(refined_glitches) > 50 and sensitivity * 1.5 <= 5.0:
        logger.warning(
            "检测到过多疑似点 (%d)，可能为底噪，自动调高阈值重试...",
            len(refined_glitches),
        )
        return detect_audio_glitches_pro(
            y, sr,
            window_size_sec=window_size_sec,
            sensitivity=sensitivity * 1.5,
            min_interval=min_interval,
        )

    return refined_glitches


def detect_glitches_from_array(
    y: np.ndarray,
    sr: int,
    sensitivity: float = 0.4,
    min_interval: float = 0.5,
) -> List[float]:
    """
    从已加载的音频数组中检测尖刺。

    与 detect_audio_glitches 使用相同算法（滑动窗口），但接受 numpy 数组输入，
    适用于已在内存中的音频数据。

    Args:
        y: 音频采样数组（mono）
        sr: 采样率
        sensitivity: 灵敏度 (0.1–1.0)
        min_interval: 最小间隔（秒）

    Returns:
        时间戳列表（秒）
    """
    return detect_audio_glitches_pro(
        y, sr, sensitivity=sensitivity, min_interval=min_interval
    )
