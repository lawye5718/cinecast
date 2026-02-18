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

    # 参数校验
    sensitivity = max(0.1, min(1.0, sensitivity))
    min_interval = max(0.01, min_interval)

    # 加载音频，sr=None 保持原始采样率
    y, sr = librosa.load(file_path, sr=None)

    if len(y) < 2:
        return []

    # 1. 计算一阶差分（变化率）
    diff = np.abs(np.diff(y))

    # 2. 动态阈值：基于全局均值+标准差，识别异常跳变点
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    if std_diff == 0:
        return []

    threshold = mean_diff + (std_diff * (1 / sensitivity) * _THRESHOLD_MULTIPLIER)

    # 3. 找到超过阈值的索引
    glitch_indices = np.where(diff > threshold)[0]
    glitch_times = librosa.samples_to_time(glitch_indices, sr=sr)

    # 4. 聚类：min_interval 秒内的多个点视为同一个噪音区
    refined_glitches: List[float] = []
    if len(glitch_times) > 0:
        last_added = -min_interval
        for t in glitch_times:
            if t - last_added > min_interval:
                refined_glitches.append(round(float(t), 3))
                last_added = t

    logger.info(
        f"🔎 分析完成: {file_path} — "
        f"发现 {len(refined_glitches)} 处疑似噪音 "
        f"(灵敏度={sensitivity})"
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

    与 detect_audio_glitches 使用相同算法，但接受 numpy 数组输入，
    适用于已在内存中的音频数据。

    Args:
        y: 音频采样数组（mono）
        sr: 采样率
        sensitivity: 灵敏度 (0.1–1.0)
        min_interval: 最小间隔（秒）

    Returns:
        时间戳列表（秒）
    """
    sensitivity = max(0.1, min(1.0, sensitivity))
    min_interval = max(0.01, min_interval)

    if len(y) < 2:
        return []

    diff = np.abs(np.diff(y))
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    if std_diff == 0:
        return []

    threshold = mean_diff + (std_diff * (1 / sensitivity) * _THRESHOLD_MULTIPLIER)

    glitch_indices = np.where(diff > threshold)[0]
    # 将采样索引转换为时间（秒）
    glitch_times = glitch_indices.astype(float) / sr

    refined_glitches: List[float] = []
    if len(glitch_times) > 0:
        last_added = -min_interval
        for t in glitch_times:
            if t - last_added > min_interval:
                refined_glitches.append(round(float(t), 3))
                last_added = t

    return refined_glitches
