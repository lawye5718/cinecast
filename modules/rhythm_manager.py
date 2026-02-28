#!/usr/bin/env python3
"""
CineCast 智能韵律控制器 (Rhythm Manager)
根据标点符号自动注入不同长度的停顿，解决 TTS 常见的"断句生硬"问题。
支持中英文标点符号的动态停顿配置。
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class RhythmManager:
    """智能韵律与停顿控制器。

    通过标点符号映射注入动态停顿指令，提升有声书的自然呼吸感。
    支持自定义停顿配置，可针对不同标点设置不同的沉默时长（秒）。
    """

    # 默认停顿配置（秒）
    DEFAULT_PAUSES = {
        "comma": 0.2,       # 逗号 (，,)
        "period": 0.5,      # 句号 (。.)
        "question": 0.6,    # 问号 (？?)
        "exclamation": 0.5, # 感叹号 (！!)
        "semicolon": 0.3,   # 分号 (；;)
        "colon": 0.3,       # 冒号 (：:)
        "ellipsis": 0.8,    # 省略号 (……...)
        "dash": 0.4,        # 破折号 (——--)
        "newline": 0.8,     # 换行符
    }

    # 标点符号到停顿类型的映射
    _PUNCT_MAP = {
        "，": "comma", ",": "comma",
        "。": "period", ".": "period",
        "？": "question", "?": "question",
        "！": "exclamation", "!": "exclamation",
        "；": "semicolon", ";": "semicolon",
        "：": "colon", ":": "colon",
    }

    def __init__(self, config: Optional[Dict[str, float]] = None):
        """初始化韵律控制器。

        Args:
            config: 自定义停顿配置字典，键为停顿类型名，值为秒数。
                    未提供的键将使用默认值。
        """
        self.pauses = dict(self.DEFAULT_PAUSES)
        if config:
            self.pauses.update(config)

    def get_pause_duration(self, punct_type: str) -> float:
        """获取指定停顿类型的时长（秒）。"""
        return self.pauses.get(punct_type, 0.0)

    def process_text_with_metadata(self, text: str) -> List[Dict]:
        """将文本拆分为带停顿信息的片段。

        根据标点符号将文本分割成多个片段，每个片段附带停顿时长元数据。
        用于后续 TTS 渲染时在片段之间注入沉默帧。

        Args:
            text: 原始输入文本

        Returns:
            带停顿元数据的片段列表，每个元素为：
            {"text": "片段文本", "pause": 停顿秒数}
        """
        if not text or not text.strip():
            return []

        segments = []

        # 先处理省略号和破折号（多字符标点）
        processed = re.sub(r'[…]{1,}|\.{3,}', f'\x00ELLIPSIS\x00', text)
        processed = re.sub(r'[—]{2,}|[-]{2,}', f'\x00DASH\x00', processed)

        # 按标点分句（保留标点在前一句末尾）
        parts = re.split(r'(?<=[\x00，,。.？?！!；;：:\n])', processed)

        for part in parts:
            if not part.strip() and '\n' not in part:
                continue

            pause = 0.0

            if '\x00ELLIPSIS\x00' in part:
                part = part.replace('\x00ELLIPSIS\x00', '')
                pause = self.pauses["ellipsis"]
            elif '\x00DASH\x00' in part:
                part = part.replace('\x00DASH\x00', '')
                pause = self.pauses["dash"]
            elif '\n' in part:
                part = part.replace('\n', ' ')
                pause = self.pauses["newline"]
            else:
                # 检查末尾标点符号
                for punct_char, punct_type in self._PUNCT_MAP.items():
                    if part.rstrip().endswith(punct_char):
                        pause = self.pauses[punct_type]
                        break

            clean_text = part.strip()
            if clean_text:
                segments.append({"text": clean_text, "pause": pause})

        return segments

    def inject_pauses(self, text: str) -> str:
        """在文本中根据标点符号注入停顿标记。

        将标点符号后添加 [pause=N.N] 标记，供下游 TTS 引擎解析。

        Args:
            text: 原始文本

        Returns:
            带停顿标记的文本
        """
        if not text:
            return text

        # 处理多字符标点
        result = re.sub(r'[…]{1,}|\.{3,}',
                        f'…[pause={self.pauses["ellipsis"]}]', text)
        result = re.sub(r'[—]{2,}|[-]{2,}',
                        f'——[pause={self.pauses["dash"]}]', result)

        # 处理单字符标点（中文）
        for punct_char, punct_type in self._PUNCT_MAP.items():
            duration = self.pauses[punct_type]
            result = result.replace(punct_char,
                                    f'{punct_char}[pause={duration}]')

        return result

    def create_silence_frames(self, duration: float, sample_rate: int = 24000):
        """创建指定时长的沉默帧数组。

        Args:
            duration: 沉默时长（秒）
            sample_rate: 采样率，默认 24000（Qwen3-TTS 1.7B 标准）

        Returns:
            numpy 零数组，表示沉默音频帧
        """
        import numpy as np
        num_frames = int(duration * sample_rate)
        return np.zeros(num_frames, dtype=np.float32)

    def update_config(self, new_config: Dict[str, float]):
        """动态更新停顿配置。

        Args:
            new_config: 新的停顿配置（部分更新，不会清除未提及的键）
        """
        self.pauses.update(new_config)
        logger.info(f"🎵 韵律配置已更新: {new_config}")
