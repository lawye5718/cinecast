#!/usr/bin/env python3
"""
CineCast Audio Shield — 独立启动入口

用法：
    python -m audio_shield              # 启动 GUI
    python -m audio_shield --scan DIR   # 仅命令行扫描
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    parser = argparse.ArgumentParser(
        description="CineCast Audio Shield — 音频质量检测与修复工具"
    )
    parser.add_argument(
        "--scan",
        metavar="DIR",
        help="命令行模式：扫描指定目录并输出分析结果（不启动 GUI）",
    )
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=0.4,
        help="噪音检测灵敏度 (0.1–1.0)，越小越灵敏，默认 0.4",
    )
    args = parser.parse_args()

    if args.scan:
        _cli_scan(args.scan, args.sensitivity)
    else:
        _launch_gui()


def _cli_scan(directory: str, sensitivity: float):
    """命令行模式：扫描并报告"""
    from audio_shield.scanner import AudioScanner, FileStatus
    from audio_shield.analyzer import detect_audio_glitches

    scanner = AudioScanner(directory)
    files = scanner.scan()

    if not files:
        print(f"未在 {directory} 中找到音频文件。")
        return

    for finfo in files:
        try:
            glitches = detect_audio_glitches(
                finfo.file_path, sensitivity=sensitivity
            )
        except Exception as exc:
            print(f"  ✗ {finfo.filename}: 分析失败 — {exc}")
            continue

        if glitches:
            scanner.update_status(finfo.file_path, FileStatus.NEEDS_FIX, glitches)
            ts = ", ".join(f"{t:.3f}s" for t in glitches)
            print(f"  ⚠️  {finfo.filename}: {len(glitches)} 处异常 [{ts}]")
        else:
            scanner.update_status(finfo.file_path, FileStatus.PASSED)
            print(f"  ✅ {finfo.filename}: 无异常")


def _launch_gui():
    """启动 PyQt6 GUI"""
    try:
        from audio_shield.gui import launch_gui
        launch_gui()
    except RuntimeError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
