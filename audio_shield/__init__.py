"""
CineCast Audio Shield（音频盾）
独立音频质量检测与修复模块，可集成到 CineCast 工作流。

模块组成：
- scanner: 文件扫描器，递归遍历文件夹建立 MP3 待审队列
- analyzer: 噪音分析引擎，基于一阶差分能量检测定位尖刺和噼啪声
- editor: 音频编辑器，支持无损试听与滑动删除
- gui: PyQt6 交互界面，提供波形展示与操作控制
"""
