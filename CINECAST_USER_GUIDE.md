# CineCast Pro 3.0 用户指南

## 📋 目录结构

```
cinecast/
├── 📁 assets/                          # 资源文件目录
│   ├── 📁 bgm/                        # 背景音乐文件
│   │   ├── fountain.mp3              # 喷泉环境音
│   │   ├── forest.mp3                # 森林环境音
│   │   └── office.mp3                # 办公室环境音
│   ├── 📁 chimes/                     # 过渡音效文件
│   │   ├── soft_chime.mp3            # 轻柔铃声
│   │   └── dramatic_chime.mp3        # 戏剧性铃声
│   ├── 📁 voices/                     # 音色文件目录
│   │   ├── narrator.wav              # 叙述者音色
│   │   ├── male_lead.wav             # 男主角音色
│   │   ├── female_lead.wav           # 女主角音色
│   │   └── character_specific/       # 角色专用音色
│   └── 📄 audio_assets_config.json   # 音频资源配置文件
│
├── 📁 modules/                        # 核心模块目录
│   ├── 📄 asset_manager.py           # 资源管理器
│   ├── 📄 cinematic_packager.py      # 电影级打包器
│   ├── 📄 llm_director.py            # LLM导演模块
│   ├── 📄 mlx_tts_engine.py          # MLX TTS引擎
│   └── 📄 webui_components.py        # WebUI组件
│
├── 📁 audio_shield/                   # 音频防护系统
│   ├── 📄 __init__.py                # 初始化文件
│   ├── 📄 analyzer.py                # 音频分析器
│   ├── 📄 editor.py                  # 音频编辑器
│   ├── 📄 gui.py                     # 图形界面
│   ├── 📄 main.py                    # 主程序入口
│   └── 📄 scanner.py                 # 音频扫描器
│
├── 📁 output/                         # 输出目录
│   └── 📁 Audiobooks/                # 有声书输出
│       ├── 📁 scripts/               # 剧本文件
│       │   ├── Chapter_001_micro.json
│       │   └── Chapter_002_micro.json
│       ├── 📁 temp_wav_cache/        # 临时WAV缓存
│       └── 📁 final/                 # 最终成品目录
│
├── 📁 tests/                          # 测试文件目录
│   ├── 📄 test_audio_shield.py       # 音频防护测试
│   ├── 📄 test_engine_hot_restart.py # 引擎热重启测试
│   ├── 📄 test_tts_punctuation_guard.py # TTS标点防护测试
│   └── 📄 test_workspace_persistence.py # 工作区持久化测试
│
├── 📄 main_producer.py               # 主生产程序
├── 📄 webui.py                       # Web用户界面
├── 📄 requirements.txt               # 依赖包列表
├── 📄 .gitignore                     # Git忽略文件配置
└── 📄 README.md                      # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/lawye5718/cinecast.git
cd cinecast

# 创建虚拟环境
python3 -m venv cinecast_venv
source cinecast_venv/bin/activate  # Linux/Mac
# 或 cinecast_venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# 启动WebUI（推荐方式）
python3 webui.py

# 或者使用命令行模式
python3 main_producer.py "你的书籍.epub"
```

### 3. 访问界面

打开浏览器访问：`http://127.0.0.1:7861`

## 🎯 核心功能使用

### WebUI操作流程

1. **上传书籍文件**
   - 支持EPUB、TXT格式
   - 拖拽或点击上传

2. **选择制作模式**
   - 🎭 **智能配音模式**：自动识别角色并分配音色
   - 🔊 **纯净旁白模式**：单一叙述者朗读

3. **配置音色设置**
   - 上传自定义音色文件
   - 选择预设音色包
   - 调整语速、音调参数

4. **开始制作**
   - 点击"极速试听"预览效果
   - 点击"全本压制"生成完整有声书

### 命令行使用

```bash
# 基本用法
python3 main_producer.py "书籍文件.epub"

# 纯净旁白模式
python3 main_producer.py "书籍文件.epub" --pure-narrator

# 指定输出目录
python3 main_producer.py "书籍文件.epub" --output "./my_audiobooks"
```

## ⚙️ 高级配置

### 音频资源配置 (audio_assets_config.json)

```json
{
  "voices": {
    "narrator": {
      "file": "./assets/voices/narrator.wav",
      "gender": "male",
      "style": "calm"
    },
    "male_lead": {
      "file": "./assets/voices/male_lead.wav",
      "gender": "male",
      "style": "energetic"
    }
  },
  "bgm": {
    "fountain": "./assets/bgm/fountain.mp3",
    "forest": "./assets/bgm/forest.mp3"
  },
  "chimes": {
    "soft": "./assets/chimes/soft_chime.mp3",
    "dramatic": "./assets/chimes/dramatic_chime.mp3"
  }
}
```

### 环境变量配置

```bash
# 设置Ollama服务地址
export OLLAMA_HOST="http://localhost:11434"

# 设置MLX模型路径
export MLX_MODEL_PATH="../qwentts/models/Qwen3-TTS-MLX-0.6B"

# 设置日志级别
export LOG_LEVEL="INFO"
```

## 🔧 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 检查端口占用
   lsof -i :7861
   
   # 修改webui.py中的端口配置
   ui.launch(server_port=7862)
   ```

2. **模型加载失败**
   ```bash
   # 检查模型文件是否存在
   ls -la ../qwentts/models/
   
   # 重新下载模型
   git clone https://github.com/your-model-repo.git
   ```

3. **内存不足**
   ```bash
   # 清理临时文件
   rm -rf output/Audiobooks/temp_wav_cache/*
   
   # 监控内存使用
   python3 monitor_cinecast.py
   ```

### 系统监控

```bash
# 启动监控脚本
python3 monitor_cinecast.py

# 查看监控数据
python3 view_monitor.py

# 生成分析报告
python3 analyze_monitor.py
```

## 🛡️ Audio Shield 音频防护系统

### 功能特性

- **智能检测**：滑动窗口算法检测音频质量问题
- **自动修复**：识别并修正音频缺陷
- **质量控制**：四级质量保障机制
- **可视化界面**：直观的GUI操作界面

### 使用方法

```bash
# 启动Audio Shield
python3 -m audio_shield

# 命令行模式
python3 audio_shield/main.py --input audio_file.wav --output cleaned_audio.wav
```

## 📊 性能优化建议

### 硬件要求
- **CPU**：推荐8核以上
- **内存**：推荐16GB以上
- **存储**：SSD存储提升处理速度

### 软件优化
```bash
# 启用MLX加速
export MLX_ENABLE_COMPILE_CACHE=1

# 设置合适的线程数
export OMP_NUM_THREADS=8

# 清理不必要的进程
pkill -f "unnecessary_process"
```

## 🔒 安全注意事项

1. **文件权限**：确保只有授权用户可以访问敏感文件
2. **网络安区**：在受信任的网络环境中运行
3. **数据备份**：定期备份重要的音频资源文件
4. **版本控制**：使用Git管理代码变更

## 🆘 技术支持

### 获取帮助
- 查看详细日志：`tail -f cinecast.log`
- 运行诊断脚本：`python3 diagnostics.py`
- 提交Issue：在GitHub仓库提交问题报告

### 社区资源
- GitHub仓库：https://github.com/lawye5718/cinecast
- 文档网站：[待补充]
- 用户论坛：[待补充]

---

*本文档最后更新：2026年2月*
*CineCast Pro 3.0 版本*