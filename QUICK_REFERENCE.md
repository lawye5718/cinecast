# CineCast Quick Reference Card

## 🚀 快速启动

```bash
# 启动WebUI（最常用）
python3 webui.py

# 命令行模式
python3 main_producer.py "书籍.epub"

# 纯净旁白模式
python3 main_producer.py "书籍.epub" --pure-narrator
```

## 📁 关键目录

```
assets/          # 音频资源文件
output/          # 生成的有声书
modules/         # 核心功能模块
audio_shield/    # 音频防护系统
tests/           # 测试文件
```

## ⚡ 常用命令

```bash
# 系统管理
ps aux | grep cinecast     # 查看运行进程
pkill -f "python.*cinecast" # 停止所有进程
lsof -i :7861              # 检查端口占用

# 代码更新
git pull origin master     # 拉取最新代码
git stash && git pull      # 保存本地修改后更新

# 监控系统
python3 monitor_cinecast.py  # 启动监控
python3 analyze_monitor.py   # 分析报告
```

## 🔧 配置文件位置

- **音频配置**：`assets/audio_assets_config.json`
- **环境变量**：`.env` 或系统环境变量
- **日志文件**：`cinecast.log`
- **监控日志**：`cinecast_monitor.log`

## 🎯 WebUI访问

**地址**：http://127.0.0.1:7861
**端口**：7861（可修改）

## 🛡️ Audio Shield

```bash
# 启动音频防护系统
python3 -m audio_shield

# 扫描音频文件
python3 audio_shield/scanner.py --input audio.wav
```

## ⚠️ 故障排除

**端口被占用**：
```bash
lsof -i :7861
kill -9 <PID>
```

**内存不足**：
```bash
rm -rf output/Audiobooks/temp_wav_cache/*
```

**模型加载失败**：
检查 `../qwentts/models/` 目录

## 📊 系统状态检查

```bash
# 查看系统资源
top -l 1 | grep Python

# 检查磁盘空间
df -h .

# 查看输出文件
ls -la output/Audiobooks/scripts/
ls -la output/Audiobooks/temp_wav_cache/
```

---

*快捷参考 - 随时可用*