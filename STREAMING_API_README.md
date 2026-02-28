# CineCast 流式实时朗读API

## 🌟 功能介绍

CineCast 流式实时朗读API是一个基于FastAPI的实时文本转语音服务，具有以下特色功能：

- **动态音色切换**：支持预设音色和自定义音色克隆
- **流式音频推送**：实现"边读边推"的实时体验
- **Web界面控制**：提供友好的Gradio前端界面
- **内存优化**：针对Mac mini等设备优化显存管理

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /Users/yuanliang/superstar/superstar3.1/projects/cinecast
pip install -r requirements.txt
```

### 2. 启动流式API服务

```bash
# 方法1：使用启动脚本
python start_stream_api.py

# 方法2：直接运行模块
python -m modules.stream_api
```

服务将在 `http://localhost:8000` 启动

### 3. 启动Web界面

```bash
# 启动主制片界面
python webui.py

# 启动流式API专用界面
python webui.py --mode stream
```

## 🎯 API接口说明

### 基础信息
- **基础URL**: `http://localhost:8000`
- **API文档**: `http://localhost:8000/docs` (Swagger UI)
- **健康检查**: `GET /health`

### 主要接口

#### 1. 获取音色列表
```
GET /voices
```
返回可用的预设音色和克隆音色列表

#### 2. 设置当前音色
```
POST /set_voice
Form Data:
- voice_name: 音色名称
- file: (可选) 上传的音频文件用于音色克隆
```

#### 3. OpenAI兼容流式朗读
```
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "qwen3-tts",
  "input": "要朗读的文本",
  "voice": "aiden",
  "response_format": "mp3"
}
```
返回MP3格式的音频流（推荐，解决WAV头部冗余问题）

#### 4. 传统流式朗读
```
GET /read_stream?text=朗读文本&voice=aiden
```
返回MP3格式的音频流

#### 4. 批量生成
```
POST /batch_generate
JSON Body:
{
    "text": "要合成的文本",
    "voice_name": "音色名称",
    "language": "zh"
}
```

## 🎨 Web界面使用

### 流式API专用界面 (`--mode stream`)

1. **连接测试**：点击"测试流式API连接"确认服务状态
2. **音色管理**：
   - 预设音色：从下拉菜单选择内置音色
   - 音色克隆：上传参考音频创建个性化音色
3. **实时朗读**：
   - 输入文本内容
   - 选择语言（中文/英文）
   - 点击"开始流式朗读"

### 主制片界面集成

在主界面的"第五步：流式实时朗读API"中可以找到相同的功能。

## 🔧 技术架构

### 核心组件

1. **FastAPI后端** (`modules/stream_api.py`)
   - 异步HTTP服务
   - 流式音频生成
   - 动态音色管理

2. **MLX TTS引擎** (`modules/mlx_tts_engine.py`)
   - 音频特征提取
   - 实时语音合成
   - 内存优化处理

3. **资产管理器** (`modules/asset_manager.py`)
   - 音色特征存储
   - 克隆音色管理
   - 路径配置管理

### 性能优化

- **显存管理**：每次生成后调用 `mx.metal.clear_cache()`
- **流式处理**：按句子分块生成，减少延迟
- **并发支持**：异步处理提高吞吐量

## 📋 使用示例

### Python客户端调用

```python
import requests

# 设置音色
response = requests.post(
    "http://localhost:8000/set_voice",
    data={"voice_name": "aiden"}
)
print(response.json())

# OpenAI兼容API调用
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "你好，世界！",
        "voice": "aiden",
        "response_format": "mp3"
    },
    stream=True
)

# 保存MP3音频
with open("output.mp3", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)

# 传统接口调用
response = requests.get(
    "http://localhost:8000/read_stream",
    params={"text": "你好，世界！", "voice": "aiden"},
    stream=True
)
```

### JavaScript前端调用

```javascript
// 设置音色
const setVoice = async (voiceName) => {
    const formData = new FormData();
    formData.append('voice_name', voiceName);
    
    const response = await fetch('http://localhost:8000/set_voice', {
        method: 'POST',
        body: formData
    });
    return response.json();
};

// OpenAI兼容API调用
const openaiTTS = async (text, voice = 'aiden') => {
    const response = await fetch('http://localhost:8000/v1/audio/speech', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            input: text,
            voice: voice,
            response_format: 'mp3'
        })
    });
    
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
};

// 传统接口调用
const streamRead = async (text, voice = 'aiden') => {
    const response = await fetch(
        `http://localhost:8000/read_stream?text=${encodeURIComponent(text)}&voice=${voice}`
    );
    
    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
};
```

## ⚠️ 注意事项

1. **硬件要求**：建议16GB以上内存，支持MLX框架的Mac设备
2. **音频格式**：上传的参考音频建议为WAV格式，采样率24kHz
3. **文本长度**：单次请求建议不超过5000字符
4. **并发限制**：避免同时发起多个长文本请求
5. **格式优化**：推荐使用MP3格式（`/v1/audio/speech`）避免WAV头部冗余问题
6. **显存管理**：每句生成后自动执行`mx.metal.clear_cache()`优化内存使用

## 🛠️ 故障排除

### 常见问题

1. **API连接失败**
   - 检查服务是否正常启动
   - 确认端口8000未被占用
   - 查看服务日志输出

2. **音色克隆失败**
   - 确认上传文件格式正确
   - 检查音频文件是否损坏
   - 查看后端错误日志

3. **音频质量不佳**
   - 确认参考音频质量良好
   - 尝试不同的预设音色
   - 检查文本内容是否适合TTS

### 日志查看

```bash
# 查看API服务日志
tail -f cinecast.log

# 查看系统资源使用
top -o MEM
```

## 📚 相关文档

- [项目主文档](README.md)
- [Qwen迁移报告](CINECAST_QWEN_MIGRATION_REPORT.md)
- [架构升级报告](ARCHITECTURE_UPGRADE_REPORT.md)

---
*Powered by CineCast - 电影级有声书制片解决方案*