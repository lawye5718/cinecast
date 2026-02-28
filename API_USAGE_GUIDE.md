# CineCast 流式TTS API 使用指南

## 📋 概述

CineCast流式TTS API提供了完整的文本转语音服务，支持多种音色选择、实时音色克隆和流式音频输出。API完全兼容OpenAI TTS标准，同时提供增强的中文语音合成能力。

## 🚀 快速开始

### 基础调用示例

#### Python客户端
```python
import requests

# OpenAI兼容接口调用
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "你好世界，欢迎使用CineCast流式TTS服务",
        "voice": "aiden",
        "response_format": "mp3"
    },
    stream=True
)

# 保存音频文件
with open("output.mp3", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            f.write(chunk)
```

#### JavaScript前端
```javascript
// OpenAI兼容接口调用
const response = await fetch('http://localhost:8000/v1/audio/speech', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        input: '你好世界，欢迎使用CineCast流式TTS服务',
        voice: 'aiden',
        response_format: 'mp3'
    })
});

const audioBlob = await response.blob();
const audioUrl = URL.createObjectURL(audioBlob);
const audio = new Audio(audioUrl);
audio.play();
```

## 🎵 音色管理

### 预设音色列表

系统内置多种高质量预设音色：

| 音色ID | 性别 | 特点 | 适用场景 |
|--------|------|------|----------|
| aiden | 男 | 清晰标准 | 通用播报 |
| dylan | 男 | 磁性深沉 | 新闻朗读 |
| emma | 女 | 温柔甜美 | 故事讲述 |
| sophia | 女 | 专业正式 | 商务场景 |

### 查询可用音色
```bash
curl http://localhost:8000/voices
```

## 🎙️ 音色克隆功能

### 上传自定义音色

#### Python示例
```python
import requests

# 准备音频文件和参考文本
files = {'file': open('my_voice_sample.wav', 'rb')}
data = {
    'voice_name': 'my_custom_voice',
    'ref_text': '你好世界，今天天气很好'  # 重要：提供准确的参考文本
}

# 上传音色
response = requests.post(
    'http://localhost:8000/set_voice',
    files=files,
    data=data
)

print(response.json())
# 输出: {"status": "success", "role": "clone_1234567890"}
```

#### JavaScript示例
```javascript
const formData = new FormData();
formData.append('file', audioFile);
formData.append('voice_name', 'my_custom_voice');
formData.append('ref_text', '你好世界，今天天气很好');

const response = await fetch('/set_voice', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(result);
```

### 使用克隆音色生成语音
```python
# 使用刚刚克隆的音色
response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "这是我的自定义音色",
        "voice": "my_custom_voice",  # 使用自定义音色ID
        "response_format": "mp3"
    },
    stream=True
)
```

## 🌐 API接口详解

### 1. OpenAI兼容接口
```
POST /v1/audio/speech
```

**请求参数**:
```json
{
  "model": "qwen3-tts",          // 模型名称（可选）
  "input": "要合成的文本",        // 必填
  "voice": "aiden",              // 音色ID（必填）
  "response_format": "mp3",      // 输出格式（可选，默认mp3）
  "speed": 1.0                   // 语速（可选，默认1.0）
}
```

**响应**: 流式MP3音频数据

### 2. 传统流式接口
```
GET /read_stream?text=文本&voice=音色ID
```

**参数**:
- `text`: 要合成的文本（必填）
- `voice`: 音色ID（可选，默认aiden）

**响应**: 流式MP3音频数据

### 3. 音色设置接口
```
POST /set_voice
```

**表单参数**:
- `voice_name`: 音色名称（必填）
- `file`: 音频文件（可选，用于音色克隆）
- `ref_text`: 参考文本（可选，用于音色克隆）

**响应**:
```json
{
  "status": "success",
  "role": "音色ID"
}
```

### 4. 健康检查
```
GET /health
```

**响应**:
```json
{
  "status": "healthy",
  "initialized": true,
  "current_voice": "aiden"
}
```

## 📱 前端集成示例

### React组件示例
```jsx
import React, { useState } from 'react';

function TTSService() {
  const [text, setText] = useState('');
  const [voice, setVoice] = useState('aiden');
  const [isLoading, setIsLoading] = useState(false);

  const generateSpeech = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/v1/audio/speech', {
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
    } catch (error) {
      console.error('TTS生成失败:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <textarea 
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="输入要转换的文本..."
      />
      <select value={voice} onChange={(e) => setVoice(e.target.value)}>
        <option value="aiden">Aiden (男声)</option>
        <option value="dylan">Dylan (男声)</option>
        <option value="emma">Emma (女声)</option>
        <option value="sophia">Sophia (女声)</option>
      </select>
      <button onClick={generateSpeech} disabled={isLoading}>
        {isLoading ? '生成中...' : '生成语音'}
      </button>
    </div>
  );
}
```

### Vue组件示例
```vue
<template>
  <div>
    <textarea v-model="text" placeholder="输入要转换的文本..."></textarea>
    <select v-model="voice">
      <option value="aiden">Aiden (男声)</option>
      <option value="dylan">Dylan (男声)</option>
      <option value="emma">Emma (女声)</option>
      <option value="sophia">Sophia (女声)</option>
    </select>
    <button @click="generateSpeech" :disabled="loading">
      {{ loading ? '生成中...' : '生成语音' }}
    </button>
  </div>
</template>

<script>
export default {
  data() {
    return {
      text: '',
      voice: 'aiden',
      loading: false
    };
  },
  methods: {
    async generateSpeech() {
      this.loading = true;
      try {
        const response = await fetch('/v1/audio/speech', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            input: this.text,
            voice: this.voice,
            response_format: 'mp3'
          })
        });

        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        audio.play();
      } catch (error) {
        console.error('TTS生成失败:', error);
      } finally {
        this.loading = false;
      }
    }
  }
};
</script>
```

## ⚙️ 高级功能

### 批量文本处理
```python
import asyncio
import aiohttp

async def batch_tts(texts, voice='aiden'):
    """批量生成TTS音频"""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, text in enumerate(texts):
            task = asyncio.create_task(
                generate_single_tts(session, text, voice, f"output_{i}.mp3")
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks)

async def generate_single_tts(session, text, voice, filename):
    async with session.post(
        'http://localhost:8000/v1/audio/speech',
        json={
            'input': text,
            'voice': voice,
            'response_format': 'mp3'
        }
    ) as response:
        with open(filename, 'wb') as f:
            async for chunk in response.content.iter_chunked(8192):
                f.write(chunk)
```

### 音频流实时播放
```javascript
// 实时流式播放
async function streamPlay(text, voice = 'aiden') {
    const response = await fetch('/v1/audio/speech', {
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

    const reader = response.body.getReader();
    const mediaSource = new MediaSource();
    const audio = document.createElement('audio');
    audio.src = URL.createObjectURL(mediaSource);
    
    mediaSource.addEventListener('sourceopen', async () => {
        const sourceBuffer = mediaSource.addSourceBuffer('audio/mpeg');
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            sourceBuffer.appendBuffer(value);
        }
        
        mediaSource.endOfStream();
        audio.play();
    });
}
```

## 🛠️ 故障排除

### 常见问题

1. **音频质量不佳**
   - 确保提供准确的参考文本
   - 检查上传音频的清晰度
   - 尝试不同的预设音色

2. **API响应缓慢**
   - 检查服务器资源使用情况
   - 确认MLX模型加载正常
   - 查看是否有并发请求过多

3. **音色克隆失败**
   - 验证音频文件格式（推荐WAV格式）
   - 确认参考文本与音频内容匹配
   - 检查文件大小限制

### 日志查看
```bash
# 查看服务日志
tail -f /var/log/cinecast/stream_api.log

# 健康检查
curl http://localhost:8000/health
```

## 🔒 安全建议

1. **API访问控制**
   - 在生产环境中配置适当的认证机制
   - 限制并发请求数量
   - 设置合理的请求频率限制

2. **文件上传安全**
   - 验证上传文件的格式和大小
   - 实施恶意文件检测
   - 定期清理临时文件

3. **数据隐私**
   - 敏感音频数据加密存储
   - 实施数据访问日志记录
   - 定期进行安全审计

## 📊 性能优化

### 推荐配置
- **内存**: 16GB以上
- **CPU**: 支持MLX框架的Apple Silicon芯片
- **存储**: SSD硬盘以提高I/O性能

### 监控指标
- 响应时间 < 500ms
- 并发处理能力 > 10请求/秒
- 内存使用率 < 80%

---
*文档版本: v1.0*  
*最后更新: 2026-02-07*

如需更多帮助，请联系技术支持团队。