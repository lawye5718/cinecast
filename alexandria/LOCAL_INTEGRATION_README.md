# Alexandria本地化集成指南

本项目将CineCast中测试通过的本地MLX Qwen模型集成到Alexandria有声书生成器中。

## 🎯 集成组件

### 1. 本地LLM客户端 (`local_llm_client.py`)
- 集成Ollama Qwen14B-Pro模型
- 使用CineCast中验证的强化System Prompt
- 支持剧本生成、上下文保持、智能分块
- 包含JSON修复和正则降级机制

### 2. 本地TTS引擎 (`local_tts_engine.py`)
- 集成MLX Qwen-TTS模型
- 基于CineCast中验证的渲染实现
- 支持情感化语音合成（预留接口）
- 优化的内存管理和清理策略

### 3. 本地化适配器 (`integrate_local_components.py`)
- 统一管理本地化组件
- 提供完整的处理流程
- 包含健康检查和测试功能

## 📋 使用方法

### 1. 配置文件 (`local_config.json`)
```json
{
    "llm": {
        "provider": "ollama",
        "model": "qwen14b-pro",
        "host": "http://localhost:11434",
        "api_url": "http://localhost:11434/api/chat",
        "temperature": 0.0,
        "num_ctx": 8192
    },
    "tts": {
        "mode": "local",
        "model_path": "../qwentts/models/Qwen3-TTS-MLX-0.6B",
        "device": "metal",
        "compile_codec": false,
        "language": "Chinese"
    }
}
```

### 2. 基本使用
```python
from integrate_local_components import AlexandriaLocalAdapter

# 初始化适配器
adapter = AlexandriaLocalAdapter()

# 健康检查
health_status = adapter.health_check()
print(health_status)

# 处理文本
text = "第一章 测试\n这是测试内容..."
script = adapter.generate_local_script(text)
print(f"生成 {len(script)} 个剧本片段")

# 渲染音频
success = adapter.render_local_audio("测试文本", {"speaker": "test"}, "output.wav")
```

### 3. 完整流程处理
```python
# 处理完整书籍片段
success = adapter.process_book_chunk(
    text_chunk="书籍内容...",
    chunk_id="001",
    output_dir="./output",
    context="上下文信息..."
)
```

## 🔧 技术特点

### LLM集成优势
- ✅ 使用CineCast中验证的Qwen14B-Pro模型
- ✅ 强化的System Prompt确保输出质量
- ✅ 完善的错误处理和降级机制
- ✅ 支持上下文保持和智能分块

### TTS集成优势
- ✅ 基于MLX框架的高效推理
- ✅ 与CineCast相同的渲染质量
- ✅ 优化的内存管理策略
- ✅ 情感化语音合成预留接口

### 系统集成优势
- ✅ 模块化设计，易于维护
- ✅ 完整的健康检查机制
- ✅ 详细的日志记录
- ✅ 兼容Alexandria原有架构

## 🚀 性能优化

### 内存管理
- 借鉴CineCast的显存回收策略
- 适度使用垃圾回收，避免频繁调用
- 支持模型按需加载和卸载

### 处理效率
- 支持批量处理和并行渲染
- 智能的任务调度和资源分配
- 断点续传和错误恢复机制

## 📊 测试验证

运行集成测试：
```bash
python integrate_local_components.py
```

预期输出：
```
🏥 健康检查结果:
  ollama_connection: True
  tts_engine_available: True
  config_loaded: True
  components_initialized: True
  overall_status: ✅ 正常

🧪 简单功能测试:
  ✅ 剧本生成成功: X 个片段
  ✅ 本地化集成测试通过!
```

## ⚠️ 注意事项

1. 确保Ollama服务正常运行
2. 确认MLX框架和相关依赖已正确安装
3. 检查模型路径配置是否正确
4. 根据硬件配置调整batch size和内存设置

## 📚 参考资料

- [CineCast项目](https://github.com/lawye5718/cinecast)
- [Alexandria项目](https://github.com/Finrandojin/alexandria-audiobook)
- [MLX框架文档](https://ml-explore.github.io/mlx/)
- [Qwen-TTS模型](https://github.com/QwenLM/Qwen3-TTS)

---
*基于CineCast v1.0 和 Alexandria v1.0 集成*
*最后更新: 2026-02-14*