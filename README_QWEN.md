# Qwen-Flash 配置指南

## 环境变量设置

要使用qwen-flash大模型进行剧本生成，需要设置以下环境变量：

```bash
export DASHSCOPE_API_KEY="your_dashscope_api_key"
```

## 获取API密钥

1. 访问 [阿里云百炼平台](https://dashscope.console.aliyun.com/)
2. 创建账号并获取API密钥
3. 在终端中设置环境变量

## 模型优势

- **上下文长度**: 1M tokens
- **最大输入长度**: 997k tokens  
- **最大输出长度**: 32k tokens
- **TPM**: 100万 tokens/分钟
- **结构化输出**: 原生支持JSON格式输出
- **批量推理**: 支持高效的批量处理

## 使用方法

在Web UI中选择"智能配音模式"，系统会自动使用qwen-flash模型进行剧本生成。

## 故障排除

如果遇到API连接问题，请检查：
1. `DASHSCOPE_API_KEY` 环境变量是否正确设置
2. 网络连接是否正常
3. API配额是否充足