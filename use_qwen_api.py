#!/usr/bin/env python3
"""
Qwen API调用示例 - 使用已保存的配置
"""

import json
import requests

def load_qwen_config():
    """加载Qwen API配置"""
    try:
        with open('./qwen_api_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise Exception("配置文件不存在，请先运行测试脚本")

def call_qwen_api(prompt: str, **kwargs) -> str:
    """调用Qwen API"""
    config = load_qwen_config()
    
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json"
    }
    
    # 默认参数
    payload = {
        "model": config['model'],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    # 更新用户提供的参数
    payload.update(kwargs)
    
    try:
        response = requests.post(
            f"{config['base_url']}/chat/completions",
            headers=headers,
            json=payload,
            timeout=config['timeout']
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")
            
    except Exception as e:
        raise Exception(f"调用过程中出错: {e}")

# 使用示例
if __name__ == "__main__":
    try:
        print("🚀 使用已保存的配置调用Qwen API")
        print("-" * 40)
        
        # 简单测试
        response = call_qwen_api("请用一句话介绍你自己")
        print(f"🤖 AI回复: {response}")
        
        print("-" * 40)
        print("✅ 调用成功! 配置已自动加载")
        
    except Exception as e:
        print(f"❌ 调用失败: {e}")