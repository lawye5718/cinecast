#!/usr/bin/env python3
"""
测试JSON解析修复效果
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from alexandria.local_llm_client import LocalLLMClient

def test_json_parsing_fix():
    """测试JSON解析修复"""
    
    print("=" * 60)
    print("🧪 测试JSON解析修复效果")
    print("=" * 60)
    
    # 配置
    config = {
        "llm": {
            "provider": "ollama",
            "model": "qwen14b-pro",
            "host": "http://localhost:11434",
            "api_url": "http://localhost:11434/api/chat",
            "temperature": 0.0,
            "num_ctx": 2048
        }
    }
    
    # 初始化客户端
    client = LocalLLMClient(config)
    
    # 测试文本
    test_text = "第一章\n夜晚的港口总是显得格外神秘。"
    
    print(f"📝 测试文本: {test_text}")
    print(f"📏 文本长度: {len(test_text)} 字符")
    
    # 生成剧本
    print("\n🧠 调用LLM生成剧本...")
    script = client.generate_script(test_text)
    
    print(f"\n📊 生成结果:")
    print(f"   返回类型: {type(script)}")
    print(f"   剧本长度: {len(script) if script else 0} 个片段")
    
    if script and len(script) > 0:
        print("✅ JSON解析成功！")
        print("\n📋 剧本内容预览:")
        for i, item in enumerate(script[:3]):
            print(f"  {i+1}. [{item['type']}] {item['speaker']}: {item['content'][:50]}...")
    else:
        print("❌ JSON解析仍然失败")
        
    return script is not None and len(script) > 0

def main():
    success = test_json_parsing_fix()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 JSON解析修复成功！")
    else:
        print("💥 JSON解析修复失败！")
    print("=" * 60)

if __name__ == "__main__":
    main()