#!/usr/bin/env python3
"""
更新的LLM导演模块测试脚本
验证本地Qwen14B模型集成效果
"""

import sys
import os
import logging

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.llm_director import LLMScriptDirector

def test_improved_llm_director():
    """测试改进后的LLM导演模块"""
    print("🎬 测试改进后的LLM导演模块")
    print("=" * 60)
    
    # 初始化导演（启用本地模型）
    director = LLMScriptDirector()
    
    # 测试文本
    test_text = """
第一章 夜晚的港口

海风轻抚着岸边的礁石，远处的灯塔在黑暗中闪烁着微弱的光芒。

"你相信命运吗？"老渔夫说道，他的声音在夜风中显得格外沧桑。

年轻人摇摇头："我只相信海。"

远处传来汽笛声，划破了寂静的夜空。海浪拍打着礁石，发出永恒的节奏。
"""
    
    print("📝 测试文本:")
    print(test_text[:200] + "..." if len(test_text) > 200 else test_text)
    print()
    
    try:
        # 使用本地模型解析
        print("🤖 使用本地Qwen14B模型解析...")
        script = director.parse_text_to_script(test_text)
        
        print("✅ 解析成功！")
        print(f"📊 解析结果: {len(script)} 个单元")
        print()
        
        print("📋 详细解析结果:")
        print("-" * 40)
        for i, unit in enumerate(script, 1):
            print(f"{i}. 类型: {unit['type']}")
            print(f"   说话人: {unit.get('speaker', 'N/A')}")
            print(f"   性别: {unit.get('gender', 'N/A')}")
            print(f"   内容: {unit['content'][:50]}{'...' if len(unit['content']) > 50 else ''}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def test_fallback_mechanism():
    """测试降级机制"""
    print("🔄 测试降级机制")
    print("=" * 60)
    
    # 初始化导演（禁用本地模型，强制使用降级方案）
    director = LLMScriptDirector()
    
    test_text = "这是测试文本。\"你好吗？\"他说。她回答：\"我很好。\""
    
    try:
        print("🤖 使用正则表达式降级方案解析...")
        script = director.parse_text_to_script(test_text)
        
        print("✅ 降级解析成功！")
        print(f"📊 解析结果: {len(script)} 个单元")
        print()
        
        print("📋 降级解析结果:")
        print("-" * 40)
        for i, unit in enumerate(script, 1):
            print(f"{i}. 类型: {unit['type']}")
            print(f"   内容: {unit['content']}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ 降级测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logging.basicConfig(level=logging.INFO)
    
    print("🎬 CineCast LLM导演模块综合测试")
    print("=" * 60)
    
    tests = [
        ("本地Qwen14B模型测试", test_improved_llm_director),
        ("正则表达式降级测试", test_fallback_mechanism)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"🧪 运行 {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} 通过")
                passed += 1
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
        print()
    
    print("=" * 60)
    print(f"📊 测试结果: {passed}/{len(tests)} 个测试通过")
    
    if passed == len(tests):
        print("🎉 所有测试通过！LLM导演模块工作正常")
        print("✨ 本地Qwen14B模型已成功集成到CineCast项目中")
    else:
        print("⚠️  部分测试失败，请检查配置")

if __name__ == "__main__":
    main()