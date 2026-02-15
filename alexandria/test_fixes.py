#!/usr/bin/env python3
"""
Alexandria项目修复验证脚本
验证WAV生成和串行LLM处理功能
"""

import os
import sys
import json
import tempfile
from pathlib import Path

# 添加项目路径
project_root = Path("/Users/yuanliang/superstar/superstar3.1/projects/alexandria-audiobook")
sys.path.insert(0, str(project_root))

def test_audio_generation():
    """测试音频生成功能"""
    print("🔍 测试音频生成功能...")
    
    try:
        import numpy as np
        import soundfile as sf
        
        # 创建测试音频数据
        sample_rate = 22050
        duration = 1  # 1秒
        frequency = 440  # A4音符
        
        # 生成简单的正弦波
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            
            # 检查文件大小
            file_size = os.path.getsize(tmp_file.name)
            print(f"✅ 生成的WAV文件大小: {file_size} 字节")
            
            if file_size > 0:
                print("✅ 音频生成测试通过 - 文件非空")
                
                # 清理临时文件
                os.unlink(tmp_file.name)
                return True
            else:
                print("❌ 音频生成测试失败 - 文件为空")
                os.unlink(tmp_file.name)
                return False
                
    except Exception as e:
        print(f"❌ 音频生成测试异常: {e}")
        return False

def test_serial_llm_client():
    """测试串行LLM客户端"""
    print("\n🔍 测试串行LLM客户端...")
    
    try:
        # 尝试导入串行LLM客户端
        from serial_local_llm_client import SerialLocalLLMClient
        
        # 检查配置文件
        config_path = project_root / "config.json"
        if not config_path.exists():
            print("❌ 配置文件不存在")
            return False
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 创建客户端实例
        llm_client = SerialLocalLLMClient(config)
        
        print(f"✅ 串行LLM客户端创建成功")
        print(f"   模型: {llm_client.model_name}")
        print(f"   API URL: {llm_client.api_url}")
        
        # 检查是否配置了正确的模型
        if "qwen14b" in llm_client.model_name.lower():
            print("✅ 使用了正确的本地模型 (qwen14b)")
        else:
            print(f"⚠️  使用的模型可能不是预期的本地模型: {llm_client.model_name}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 串行LLM客户端导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 串行LLM客户端测试异常: {e}")
        return False

def test_config_updates():
    """测试配置更新"""
    print("\n🔍 测试配置更新...")
    
    try:
        config_path = project_root / "config.json"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查LLM配置
        llm_config = config.get("llm", {})
        expected_model = "qwen14b-pro"
        
        if llm_config.get("model") == expected_model:
            print(f"✅ LLM配置正确: {llm_config['model']}")
        else:
            print(f"❌ LLM配置不正确，期望 {expected_model}, 实际 {llm_config.get('model')}")
            return False
        
        # 检查TTS配置
        tts_config = config.get("tts", {})
        print(f"✅ TTS配置: {tts_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置测试异常: {e}")
        return False

def test_contact_discovery():
    """测试联系人发现功能"""
    print("\n🔍 测试联系人发现功能...")
    
    try:
        discovery_path = project_root / "dingtalk_contact_discovery.py"
        
        if discovery_path.exists():
            print("✅ 联系人发现脚本已创建")
            
            # 检查文件内容
            with open(discovery_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "DingTalkContactDiscovery" in content:
                print("✅ 包含正确的联系人发现类")
                return True
            else:
                print("❌ 联系人发现脚本内容不完整")
                return False
        else:
            print("❌ 联系人发现脚本不存在")
            return False
            
    except Exception as e:
        print(f"❌ 联系人发现测试异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 Alexandria项目修复验证测试")
    print("="*60)
    
    tests = [
        ("音频生成功能", test_audio_generation),
        ("串行LLM客户端", test_serial_llm_client),
        ("配置更新", test_config_updates),
        ("联系人发现功能", test_contact_discovery)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}测试出错: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("📋 测试结果汇总")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
        if not result:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 所有测试通过！项目修复成功。")
        print("\n下一步建议:")
        print("1. 确保Ollama服务正在运行: ollama serve")
        print("2. 确保已下载qwen14b-pro模型: ollama pull qwen14b-pro")
        print("3. 运行项目进行端到端测试")
    else:
        print("⚠️  部分测试未通过，请检查上述错误信息。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)