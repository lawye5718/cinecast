#!/usr/bin/env python3
"""
Alexandria项目最终集成验证脚本
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_project_structure():
    """检查项目结构"""
    print("🔍 检查项目结构...")
    
    project_root = Path("/Users/yuanliang/superstar/superstar3.1/projects/alexandria-audiobook")
    
    required_dirs = [
        "app",
        "app/static",
        "chunks",
        "scripts",
        "voicelines",
        "designed_voices",
        "lora_models",
        "lora_datasets"
    ]
    
    required_files = [
        "app/app.py",
        "app/tts.py",
        "app/project.py",
        "app/generate_script.py",
        "README.md",
        "requirements.txt",
        "config.json"  # 新创建的配置文件
    ]
    
    missing_dirs = []
    missing_files = []
    
    for d in required_dirs:
        if not (project_root / d).exists():
            missing_dirs.append(str(project_root / d))
    
    for f in required_files:
        if not (project_root / f).exists():
            missing_files.append(str(project_root / f))
    
    if missing_dirs:
        print(f"⚠️  缺失目录: {missing_dirs}")
    else:
        print("✅ 所有必需目录存在")
    
    if missing_files:
        print(f"⚠️  缺失文件: {missing_files}")
    else:
        print("✅ 所有必需文件存在")
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def check_config_file():
    """检查配置文件"""
    print("\n🔍 检查配置文件...")
    
    config_path = Path("/Users/yuanliang/superstar/superstar3.1/projects/alexandria-audiobook/config.json")
    
    if not config_path.exists():
        print("❌ 配置文件不存在")
        return False
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查关键配置
        tts_config = config.get('tts', {})
        llm_config = config.get('llm', {})
        
        if tts_config.get('mode') == 'local':
            print("✅ TTS模式设置为本地")
        else:
            print(f"⚠️  TTS模式未设置为本地: {tts_config.get('mode')}")
        
        if 'api_url' in llm_config:
            print(f"✅ LLM API URL配置: {llm_config['api_url']}")
        else:
            print("⚠️  LLM API URL未配置")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        return False

def check_code_modifications():
    """检查代码修改"""
    print("\n🔍 检查代码修改...")
    
    # 检查project.py中的修改
    project_py_path = Path("/Users/yuanliang/superstar/superstar3.1/projects/alexandria-audiobook/app/project.py")
    
    if project_py_path.exists():
        with open(project_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否添加了串行执行相关代码
        if "serial_execution_lock" in content:
            print("✅ 串行执行锁已添加到project.py")
        else:
            print("⚠️  project.py中未找到串行执行锁")
        
        if "DEBUG:" in content:
            print("✅ 调试信息已添加到project.py")
        else:
            print("⚠️  project.py中未找到调试信息")
    else:
        print("❌ project.py文件不存在")
        return False
    
    # 检查tts.py中的修改
    tts_py_path = Path("/Users/yuanliang/superstar/superstar3.1/projects/alexandria-audiobook/app/tts.py")
    
    if tts_py_path.exists():
        with open(tts_py_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if "_serial_lock" in content:
            print("✅ 串行锁已添加到tts.py")
        else:
            print("⚠️  tts.py中未找到串行锁")
        
        if "DEBUG:" in content:
            print("✅ 调试信息已添加到tts.py")
        else:
            print("⚠️  tts.py中未找到调试信息")
    else:
        print("❌ tts.py文件不存在")
        return False
    
    return True

def check_new_files():
    """检查新创建的文件"""
    print("\n🔍 检查新创建的文件...")
    
    new_files = [
        "serial_local_llm_client.py",
        "fix_alexandria_issues.py",
        "PROJECT_FIX_REPORT.md"
    ]
    
    missing_files = []
    
    for f in new_files:
        file_path = Path(f"/Users/yuanliang/superstar/superstar3.1/projects/alexandria-audiobook/{f}")
        if file_path.exists():
            print(f"✅ 新文件已创建: {f}")
        else:
            print(f"❌ 新文件未创建: {f}")
            missing_files.append(f)
    
    return len(missing_files) == 0

def check_dependencies():
    """检查依赖"""
    print("\n🔍 检查依赖...")
    
    requirements_path = Path("/Users/yuanliang/superstar/superstar3.1/projects/alexandria-audiobook/app/requirements.txt")
    
    if requirements_path.exists():
        with open(requirements_path, 'r') as f:
            deps = f.read()
        
        required_deps = ['soundfile', 'pydub', 'numpy', 'torch', 'transformers']
        missing_deps = []
        
        for dep in required_deps:
            if dep.lower() not in deps.lower():
                missing_deps.append(dep)
        
        if missing_deps:
            print(f"⚠️  缺失依赖: {missing_deps}")
        else:
            print("✅ 所必要依赖都在requirements.txt中")
    else:
        print("⚠️  未找到requirements.txt文件")
    
    return True

def run_final_verification():
    """运行最终验证"""
    print("\n" + "="*60)
    print("🚀 Alexandria项目修复验证")
    print("="*60)
    
    checks = [
        ("项目结构", check_project_structure),
        ("配置文件", check_config_file),
        ("代码修改", check_code_modifications),
        ("新文件", check_new_files),
        ("依赖检查", check_dependencies)
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}检查出错: {e}")
            results.append((name, False))
    
    print("\n" + "="*60)
    print("📋 验证结果汇总")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所部检查通过！项目修复完成")
        print("\n下一步建议:")
        print("1. 安装依赖: pip3 install -r app/requirements.txt")
        print("2. 启动Ollama服务: ollama serve")
        print("3. 下载模型: ollama pull qwen:14b")
        print("4. 运行项目进行测试")
    else:
        print("⚠️  部分检查未通过，请检查上述问题")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = run_final_verification()
    sys.exit(0 if success else 1)