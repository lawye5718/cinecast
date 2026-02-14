#!/usr/bin/env python3
"""
CineCast 本地测试脚本
用于验证系统基本功能是否正常工作
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from modules.llm_director import LLMScriptDirector
from modules.asset_manager import AssetManager
from modules.mlx_tts_engine import MLXRenderEngine
from modules.cinematic_packager import CinematicPackager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('local_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_basic_components():
    """测试基本组件初始化"""
    logger.info("🔍 测试基本组件初始化...")
    
    try:
        # 测试资产管理系统
        assets = AssetManager("./assets")
        logger.info("✅ 资产管理系统初始化成功")
        
        # 测试各种音色获取
        narrator_voice = assets.get_voice_for_role("narration")
        title_voice = assets.get_voice_for_role("title")
        dialogue_voice = assets.get_voice_for_role("dialogue", "测试角色", "male")
        
        logger.info(f"✅ 旁白音色速度: {narrator_voice['speed']}")
        logger.info(f"✅ 标题音色速度: {title_voice['speed']}")
        logger.info(f"✅ 对话音色速度: {dialogue_voice['speed']}")
        
        # 测试环境音和过渡音
        ambient = assets.get_ambient_sound()
        chime = assets.get_transition_chime()
        logger.info(f"✅ 环境音时长: {len(ambient)}ms")
        logger.info(f"✅ 过渡音时长: {len(chime)}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 基本组件测试失败: {e}")
        return False

def test_ollama_integration():
    """测试Ollama集成"""
    logger.info("🔍 测试Ollama集成...")
    
    try:
        director = LLMScriptDirector(use_local_mlx_lm=True)
        logger.info("✅ Ollama导演模块初始化成功")
        
        # 测试短文本解析
        test_text = "第一章 测试\n这是测试内容。\"你好世界！\"他说。"
        script = director.parse_text_to_script(test_text)
        
        logger.info(f"✅ Ollama解析成功，生成 {len(script)} 个单元")
        for i, unit in enumerate(script[:3]):
            logger.info(f"   单元{i+1}: {unit['type']} - {unit.get('speaker', 'N/A')}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ollama集成测试失败: {e}")
        return False

def test_script_generation():
    """测试剧本生成功能"""
    logger.info("🔍 测试剧本生成功能...")
    
    try:
        # 创建测试输出目录
        script_dir = "./output/local_test/scripts"
        os.makedirs(script_dir, exist_ok=True)
        
        director = LLMScriptDirector(use_local_mlx_lm=True)
        
        # 读取测试章节
        test_files = ["./input_chapters/chapter_01.txt", "./input_chapters/chapter_02.txt"]
        
        for file_path in test_files:
            if not os.path.exists(file_path):
                logger.warning(f"⚠️ 测试文件不存在: {file_path}")
                continue
                
            chapter_name = os.path.splitext(os.path.basename(file_path))[0]
            script_path = os.path.join(script_dir, f"{chapter_name}.json")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"📝 处理章节: {chapter_name} ({len(content)}字符)")
            script = director.parse_text_to_script(content)
            
            with open(script_path, 'w', encoding='utf-8') as f:
                json.dump(script, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 生成剧本: {len(script)} 个单元")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 剧本生成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🎬 开始CineCast本地测试")
    logger.info("=" * 50)
    
    tests = [
        ("基本组件测试", test_basic_components),
        ("Ollama集成测试", test_ollama_integration),
        ("剧本生成测试", test_script_generation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"{status} {test_name}")
        except Exception as e:
            logger.error(f"❌ {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 输出总结
    logger.info("\n" + "=" * 50)
    logger.info("📊 本地测试总结:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {name}: {status}")
    
    logger.info(f"\n📈 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 本地测试完成！系统基本功能正常")
        logger.info("💡 现在可以运行完整流程进行实际测试")
    else:
        logger.warning("⚠️ 部分功能存在问题，请检查相关配置")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)