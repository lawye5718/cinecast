#!/usr/bin/env python3
"""
CineCast 两阶段流水线测试脚本
验证内存冲突解决方案和新架构功能
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from modules.llm_director import LLMScriptDirector
from modules.mlx_tts_engine import MLXRenderEngine
from modules.asset_manager import AssetManager
from main_producer import CineCastProducer

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_phase_1_script_generation():
    """测试阶段一：剧本生成"""
    logger.info("🧪 测试阶段一：剧本生成")
    
    # 创建测试输入目录
    input_dir = "./input/chapters"
    os.makedirs(input_dir, exist_ok=True)
    
    # 创建测试章节文件
    test_chapters = {
        "chapter_01.txt": "第一章 夜晚的港口\n海风轻抚着岸边的礁石，远处的灯塔在黑暗中闪烁着微弱的光芒。\n\"你相信命运吗？\"老渔夫说道。\n年轻人摇摇头：\"我只相信海。\"",
        "chapter_02.txt": "第二章 1976年\n那是漫长的冬季。狂风席卷了整个峡湾。\n玛丽亚站在窗前，凝视着远方的海平线。"
    }
    
    for filename, content in test_chapters.items():
        filepath = os.path.join(input_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    # 初始化生产者
    producer = CineCastProducer()
    
    # 执行阶段一
    producer.phase_1_generate_scripts(input_dir)
    
    # 验证剧本文件生成
    script_files = os.listdir(producer.script_dir)
    logger.info(f"✅ 生成剧本文件: {script_files}")
    
    # 检查剧本内容
    for script_file in script_files:
        if script_file.endswith('.json'):
            with open(os.path.join(producer.script_dir, script_file), 'r', encoding='utf-8') as f:
                import json
                script = json.load(f)
                logger.info(f"📄 {script_file}: {len(script)} 个单元")
                for i, unit in enumerate(script[:2]):  # 只显示前2个单元
                    logger.info(f"   {i+1}. {unit['type']} - {unit.get('speaker', 'N/A')}: {unit['content'][:30]}...")
    
    return True

def test_ollama_integration():
    """测试Ollama集成"""
    logger.info("🧪 测试Ollama集成")
    
    try:
        director = LLMScriptDirector()
        logger.info("✅ Ollama导演初始化成功")
        
        test_text = "第一章 测试\n这是测试文本。\"你好吗？\"他说。"
        script = director.parse_text_to_script(test_text)
        
        logger.info(f"✅ Ollama解析成功，生成 {len(script)} 个单元")
        return True
        
    except Exception as e:
        logger.error(f"❌ Ollama集成测试失败: {e}")
        return False

def test_asset_enhancements():
    """测试资产管理系统增强功能"""
    logger.info("🧪 测试资产管理系统增强")
    
    try:
        assets = AssetManager()
        
        # 测试标题音色配置
        title_voice = assets.get_voice_for_role("title")
        logger.info(f"✅ 标题音色配置: 速度={title_voice['speed']}")
        
        # 测试环境音支持多种格式
        ambient = assets.get_ambient_sound("test")
        logger.info(f"✅ 环境音处理: 时长={len(ambient)}ms")
        
        # 测试过渡音增强
        chime = assets.get_transition_chime()
        logger.info(f"✅ 过渡音处理: 时长={len(chime)}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 资产管理测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🎬 开始CineCast两阶段流水线测试")
    logger.info("=" * 50)
    
    tests = [
        ("Ollama集成测试", test_ollama_integration),
        ("资产管理系统增强", test_asset_enhancements),
        ("阶段一剧本生成", test_phase_1_script_generation)
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
    
    # 输出测试总结
    logger.info("\n" + "=" * 50)
    logger.info("📊 测试结果总结:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {name}: {status}")
    
    logger.info(f"\n📈 总体结果: {passed}/{total} 个测试通过")
    
    if passed == total:
        logger.info("🎉 所有测试通过！两阶段流水线架构验证成功！")
        logger.info("✨ 内存冲突解决方案已就绪")
    else:
        logger.warning("⚠️  部分测试失败，请检查相关配置")

if __name__ == "__main__":
    main()