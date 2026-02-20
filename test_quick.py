#!/usr/bin/env python3
"""
CineCast 快速测试脚本
验证各个模块的基本功能
"""

import logging
import os
from pydub import AudioSegment

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_asset_manager():
    """测试资产管理器"""
    logger.info("🧪 测试资产管理器...")
    
    try:
        from modules.asset_manager import AssetManager
        manager = AssetManager()
        
        # 测试音色获取
        narrator_voice = manager.get_voice_for_role("narration")
        title_voice = manager.get_voice_for_role("title")
        dialogue_voice = manager.get_voice_for_role("dialogue", "张三", "male")
        
        logger.info(f"✅ 旁白音色: {narrator_voice}")
        logger.info(f"✅ 标题音色: {title_voice}")
        logger.info(f"✅ 对话音色: {dialogue_voice}")
        
        # 测试环境音
        ambient = manager.get_ambient_sound()
        chime = manager.get_transition_chime()
        
        logger.info(f"✅ 环境音时长: {len(ambient)}ms")
        logger.info(f"✅ 过渡音时长: {len(chime)}ms")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 资产管理器测试失败: {e}")
        return False

def test_llm_director():
    """测试LLM剧本导演"""
    logger.info("🧪 测试LLM剧本导演...")
    
    try:
        from modules.llm_director import LLMScriptDirector
        director = LLMScriptDirector()  # 使用降级方案测试
        
        test_text = "第一章 测试\n这是测试文本。\n\"你好吗？\"他说。\n她回答：\"我很好。\""
        
        script = director.parse_text_to_script(test_text)
        
        logger.info(f"✅ 解析完成，共 {len(script)} 个单元:")
        for i, unit in enumerate(script, 1):
            logger.info(f"   {i}. {unit['type']} - {unit.get('speaker', 'N/A')}: {unit['content'][:30]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM剧本导演测试失败: {e}")
        return False

def test_cinematic_packager():
    """测试混音打包器"""
    logger.info("🧪 测试混音打包器...")
    
    try:
        from modules.cinematic_packager import CinematicPackager
        packager = CinematicPackager("./test_output")
        
        # 创建测试音频
        test_audio = AudioSegment.silent(duration=5000)  # 5秒静音
        
        # 测试添加音频
        packager.add_audio(test_audio)
        
        # 检查状态
        status = packager.get_buffer_status()
        logger.info(f"✅ 缓冲区状态: {status}")
        
        # 测试最终化
        packager.finalize()
        
        logger.info("✅ 混音打包器测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 混音打包器测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🎬 开始CineCast快速测试")
    logger.info("=" * 50)
    
    results = []
    
    # 依次测试各个模块
    results.append(("资产管理器", test_asset_manager()))
    results.append(("LLM剧本导演", test_llm_director()))
    results.append(("混音打包器", test_cinematic_packager()))
    
    # 输出测试总结
    logger.info("\n" + "=" * 50)
    logger.info("📊 测试结果总结:")
    
    passed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\n📈 总体结果: {passed}/{len(results)} 个模块通过测试")
    
    if passed == len(results):
        logger.info("🎉 所有测试通过！CineCast系统准备就绪！")
    else:
        logger.warning("⚠️  部分测试失败，请检查相关模块")

if __name__ == "__main__":
    main()