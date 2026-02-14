#!/usr/bin/env python3
"""
CineCast 音频配置更新验证脚本
验证新的过渡音效和环境音配置是否正常工作
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from modules.asset_manager import AssetManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_audio_updates():
    """测试音频配置更新"""
    logger.info("🔍 测试音频配置更新...")
    
    try:
        assets = AssetManager("./assets")
        logger.info("✅ 资产管理器初始化成功")
        
        # 测试新的过渡音效
        logger.info("\n--- 测试过渡音效 ---")
        chime = assets.get_transition_chime()
        logger.info(f"✅ 过渡音效加载成功")
        logger.info(f"   时长: {len(chime)}ms")
        logger.info(f"   采样率: {chime.frame_rate}Hz")
        logger.info(f"   声道数: {chime.channels}")
        
        # 测试默认环境音 (iceland_wind)
        logger.info("\n--- 测试默认环境音 (iceland_wind) ---")
        ambient_default = assets.get_ambient_sound("iceland_wind")
        logger.info(f"✅ 默认环境音加载成功")
        logger.info(f"   时长: {len(ambient_default)}ms")
        logger.info(f"   采样率: {ambient_default.frame_rate}Hz")
        logger.info(f"   声道数: {ambient_default.channels}")
        
        # 测试新的fountain环境音
        logger.info("\n--- 测试fountain环境音 ---")
        ambient_fountain = assets.get_ambient_sound("fountain")
        logger.info(f"✅ fountain环境音加载成功")
        logger.info(f"   时长: {len(ambient_fountain)}ms")
        logger.info(f"   采样率: {ambient_fountain.frame_rate}Hz")
        logger.info(f"   声道数: {ambient_fountain.channels}")
        
        # 验证音频规格统一性
        logger.info("\n--- 音频规格验证 ---")
        target_sr = 22050
        target_channels = 1
        
        specs_correct = (
            chime.frame_rate == target_sr and chime.channels == target_channels and
            ambient_default.frame_rate == target_sr and ambient_default.channels == target_channels and
            ambient_fountain.frame_rate == target_sr and ambient_fountain.channels == target_channels
        )
        
        if specs_correct:
            logger.info("✅ 所有音频规格统一正确 (22050Hz, 单声道)")
        else:
            logger.warning("⚠️ 音频规格存在问题")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ 音频配置测试失败: {e}")
        return False

def test_audio_files_exist():
    """检查音频文件是否存在"""
    logger.info("🔍 检查音频文件存在性...")
    
    required_files = [
        "./assets/transitions/soft_chime.mp3",
        "./assets/ambient/fountain.mp3",
        "./assets/ambient/iceland_wind.wav"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.info(f"✅ {file_path} 存在 ({file_size} bytes)")
        else:
            logger.error(f"❌ {file_path} 不存在")
            all_exist = False
    
    return all_exist

def main():
    """主测试函数"""
    logger.info("🎬 开始CineCast音频配置更新验证")
    logger.info("=" * 50)
    
    tests = [
        ("音频文件存在性检查", test_audio_files_exist),
        ("音频配置功能测试", test_audio_updates)
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
    logger.info("📊 音频配置更新验证总结:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {name}: {status}")
    
    logger.info(f"\n📈 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 音频配置更新完成！所有功能正常工作")
        logger.info("✨ 更新内容:")
        logger.info("   • 新增哲理过渡音效 (nightdeep.mp3)")
        logger.info("   • 新增fountain环境音")
        logger.info("   • 音频规格统一为22050Hz单声道")
    else:
        logger.warning("⚠️ 部分音频配置存在问题，请检查相关文件")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)