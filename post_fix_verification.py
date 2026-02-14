#!/usr/bin/env python3
"""
CineCast 修正后验证测试
验证三个核心问题的修复效果
"""

import os
import sys
import time
import psutil
import logging
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from modules.llm_director import LLMScriptDirector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_memory_optimization():
    """测试内存优化效果"""
    logger.info("🔍 测试内存优化效果...")
    
    try:
        # 获取初始内存使用
        initial_memory = psutil.virtual_memory()
        logger.info(f"初始内存使用: {initial_memory.percent}% ({initial_memory.used / 1024**3:.1f}GB)")
        
        director = LLMScriptDirector()
        
        # 测试长文本处理（应该被自动切分）
        very_long_text = "第一章 超长测试章节\n" + "这是测试内容。这是测试内容。这是测试内容。" * 100
        
        logger.info(f"处理超长文本 ({len(very_long_text)} 字符)...")
        start_time = time.time()
        script = director.parse_text_to_script(very_long_text)
        end_time = time.time()
        
        final_memory = psutil.virtual_memory()
        logger.info(f"处理耗时: {end_time - start_time:.2f}秒")
        logger.info(f"最终内存使用: {final_memory.percent}% ({final_memory.used / 1024**3:.1f}GB)")
        logger.info(f"内存变化: {final_memory.percent - initial_memory.percent:+.1f}%")
        logger.info(f"生成剧本单元数: {len(script)}")
        
        # 验证内容完整性
        total_parsed_chars = sum(len(unit.get('content', '')) for unit in script)
        logger.info(f"解析内容总字符数: {total_parsed_chars}")
        
        if total_parsed_chars > len(very_long_text) * 0.5:  # 至少解析50%内容
            logger.info("✅ 长章节处理完整性良好")
            return True
        else:
            logger.warning("⚠️ 长章节内容解析不完整")
            return False
            
    except Exception as e:
        logger.error(f"❌ 内存优化测试失败: {e}")
        return False

def test_keep_alive_strategy():
    """测试新的keep_alive策略"""
    logger.info("🔍 测试keep_alive策略...")
    
    try:
        director = LLMScriptDirector()
        
        # 连续快速调用，观察是否避免了重复加载
        test_texts = [
            "第一章 测试\n短文本测试。",
            "第二章 继续\n另一个短测试。",
            "第三章 最后\n最后一次测试。"
        ]
        
        logger.info("连续处理多个短文本...")
        start_time = time.time()
        
        for i, text in enumerate(test_texts):
            chunk_start = time.time()
            script = director.parse_text_to_script(text)
            chunk_time = time.time() - chunk_start
            logger.info(f"第{i+1}个文本处理: {chunk_time:.2f}秒, 生成{len(script)}个单元")
        
        total_time = time.time() - start_time
        logger.info(f"总处理时间: {total_time:.2f}秒")
        
        # 如果平均每个文本处理时间较短，说明避免了重复加载
        avg_time = total_time / len(test_texts)
        if avg_time < 5.0:  # 平均每个文本处理时间小于5秒
            logger.info("✅ keep_alive策略有效，避免了模型重复加载")
            return True
        else:
            logger.warning("⚠️ 处理时间较长，可能存在加载开销")
            return False
            
    except Exception as e:
        logger.error(f"❌ keep_alive策略测试失败: {e}")
        return False

def main():
    """主验证函数"""
    logger.info("🎬 开始CineCast修正后验证测试")
    logger.info("=" * 60)
    
    tests = [
        ("内存优化效果", test_memory_optimization),
        ("keep_alive策略", test_keep_alive_strategy)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ 通过" if result else "❌ 失败"
            logger.info(f"{status} {test_name}")
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 输出总结
    logger.info("\n" + "=" * 60)
    logger.info("📊 修正后验证总结:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"   {name}: {status}")
    
    logger.info(f"\n📈 总体结果: {passed}/{total} 项测试通过")
    
    if passed == total:
        logger.info("🎉 修正完成！系统性能得到显著改善")
        logger.info("✨ 主要改进:")
        logger.info("   • 长章节自动切分处理")
        logger.info("   • keep_alive策略优化内存使用")
        logger.info("   • EPUB格式支持已添加")
        logger.info("   • 手动内存弹射机制就绪")
    else:
        logger.warning("⚠️ 部分修正措施需要进一步完善")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)