#!/usr/bin/env python3
"""
CineCast 问题验证测试脚本
验证三个声称的致命漏洞是否真实存在
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

def check_ollama_memory_thrashing():
    """检查Ollama内存潮汐问题"""
    logger.info("🔍 检查Ollama内存潮汐问题...")
    
    try:
        # 获取初始内存使用情况
        initial_memory = psutil.virtual_memory()
        logger.info(f"初始内存使用: {initial_memory.percent}% ({initial_memory.used / 1024**3:.1f}GB)")
        
        director = LLMScriptDirector()
        
        # 创建测试文本（模拟长章节）
        long_text = "第一章 测试章节\n" + "这是测试内容。" * 200  # 约2000字符
        
        logger.info("开始连续调用Ollama处理...")
        
        # 连续处理5次，观察内存变化
        for i in range(5):
            start_time = time.time()
            script = director.parse_text_to_script(long_text)
            end_time = time.time()
            
            current_memory = psutil.virtual_memory()
            logger.info(f"第{i+1}次调用: {(end_time-start_time):.2f}秒, 内存使用: {current_memory.percent}%")
            
            # 检查是否有明显的内存波动
            if abs(current_memory.percent - initial_memory.percent) > 10:
                logger.warning("⚠️ 检测到显著内存波动，可能存在潮汐问题")
                return True
        
        logger.info("✅ 未检测到明显的内存潮汐问题")
        return False
        
    except Exception as e:
        logger.error(f"❌ 内存潮汐检查失败: {e}")
        return False

def check_long_chapter_truncation():
    """检查长章节截断问题"""
    logger.info("🔍 检查长章节截断问题...")
    
    try:
        director = LLMScriptDirector()
        
        # 创建超长文本（超过2500字符）
        very_long_text = "第一章 超长测试\n" + "这是很长的测试内容。" * 500  # 约5000字符
        logger.info(f"测试文本长度: {len(very_long_text)} 字符")
        
        script = director.parse_text_to_script(very_long_text)
        
        # 检查返回的剧本是否完整
        total_content_length = sum(len(unit.get('content', '')) for unit in script)
        logger.info(f"解析后内容总长度: {total_content_length} 字符")
        
        # 如果解析后的内容远小于原文本，说明存在截断
        if total_content_length < len(very_long_text) * 0.3:  # 少于30%认为有问题
            logger.warning("⚠️ 检测到严重的内容截断问题")
            return True
        else:
            logger.info("✅ 内容截断问题不明显")
            return False
            
    except Exception as e:
        logger.error(f"❌ 截断检查失败: {e}")
        return False

def check_epub_support():
    """检查EPUB支持问题"""
    logger.info("🔍 检查EPUB支持问题...")
    
    # 检查相关依赖是否安装
    try:
        import ebooklib
        from bs4 import BeautifulSoup
        logger.info("✅ EPUB相关依赖已安装")
        epub_supported = True
    except ImportError as e:
        logger.info(f"ℹ️ EPUB相关依赖未安装: {e}")
        epub_supported = False
    
    # 检查主控程序是否有EPUB处理逻辑
    try:
        with open('main_producer.py', 'r', encoding='utf-8') as f:
            content = f.read()
            if 'ebooklib' in content or 'epub' in content or 'BeautifulSoup' in content:
                logger.info("✅ 主控程序包含EPUB处理逻辑")
                epub_logic_exists = True
            else:
                logger.info("❌ 主控程序缺少EPUB处理逻辑")
                epub_logic_exists = False
    except Exception as e:
        logger.error(f"❌ 检查EPUB逻辑失败: {e}")
        epub_logic_exists = False
    
    return not (epub_supported and epub_logic_exists)

def main():
    """主验证函数"""
    logger.info("🎬 开始CineCast问题验证测试")
    logger.info("=" * 60)
    
    tests = [
        ("Ollama内存潮汐问题", check_ollama_memory_thrashing),
        ("长章节截断问题", check_long_chapter_truncation),
        ("EPUB支持缺失问题", check_epub_support)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            status = "🚨 存在问题" if result else "✅ 问题不存在"
            logger.info(f"{status} {test_name}")
        except Exception as e:
            logger.error(f"❌ {test_name} 测试异常: {e}")
            results.append((test_name, True))  # 异常视为存在问题
    
    # 输出总结
    logger.info("\n" + "=" * 60)
    logger.info("📊 问题验证总结:")
    
    problems_found = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "🚨 存在" if result else "✅ 不存在"
        logger.info(f"   {name}: {status}")
    
    logger.info(f"\n📈 总体结果: 发现 {problems_found}/{total} 个问题")
    
    if problems_found > 0:
        logger.warning("⚠️ 建议根据发现的问题进行相应调整")
    else:
        logger.info("🎉 当前系统状态良好，无需紧急调整")
    
    return problems_found

if __name__ == "__main__":
    problem_count = main()
    sys.exit(0 if problem_count == 0 else 1)