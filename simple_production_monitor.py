#!/usr/bin/env python3
"""
简单生产测试监控脚本
记录基本的测试信息和系统状态
"""

import os
import time
import psutil
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler('production_test_monitor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleMonitor:
    def __init__(self):
        self.start_time = datetime.now()
        self.output_dir = "./output/Audiobooks"
        
    def get_system_status(self):
        """获取系统基本状态"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'available_memory_gb': round(memory.available / (1024**3), 2)
        }
    
    def get_file_counts(self):
        """获取各目录文件数量"""
        counts = {
            'scripts': 0,
            'temp_wav_cache': 0,
            'final_output': 0
        }
        
        scripts_dir = os.path.join(self.output_dir, "scripts")
        cache_dir = os.path.join(self.output_dir, "temp_wav_cache")
        output_dir = os.path.join(self.output_dir, "final_output")
        
        if os.path.exists(scripts_dir):
            counts['scripts'] = len([f for f in os.listdir(scripts_dir) if f.endswith('.json')])
            
        if os.path.exists(cache_dir):
            counts['temp_wav_cache'] = len([f for f in os.listdir(cache_dir) if f.endswith('.wav')])
            
        if os.path.exists(output_dir):
            counts['final_output'] = len([f for f in os.listdir(output_dir) if f.endswith('.mp3')])
            
        return counts
    
    def monitor_loop(self):
        """监控循环"""
        logger.info("🔍 开始生产测试监控...")
        logger.info(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 50)
        
        try:
            while True:
                # 获取系统状态
                sys_status = self.get_system_status()
                file_counts = self.get_file_counts()
                
                # 记录状态
                elapsed_time = datetime.now() - self.start_time
                logger.info(f"⏱️  运行时间: {str(elapsed_time).split('.')[0]}")
                logger.info(f"📊 系统状态 - CPU: {sys_status['cpu_percent']:.1f}%, 内存: {sys_status['memory_percent']:.1f}%")
                logger.info(f"📁 文件统计 - 剧本: {file_counts['scripts']}个, WAV: {file_counts['temp_wav_cache']}个, 成品: {file_counts['final_output']}个")
                logger.info("-" * 30)
                
                # 检查是否有错误日志
                if os.path.exists('cinecast.log'):
                    with open('cinecast.log', 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        error_lines = [line for line in lines[-10:] if 'ERROR' in line or '❌' in line]
                        if error_lines:
                            logger.warning("⚠️  发现错误信息:")
                            for error_line in error_lines[-3:]:  # 只显示最近3个错误
                                logger.warning(f"  {error_line.strip()}")
                
                time.sleep(300)  # 每5分钟检查一次
                
        except KeyboardInterrupt:
            logger.info("🛑 监控被用户中断")
        except Exception as e:
            logger.error(f"监控过程中出现错误: {e}")

def main():
    """主函数"""
    monitor = SimpleMonitor()
    monitor.monitor_loop()

if __name__ == "__main__":
    main()