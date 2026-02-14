#!/usr/bin/env python3
"""
测试监控脚本 - 定期记录《鱼没有脚》生产测试状态
"""

import os
import time
import json
import psutil
import logging
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_monitoring.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestMonitor:
    def __init__(self, test_log_file="yu_meiyou_jiao_full_test.log"):
        self.test_log_file = test_log_file
        self.monitor_log = "monitoring_status.log"
        self.output_base = "./output/yu_meiyou_jiao_production"
        
    def get_system_status(self):
        """获取系统状态"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_used_percent': round((disk.used / disk.total) * 100, 2),
                'active_processes': len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {}
    
    def get_test_progress(self):
        """获取测试进度信息"""
        progress_info = {
            'timestamp': datetime.now().isoformat(),
            'directories': {},
            'files_count': {}
        }
        
        try:
            # 检查各目录文件数量
            dirs_to_check = [
                'scripts',
                'temp_wav_cache', 
                'final_audiobooks'
            ]
            
            for dir_name in dirs_to_check:
                dir_path = os.path.join(self.output_base, dir_name)
                if os.path.exists(dir_path):
                    files = [f for f in os.listdir(dir_path) if not f.startswith('.')]
                    progress_info['files_count'][dir_name] = len(files)
                    progress_info['directories'][dir_name] = dir_path
                else:
                    progress_info['files_count'][dir_name] = 0
                    
        except Exception as e:
            logger.error(f"获取测试进度失败: {e}")
            
        return progress_info
    
    def check_test_log(self):
        """检查测试日志中的关键信息"""
        if not os.path.exists(self.test_log_file):
            return "测试日志文件不存在"
            
        try:
            with open(self.test_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # 获取最后几行日志
            recent_lines = lines[-10:] if len(lines) > 10 else lines
            
            # 查找关键状态信息
            status_indicators = []
            for line in recent_lines:
                if any(keyword in line for keyword in ['阶段', '完成', '错误', '失败', '开始']):
                    status_indicators.append(line.strip())
                    
            return status_indicators[-3:] if status_indicators else ["暂无关键状态信息"]
            
        except Exception as e:
            return f"读取测试日志失败: {e}"
    
    def record_monitoring_data(self):
        """记录监控数据"""
        monitoring_record = {
            'system_status': self.get_system_status(),
            'test_progress': self.get_test_progress(),
            'recent_log_entries': self.check_test_log()
        }
        
        # 保存到监控日志
        with open(self.monitor_log, 'a', encoding='utf-8') as f:
            f.write(f"\n=== 监控记录 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(json.dumps(monitoring_record, ensure_ascii=False, indent=2))
            f.write("\n" + "="*50 + "\n")
            
        # 打印摘要信息
        sys_status = monitoring_record['system_status']
        progress = monitoring_record['test_progress']
        
        logger.info(f"📊 系统状态 - CPU: {sys_status.get('cpu_percent', 'N/A')}%, "
                   f"内存: {sys_status.get('memory_percent', 'N/A')}%, "
                   f"磁盘: {sys_status.get('disk_used_percent', 'N/A')}%")
        
        logger.info(f"📁 文件统计 - 剧本: {progress['files_count'].get('scripts', 0)}, "
                   f"WAV缓存: {progress['files_count'].get('temp_wav_cache', 0)}, "
                   f"成品: {progress['files_count'].get('final_audiobooks', 0)}")
        
        # 显示最近的日志条目
        recent_entries = monitoring_record['recent_log_entries']
        if recent_entries and isinstance(recent_entries, list):
            logger.info("📝 最近日志:")
            for entry in recent_entries[-2:]:  # 只显示最后2条
                logger.info(f"  {entry}")
    
    def monitor_loop(self, interval_minutes=5):
        """监控循环"""
        logger.info("🔍 开始测试监控...")
        logger.info(f"🕒 监控间隔: {interval_minutes}分钟")
        logger.info("="*50)
        
        try:
            while True:
                self.record_monitoring_data()
                logger.info(f"💤 等待 {interval_minutes} 分钟后下次检查...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            logger.info("🛑 监控被用户中断")
        except Exception as e:
            logger.error(f"监控过程中出现错误: {e}")

def main():
    """主函数"""
    monitor = TestMonitor()
    monitor.monitor_loop(interval_minutes=5)  # 每5分钟检查一次

if __name__ == "__main__":
    main()