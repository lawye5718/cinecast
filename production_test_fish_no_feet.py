#!/usr/bin/env python3
"""
CineCast 生产测试脚本 - 《鱼没有脚》完整流程测试
记录详细的运行信息、CPU使用率、内存使用等关键数据
"""

import os
import sys
import json
import time
import psutil
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 配置详细的日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_test_detailed.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionTestMonitor:
    def __init__(self):
        self.start_time = None
        self.system_metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'network_io': []
        }
        self.test_results = {}
        
    def start_monitoring(self):
        """开始系统监控"""
        self.start_time = time.time()
        logger.info("🔬 开始系统性能监控...")
        
    def collect_metrics(self, stage=""):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用
            memory = psutil.virtual_memory()
            
            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            
            # 网络IO
            net_io = psutil.net_io_counters()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'stage': stage,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_read_mb': round(disk_io.read_bytes / (1024**2), 2) if disk_io else 0,
                'disk_write_mb': round(disk_io.write_bytes / (1024**2), 2) if disk_io else 0,
                'network_sent_mb': round(net_io.bytes_sent / (1024**2), 2) if net_io else 0,
                'network_recv_mb': round(net_io.bytes_recv / (1024**2), 2) if net_io else 0
            }
            
            self.system_metrics['cpu_usage'].append(cpu_percent)
            self.system_metrics['memory_usage'].append(memory.percent)
            
            logger.info(f"[{stage}] CPU: {cpu_percent}% | 内存: {memory.percent}% ({memory.used/1024/1024:.0f}MB)")
            
            return metrics
            
        except Exception as e:
            logger.error(f"收集系统指标时出错: {e}")
            return {}

    def run_production_test(self):
        """运行完整的生产测试"""
        logger.info("=" * 60)
        logger.info("🎬 开始《鱼没有脚》生产测试")
        logger.info("=" * 60)
        
        # 开始监控
        self.start_monitoring()
        self.collect_metrics("测试开始")
        
        try:
            # 导入主控程序
            from main_producer import CineCastProducer
            
            # 创建生产者实例
            logger.info("🔧 初始化CineCast生产线...")
            producer = CineCastProducer()
            self.collect_metrics("初始化完成")
            
            # 获取EPUB文件路径（通过环境变量或默认路径）
            epub_path = os.environ.get("CINECAST_EPUB_PATH", "./input/test.epub")
            
            if not os.path.exists(epub_path):
                raise FileNotFoundError(f"EPUB文件不存在: {epub_path}，请设置 CINECAST_EPUB_PATH 环境变量")
            
            logger.info(f"📚 使用EPUB文件: {epub_path}")
            logger.info(f"📁 文件大小: {os.path.getsize(epub_path) / (1024*1024):.2f} MB")
            
            # 记录开始时间
            test_start_time = time.time()
            
            # 阶段一：剧本生成
            logger.info("\n" + "="*50)
            logger.info("🎬 [阶段一] 剧本生成阶段开始")
            logger.info("="*50)
            
            self.collect_metrics("阶段一开始")
            
            phase1_start = time.time()
            success = producer.phase_1_generate_scripts(epub_path)
            phase1_end = time.time()
            
            self.collect_metrics("阶段一结束")
            
            if not success:
                raise Exception("阶段一剧本生成失败")
            
            phase1_duration = phase1_end - phase1_start
            logger.info(f"⏱️ 阶段一耗时: {phase1_duration:.2f} 秒")
            
            # 阶段二：音频渲染
            logger.info("\n" + "="*50)
            logger.info("🎬 [阶段二] 音频渲染阶段开始")
            logger.info("="*50)
            
            self.collect_metrics("阶段二开始")
            
            phase2_start = time.time()
            producer.phase_2_render_audio()
            phase2_end = time.time()
            
            self.collect_metrics("阶段二结束")
            
            phase2_duration = phase2_end - phase2_start
            logger.info(f"⏱️ 阶段二耗时: {phase2_duration:.2f} 秒")
            
            # 总体统计
            total_duration = time.time() - test_start_time
            self.collect_metrics("测试完成")
            
            # 收集最终结果
            self.test_results = {
                'test_start_time': datetime.fromtimestamp(test_start_time).isoformat(),
                'test_end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'phase1_duration_seconds': phase1_duration,
                'phase2_duration_seconds': phase2_duration,
                'epub_file': epub_path,
                'epub_size_mb': os.path.getsize(epub_path) / (1024*1024),
                'output_directory': producer.config["output_dir"],
                'success': True
            }
            
            logger.info("\n" + "="*60)
            logger.info("🎉 生产测试完成!")
            logger.info("="*60)
            logger.info(f"📊 总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
            logger.info(f"📊 阶段一耗时: {phase1_duration:.2f} 秒")
            logger.info(f"📊 阶段二耗时: {phase2_duration:.2f} 秒")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 生产测试失败: {e}")
            self.test_results['success'] = False
            self.test_results['error'] = str(e)
            return False

    def generate_report(self):
        """生成详细的测试报告"""
        logger.info("\n" + "="*60)
        logger.info("📊 生成测试报告...")
        logger.info("="*60)
        
        # 计算系统性能统计
        if self.system_metrics['cpu_usage']:
            avg_cpu = sum(self.system_metrics['cpu_usage']) / len(self.system_metrics['cpu_usage'])
            max_cpu = max(self.system_metrics['cpu_usage'])
        else:
            avg_cpu = max_cpu = 0
            
        if self.system_metrics['memory_usage']:
            avg_memory = sum(self.system_metrics['memory_usage']) / len(self.system_metrics['memory_usage'])
            max_memory = max(self.system_metrics['memory_usage'])
        else:
            avg_memory = max_memory = 0
        
        # 生成报告内容
        report_data = {
            'test_summary': self.test_results,
            'system_performance': {
                'average_cpu_usage': round(avg_cpu, 2),
                'peak_cpu_usage': round(max_cpu, 2),
                'average_memory_usage': round(avg_memory, 2),
                'peak_memory_usage': round(max_memory, 2),
                'total_monitoring_points': len(self.system_metrics['cpu_usage'])
            },
            'detailed_metrics': self.system_metrics,
            'report_generated_at': datetime.now().isoformat()
        }
        
        # 保存JSON报告
        with open('production_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # 生成人类可读的报告
        self.generate_human_readable_report(report_data)
        
        logger.info("✅ 测试报告生成完成")
        logger.info("📄 报告文件:")
        logger.info("   - production_test_report.json (详细数据)")
        logger.info("   - PRODUCTION_TEST_REPORT.md (可读报告)")

    def generate_human_readable_report(self, report_data):
        """生成人类可读的报告"""
        with open('PRODUCTION_TEST_REPORT.md', 'w', encoding='utf-8') as f:
            f.write("# 🎬 CineCast《鱼没有脚》生产测试报告\n\n")
            
            f.write("## 📋 测试概述\n\n")
            summary = report_data['test_summary']
            f.write(f"- **测试时间**: {summary.get('test_start_time', 'N/A')} 至 {summary.get('test_end_time', 'N/A')}\n")
            f.write(f"- **测试对象**: 《鱼没有脚》EPUB文件\n")
            f.write(f"- **文件大小**: {summary.get('epub_size_mb', 0):.2f} MB\n")
            f.write(f"- **总耗时**: {summary.get('total_duration_seconds', 0):.2f} 秒 ({summary.get('total_duration_seconds', 0)/60:.2f} 分钟)\n")
            f.write(f"- **测试结果**: {'✅ 成功' if summary.get('success', False) else '❌ 失败'}\n\n")
            
            f.write("## ⏱️ 阶段时间分析\n\n")
            f.write(f"- **阶段一 (剧本生成)**: {summary.get('phase1_duration_seconds', 0):.2f} 秒\n")
            f.write(f"- **阶段二 (音频渲染)**: {summary.get('phase2_duration_seconds', 0):.2f} 秒\n\n")
            
            f.write("## 🖥️ 系统性能指标\n\n")
            perf = report_data['system_performance']
            f.write(f"- **平均CPU使用率**: {perf['average_cpu_usage']}%\n")
            f.write(f"- **峰值CPU使用率**: {perf['peak_cpu_usage']}%\n")
            f.write(f"- **平均内存使用率**: {perf['average_memory_usage']}%\n")
            f.write(f"- **峰值内存使用率**: {perf['peak_memory_usage']}%\n")
            f.write(f"- **监控采样点数**: {perf['total_monitoring_points']} 次\n\n")
            
            f.write("## 📁 输出信息\n\n")
            f.write(f"- **输出目录**: {summary.get('output_directory', 'N/A')}\n")
            f.write("- **生成文件**: 有声书成品文件\n\n")
            
            if not summary.get('success', True):
                f.write("## ❌ 错误信息\n\n")
                f.write(f"```\n{summary.get('error', '未知错误')}\n```\n\n")
            
            f.write("---\n")
            f.write("**报告生成时间**: " + report_data['report_generated_at'] + "\n")
            f.write("**测试环境**: CineCast v1.0\n")

def main():
    """主函数"""
    monitor = ProductionTestMonitor()
    
    try:
        # 运行生产测试
        success = monitor.run_production_test()
        
        # 生成报告
        monitor.generate_report()
        
        return success
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ 测试被用户中断")
        monitor.test_results['interrupted'] = True
        monitor.generate_report()
        return False
    except Exception as e:
        logger.error(f"❌ 测试过程中发生未预期错误: {e}")
        monitor.test_results['success'] = False
        monitor.test_results['error'] = str(e)
        monitor.generate_report()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)