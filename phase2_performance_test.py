#!/usr/bin/env python3
"""
CineCast 阶段二专项测试脚本 - 直接使用已生成的剧本进行音频渲染测试
基于已有的剧本文件，跳过阶段一，直接测试音频渲染性能和质量
"""

import os
import sys
import json
import time
import psutil
import logging
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

# 配置详细的日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase2_performance_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Phase2PerformanceTester:
    def __init__(self):
        self.metrics_log = []
        self.test_results = {}
        self.script_dir = "./output/Audiobooks/scripts"
        
    def collect_metrics(self, stage=""):
        """收集系统性能指标"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'stage': stage,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': round(memory.used / (1024**3), 2),
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_read_mb': round(disk_io.read_bytes / (1024**2), 2) if disk_io else 0,
                'disk_write_mb': round(disk_io.write_bytes / (1024**2), 2) if disk_io else 0
            }
            
            self.metrics_log.append(metrics)
            logger.info(f"[{stage}] CPU: {cpu_percent}% | 内存: {memory.percent}% ({memory.used/1024/1024:.0f}MB)")
            
            return metrics
        except Exception as e:
            logger.error(f"收集系统指标时出错: {e}")
            return {}

    def load_existing_scripts(self):
        """加载已生成的剧本文件"""
        logger.info("📂 加载已生成的剧本文件...")
        
        script_files = sorted([f for f in os.listdir(self.script_dir) if f.endswith('.json')])
        scripts_data = []
        
        for script_file in script_files:
            script_path = os.path.join(self.script_dir, script_file)
            try:
                with open(script_path, 'r', encoding='utf-8') as f:
                    script_content = json.load(f)
                    scripts_data.append({
                        'filename': script_file,
                        'content': script_content,
                        'unit_count': len(script_content)
                    })
                    logger.info(f"✅ 加载 {script_file}: {len(script_content)} 个单元")
            except Exception as e:
                logger.error(f"❌ 加载 {script_file} 失败: {e}")
        
        logger.info(f"📊 总共加载 {len(scripts_data)} 个剧本文件")
        return scripts_data

    def run_phase2_test(self):
        """运行阶段二性能测试"""
        logger.info("=" * 60)
        logger.info("🎬 开始阶段二音频渲染性能测试")
        logger.info("=" * 60)
        
        # 初始状态收集
        self.collect_metrics("测试开始")
        
        try:
            # 加载已有的剧本
            scripts_data = self.load_existing_scripts()
            if not scripts_data:
                raise Exception("没有找到可用的剧本文件")
            
            # 导入必要的模块
            from modules.mlx_tts_engine import MLXRenderEngine
            from modules.cinematic_packager import CinematicPackager
            from modules.asset_manager import AssetManager
            
            # 初始化组件
            logger.info("🔧 初始化音频渲染组件...")
            
            # 初始化资产管理系统
            assets = AssetManager("./assets")
            self.collect_metrics("资产管理系统初始化")
            
            # 初始化MLX渲染引擎
            model_path = "../qwentts/models/Qwen3-TTS-MLX-0.6B"
            engine = MLXRenderEngine(model_path)
            self.collect_metrics("MLX渲染引擎初始化")
            
            # 初始化混音打包器
            packager = CinematicPackager("./output/Audiobooks")
            self.collect_metrics("混音打包器初始化")
            
            logger.info("✅ 所有组件初始化完成")
            
            # 获取音频素材
            ambient_bgm = assets.get_ambient_sound("fountain")  # 使用新的fountain环境音
            chime_sound = assets.get_transition_chime()  # 使用新的哲理过渡音效
            
            logger.info(f"🎵 环境音: fountain ({len(ambient_bgm)}ms)")
            logger.info(f"🎵 过渡音: soft_chime ({len(chime_sound)}ms)")
            
            # 开始渲染测试
            test_start_time = time.time()
            total_units_processed = 0
            successful_units = 0
            
            logger.info("\n" + "="*50)
            logger.info("🎙️ 开始音频渲染测试")
            logger.info("="*50)
            
            self.collect_metrics("渲染开始")
            
            # 按章节顺序处理
            for script_data in scripts_data:
                chapter_name = script_data['filename']
                script_content = script_data['content']
                unit_count = script_data['unit_count']
                
                logger.info(f"\n챕터 {chapter_name} ({unit_count} 个单元)")
                logger.info("-" * 30)
                
                chapter_start_time = time.time()
                chapter_successful = 0
                
                # 处理每个单元
                for i, unit in enumerate(script_content, 1):
                    try:
                        unit_start_time = time.time()
                        
                        # 获取适当的音色配置
                        voice_cfg = assets.get_voice_for_role(
                            unit["type"], 
                            unit.get("speaker"), 
                            unit.get("gender", "male")
                        )
                        
                        # 渲染音频单元
                        unit_audio = engine.render_unit(unit["content"], voice_cfg)
                        
                        # 添加到打包器
                        packager.add_audio(unit_audio, ambient=ambient_bgm, chime=chime_sound)
                        
                        unit_duration = time.time() - unit_start_time
                        chapter_successful += 1
                        total_units_processed += 1
                        successful_units += 1
                        
                        logger.info(f"   ✓ 单元 {i}/{unit_count}: {unit['type']} - {unit.get('speaker', 'N/A')} ({len(unit_audio)}ms, {unit_duration:.2f}s)")
                        
                        # 每处理10个单元收集一次系统指标
                        if i % 10 == 0:
                            self.collect_metrics(f"{chapter_name}_progress_{i}")
                            
                    except Exception as e:
                        logger.error(f"   ✗ 单元 {i} 处理失败: {e}")
                        total_units_processed += 1
                
                chapter_duration = time.time() - chapter_start_time
                success_rate = (chapter_successful / unit_count) * 100 if unit_count > 0 else 0
                logger.info(f"챕터完成: 成功率 {success_rate:.1f}% ({chapter_successful}/{unit_count}), 耗时 {chapter_duration:.2f}s")
            
            # 完成打包
            logger.info("\n📦 完成音频打包...")
            packager.finalize(ambient=ambient_bgm, chime=chime_sound)
            self.collect_metrics("打包完成")
            
            # 总体统计
            total_duration = time.time() - test_start_time
            overall_success_rate = (successful_units / total_units_processed) * 100 if total_units_processed > 0 else 0
            
            self.test_results = {
                'test_start_time': datetime.fromtimestamp(test_start_time).isoformat(),
                'test_end_time': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'total_chapters': len(scripts_data),
                'total_units_processed': total_units_processed,
                'successful_units': successful_units,
                'overall_success_rate': overall_success_rate,
                'average_time_per_unit': total_duration / total_units_processed if total_units_processed > 0 else 0,
                'scripts_used': [s['filename'] for s in scripts_data]
            }
            
            logger.info("\n" + "="*60)
            logger.info("🎉 阶段二测试完成!")
            logger.info("="*60)
            logger.info(f"📊 总耗时: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
            logger.info(f"📊 处理章节: {len(scripts_data)} 个")
            logger.info(f"📊 处理单元: {total_units_processed} 个")
            logger.info(f"📊 成功单元: {successful_units} 个")
            logger.info(f"📊 成功率: {overall_success_rate:.1f}%")
            logger.info(f"📊 平均每单元耗时: {self.test_results['average_time_per_unit']:.2f} 秒")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 阶段二测试失败: {e}")
            self.test_results['success'] = False
            self.test_results['error'] = str(e)
            return False

    def generate_report(self):
        """生成测试报告"""
        logger.info("\n" + "="*60)
        logger.info("📊 生成测试报告...")
        logger.info("="*60)
        
        # 计算性能统计
        cpu_usage_list = [m['cpu_percent'] for m in self.metrics_log if 'cpu_percent' in m]
        memory_usage_list = [m['memory_percent'] for m in self.metrics_log if 'memory_percent' in m]
        
        avg_cpu = sum(cpu_usage_list) / len(cpu_usage_list) if cpu_usage_list else 0
        peak_cpu = max(cpu_usage_list) if cpu_usage_list else 0
        avg_memory = sum(memory_usage_list) / len(memory_usage_list) if memory_usage_list else 0
        peak_memory = max(memory_usage_list) if memory_usage_list else 0
        
        # 生成报告数据
        report_data = {
            'test_summary': self.test_results,
            'performance_metrics': {
                'average_cpu_usage': round(avg_cpu, 2),
                'peak_cpu_usage': round(peak_cpu, 2),
                'average_memory_usage': round(avg_memory, 2),
                'peak_memory_usage': round(peak_memory, 2),
                'total_metric_samples': len(self.metrics_log)
            },
            'system_metrics_log': self.metrics_log,
            'report_generated_at': datetime.now().isoformat()
        }
        
        # 保存JSON报告
        with open('phase2_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        # 生成人类可读报告
        self.generate_human_readable_report(report_data)
        
        logger.info("✅ 测试报告生成完成")

    def generate_human_readable_report(self, report_data):
        """生成人类可读的报告"""
        with open('PHASE2_PERFORMANCE_TEST_REPORT.md', 'w', encoding='utf-8') as f:
            f.write("# 🎵 CineCast 阶段二音频渲染性能测试报告\n\n")
            
            f.write("## 📋 测试概述\n\n")
            summary = report_data['test_summary']
            f.write(f"- **测试时间**: {summary.get('test_start_time', 'N/A')} 至 {summary.get('test_end_time', 'N/A')}\n")
            f.write(f"- **测试类型**: 阶段二音频渲染专项测试\n")
            f.write(f"- **总耗时**: {summary.get('total_duration_seconds', 0):.2f} 秒 ({summary.get('total_duration_seconds', 0)/60:.2f} 分钟)\n")
            f.write(f"- **处理章节**: {summary.get('total_chapters', 0)} 个\n")
            f.write(f"- **处理单元**: {summary.get('total_units_processed', 0)} 个\n")
            f.write(f"- **成功率**: {summary.get('overall_success_rate', 0):.1f}%\n")
            f.write(f"- **平均每单元耗时**: {summary.get('average_time_per_unit', 0):.2f} 秒\n\n")
            
            f.write("## 🖥️ 系统性能指标\n\n")
            perf = report_data['performance_metrics']
            f.write(f"- **平均CPU使用率**: {perf['average_cpu_usage']}%\n")
            f.write(f"- **峰值CPU使用率**: {perf['peak_cpu_usage']}%\n")
            f.write(f"- **平均内存使用率**: {perf['average_memory_usage']}%\n")
            f.write(f"- **峰值内存使用率**: {perf['peak_memory_usage']}%\n")
            f.write(f"- **性能采样点数**: {perf['total_metric_samples']} 次\n\n")
            
            f.write("## 🎧 音频配置\n\n")
            f.write("- **环境音**: fountain.mp3 (喷泉环境音效)\n")
            f.write("- **过渡音**: soft_chime.mp3 (哲理过渡音效)\n")
            f.write("- **音色配置**: 根据角色类型自动匹配\n\n")
            
            f.write("## 📁 测试材料\n\n")
            f.write("使用以下已生成的剧本文件:\n")
            for script_file in summary.get('scripts_used', []):
                f.write(f"- {script_file}\n")
            f.write("\n")
            
            if not summary.get('success', True):
                f.write("## ❌ 错误信息\n\n")
                f.write(f"```\n{summary.get('error', '未知错误')}\n```\n\n")
            
            f.write("---\n")
            f.write("**报告生成时间**: " + report_data['report_generated_at'] + "\n")
            f.write("**测试环境**: CineCast v1.0\n")

def main():
    """主函数"""
    tester = Phase2PerformanceTester()
    
    try:
        # 运行阶段二测试
        success = tester.run_phase2_test()
        
        # 生成报告
        tester.generate_report()
        
        return success
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ 测试被用户中断")
        tester.test_results['interrupted'] = True
        tester.generate_report()
        return False
    except Exception as e:
        logger.error(f"❌ 测试过程中发生未预期错误: {e}")
        tester.test_results['success'] = False
        tester.test_results['error'] = str(e)
        tester.generate_report()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)