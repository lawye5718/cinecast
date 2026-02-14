#!/usr/bin/env python3
"""
自动化测试状态更新脚本
定期更新详细的测试监控报告
"""

import os
import json
import time
import psutil
from datetime import datetime

def update_test_report():
    """更新测试报告"""
    report_path = "./DETAILED_TEST_MONITORING_REPORT.md"
    
    # 读取现有报告内容
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = "# 📊 《鱼没有脚》生产测试详细监控报告\n\n"
    
    # 获取当前状态
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 系统状态
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # 文件统计
    output_base = "./output/yu_meiyou_jiao_production"
    scripts_count = len([f for f in os.listdir(os.path.join(output_base, "scripts")) if f.endswith('.json')]) if os.path.exists(os.path.join(output_base, "scripts")) else 0
    wav_count = len([f for f in os.listdir(os.path.join(output_base, "temp_wav_cache")) if f.endswith('.wav')]) if os.path.exists(os.path.join(output_base, "temp_wav_cache")) else 0
    final_count = len([f for f in os.listdir(os.path.join(output_base, "final_audiobooks")) if f.endswith('.mp3')]) if os.path.exists(os.path.join(output_base, "final_audiobooks")) else 0
    
    # 更新报告内容
    update_section = f"""
## 📈 实时状态更新 ({timestamp})

### 系统性能
- **CPU使用率**: {cpu_percent:.1f}%
- **内存使用率**: {memory.percent:.1f}%
- **磁盘使用率**: {(disk.used/disk.total)*100:.2f}%
- **可用内存**: {memory.available/(1024**3):.2f}GB

### 文件产出进度
- **已生成剧本**: {scripts_count} 个
- **WAV缓存文件**: {wav_count} 个  
- **最终成品**: {final_count} 个

### 当前状态分析
{'✅ 系统运行稳定' if cpu_percent < 80 and memory.percent < 80 else '⚠️ 资源使用较高'}
{'✅ 无内存泄漏风险' if memory.percent < 85 else '⚠️ 内存使用接近上限'}
{'✅ 磁盘空间充足' if (disk.used/disk.total) < 0.8 else '⚠️ 磁盘空间紧张'}

---
"""
    
    # 将更新内容插入到报告中合适位置
    if "## 📈 实时状态更新" in content:
        # 替换现有的实时状态部分
        lines = content.split('\n')
        new_lines = []
        in_realtime_section = False
        
        for line in lines:
            if line.startswith("## 📈 实时状态更新"):
                in_realtime_section = True
                new_lines.append(update_section.strip())
                continue
            elif in_realtime_section and line.startswith("## "):
                in_realtime_section = False
                new_lines.append(line)
            elif not in_realtime_section:
                new_lines.append(line)
        
        content = '\n'.join(new_lines)
    else:
        # 添加新的实时状态部分
        content += update_section
    
    # 保存更新后的报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"[{timestamp}] 测试报告已更新")
    print(f"  系统状态: CPU {cpu_percent:.1f}%, 内存 {memory.percent:.1f}%")
    print(f"  文件进度: 剧本{scripts_count}个, WAV{wav_count}个, 成品{final_count}个")

def main():
    """主函数"""
    print("🚀 启动自动化测试状态更新服务...")
    print("🕒 更新间隔: 30分钟")
    print("=" * 50)
    
    while True:
        try:
            update_test_report()
            print(f"💤 等待30分钟后下次更新...")
            time.sleep(30 * 60)  # 30分钟
        except KeyboardInterrupt:
            print("\n🛑 状态更新服务被用户中断")
            break
        except Exception as e:
            print(f"❌ 状态更新出现错误: {e}")
            time.sleep(60)  # 出错后等待1分钟再试

if __name__ == "__main__":
    main()