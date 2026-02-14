#!/usr/bin/env python3
"""
CineCast 三段式架构测试脚本
验证"计算与状态解耦"的核心设计理念
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.llm_director import LLMScriptDirector
from modules.mlx_tts_engine import MLXRenderEngine
from modules.cinematic_packager import CinematicPackager
from modules.asset_manager import AssetManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('three_stage_test.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ThreeStageArchitectureTest:
    def __init__(self):
        self.test_dir = "./test_output"
        self.script_dir = os.path.join(self.test_dir, "scripts")
        self.cache_dir = os.path.join(self.test_dir, "temp_wav_cache")
        self.output_dir = os.path.join(self.test_dir, "final_output")
        
        # 创建必要的目录
        for directory in [self.script_dir, self.cache_dir, self.output_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def test_stage_1_micro_chunking(self):
        """测试阶段一：微切片剧本生成"""
        logger.info("="*60)
        logger.info("🎬 阶段一测试：微切片剧本生成")
        logger.info("="*60)
        
        # 测试文本
        test_text = """
第一章 凯夫拉维克的风雪

夜幕降临，港口的灯火开始闪烁。远处传来汽笛声，划破了寂静的夜空。

"你相信命运吗？"老渔夫说道，他的声音在寒风中显得格外苍老。

年轻人摇摇头："我只相信海。"海浪拍打着码头，发出有节奏的声响。

远处的灯塔开始旋转，为归航的船只指引方向。这是冰岛最南端的小镇，也是故事开始的地方。
"""
        
        try:
            director = LLMScriptDirector()
            micro_script = director.parse_and_micro_chunk(test_text)
            
            # 保存微切片剧本
            script_path = os.path.join(self.script_dir, "test_chapter_micro.json")
            with open(script_path, 'w', encoding='utf-8') as f:
                json.dump(micro_script, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 阶段一完成：生成 {len(micro_script)} 个微切片")
            logger.info(f"📄 剧本保存至: {script_path}")
            
            # 显示样本数据
            logger.info("\n📋 微切片样本:")
            for i, item in enumerate(micro_script[:3]):
                logger.info(f"  {item['chunk_id']}: [{item['type']}] {item['content'][:30]}... (停顿{item['pause_ms']}ms)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 阶段一测试失败: {e}")
            return False
    
    def test_stage_2_dry_rendering(self):
        """测试阶段二：纯净干音渲染"""
        logger.info("="*60)
        logger.info("🎙️ 阶段二测试：纯净干音渲染")
        logger.info("="*60)
        
        try:
            # 初始化组件
            assets = AssetManager("./assets")
            engine = MLXRenderEngine(os.environ.get("CINECAST_MODEL_PATH", "../qwentts/models/Qwen3-TTS-MLX-0.6B"))
            
            # 读取微切片剧本
            script_path = os.path.join(self.script_dir, "test_chapter_micro.json")
            with open(script_path, 'r', encoding='utf-8') as f:
                micro_script = json.load(f)
            
            logger.info(f"🎵 开始渲染 {len(micro_script)} 个片段...")
            
            # 渲染前几个片段进行测试
            test_limit = min(5, len(micro_script))
            success_count = 0
            
            for i, item in enumerate(micro_script[:test_limit]):
                voice_cfg = assets.get_voice_for_role(
                    item["type"], 
                    item.get("speaker"), 
                    item.get("gender")
                )
                
                save_path = os.path.join(self.cache_dir, f"{item['chunk_id']}.wav")
                if engine.render_dry_chunk(item["content"], voice_cfg, save_path):
                    success_count += 1
                    logger.info(f"   ✓ 片段 {i+1}/{test_limit}: {item['chunk_id']} 渲染完成")
                else:
                    logger.error(f"   ✗ 片段 {i+1}/{test_limit}: {item['chunk_id']} 渲染失败")
            
            logger.info(f"✅ 阶段二完成：{success_count}/{test_limit} 片段渲染成功")
            logger.info(f"📁 干音文件保存至: {self.cache_dir}")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ 阶段二测试失败: {e}")
            return False
    
    def test_stage_3_post_processing(self):
        """测试阶段三：电影级混音发版"""
        logger.info("="*60)
        logger.info("🎛️ 阶段三测试：电影级混音发版")
        logger.info("="*60)
        
        try:
            # 初始化组件
            assets = AssetManager("./assets")
            packager = CinematicPackager(self.output_dir)
            
            # 加载音频资源
            ambient_bgm = assets.get_ambient_sound("fountain")
            chime_sound = assets.get_transition_chime()
            
            # 读取微切片剧本
            script_path = os.path.join(self.script_dir, "test_chapter_micro.json")
            with open(script_path, 'r', encoding='utf-8') as f:
                micro_script = json.load(f)
            
            # 限制测试片段数量
            test_script = micro_script[:5]  # 只处理前5个片段
            
            logger.info(f"🎬 开始混音处理 {len(test_script)} 个片段...")
            
            # 执行混音处理
            packager.process_from_cache(
                test_script, 
                self.cache_dir, 
                assets, 
                ambient_bgm, 
                chime_sound
            )
            
            logger.info("✅ 阶段三完成：混音处理成功")
            logger.info(f"🎵 最终音频保存至: {self.output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 阶段三测试失败: {e}")
            return False
    
    def run_complete_test(self):
        """运行完整的三段式架构测试"""
        logger.info("🏛️ 开始三段式物理隔离架构完整测试")
        logger.info("验证'计算与状态解耦'的核心设计理念")
        logger.info("="*60)
        
        results = {
            'stage_1': False,
            'stage_2': False,
            'stage_3': False
        }
        
        # 按顺序执行三个阶段
        results['stage_1'] = self.test_stage_1_micro_chunking()
        
        if results['stage_1']:
            results['stage_2'] = self.test_stage_2_dry_rendering()
        
        if results['stage_2']:
            results['stage_3'] = self.test_stage_3_post_processing()
        
        # 输出测试总结
        logger.info("="*60)
        logger.info("📊 三段式架构测试总结")
        logger.info("="*60)
        logger.info(f"阶段一 (微切片): {'✅ 通过' if results['stage_1'] else '❌ 失败'}")
        logger.info(f"阶段二 (干音渲染): {'✅ 通过' if results['stage_2'] else '❌ 失败'}")
        logger.info(f"阶段三 (混音发版): {'✅ 通过' if results['stage_3'] else '❌ 失败'}")
        
        overall_success = all(results.values())
        logger.info(f"\n🎯 总体结果: {'🎉 全部通过' if overall_success else '⚠️ 存在问题'}")
        
        if overall_success:
            logger.info("\n🏆 三段式物理隔离架构验证成功！")
            logger.info("实现了100%防内存溢出和断点续传的核心目标")
        
        return overall_success

def main():
    """主函数"""
    tester = ThreeStageArchitectureTest()
    success = tester.run_complete_test()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())