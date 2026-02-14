#!/usr/bin/env python3
"""
CineCast 混音与发行打包器
实现30分钟时长控制、环境音混流、防惊跳处理、尾部回收
"""

import os
import logging
from pydub import AudioSegment
from typing import Optional

logger = logging.getLogger(__name__)

class CinematicPackager:
    def __init__(self, output_dir="output"):
        """
        初始化混音打包器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.target_duration_ms = 30 * 60 * 1000  # 30分钟打包
        self.min_tail_ms = 10 * 60 * 1000         # 10分钟尾部阈值
        
        self.buffer = AudioSegment.empty()
        self.file_index = 1
        
        logger.info(f"📦 混音打包器初始化完成，输出目录: {output_dir}")
    
    def mix_ambient(self, main_audio: AudioSegment, ambient: AudioSegment) -> AudioSegment:
        """
        混入沉浸式声场
        
        Args:
            main_audio: 主音频
            ambient: 环境音背景
            
        Returns:
            AudioSegment: 混合后的音频
        """
        if len(ambient) < 500:
            logger.debug("环境音过短，跳过混音")
            return main_audio  # 无有效环境音
        
        try:
            # 将环境音量降低25dB，避免喧宾夺主
            ambient = ambient - 25 
            
            # 循环环境音使其与主音频等长
            loop_count = len(main_audio) // len(ambient) + 1
            ambient_looped = ambient * loop_count
            ambient_looped = ambient_looped[:len(main_audio)]
            
            # 混合音频
            mixed_audio = main_audio.overlay(ambient_looped)
            logger.debug("✅ 环境音混音完成")
            return mixed_audio
            
        except Exception as e:
            logger.error(f"❌ 环境音混音失败: {e}")
            return main_audio
    
    def add_audio(self, audio: AudioSegment, ambient: Optional[AudioSegment] = None, 
                  chime: Optional[AudioSegment] = None):
        """
        向缓冲区添加音频，满30分钟则发版
        
        Args:
            audio: 要添加的音频片段
            ambient: 环境音背景（可选）
            chime: 过渡音效（可选）
        """
        # 如果有环境音，先进行混音
        if ambient:
            audio = self.mix_ambient(audio, ambient)
        
        self.buffer += audio
        
        # 检查是否达到目标时长
        if len(self.buffer) >= self.target_duration_ms:
            self.export_volume(chime)
    
    def export_volume(self, chime: Optional[AudioSegment] = None):
        """
        导出一卷（一个完整的MP3）
        
        Args:
            chime: 开头过渡音效（可选）
        """
        if len(self.buffer) == 0:
            logger.warning("缓冲区为空，跳过导出")
            return
        
        try:
            final_audio = self.buffer
            
            # 1. 睡眠唤醒防惊跳：添加Chime，并对主干开头做3秒淡入
            final_audio = final_audio.fade_in(3000)
            if chime and len(chime) > 500:
                final_audio = chime + final_audio
                
            # 2. 尾部淡出，防止突兀结束
            final_audio = final_audio.fade_out(2000)
            
            # 3. 导出文件
            file_name = f"Audiobook_Part_{self.file_index:03d}.mp3"
            save_path = os.path.join(self.output_dir, file_name)
            
            logger.info(f"📦 正在压制: {file_name} ({len(final_audio)/1000/60:.1f}分钟)")
            
            # 导出为MP3格式
            final_audio.export(
                save_path, 
                format="mp3", 
                bitrate="128k",
                parameters=["-q:a", "2"]  # VBR质量等级
            )
            
            # 重置缓冲区
            self.buffer = AudioSegment.empty()
            self.file_index += 1
            
            logger.info(f"✅ 成功导出: {file_name}")
            
        except Exception as e:
            logger.error(f"❌ 导出失败: {e}")
    
    def finalize(self, ambient: Optional[AudioSegment] = None, 
                 chime: Optional[AudioSegment] = None):
        """
        处理书籍结尾的碎片
        
        Args:
            ambient: 环境音背景（可选）
            chime: 过渡音效（可选）
        """
        remaining_ms = len(self.buffer)
        if remaining_ms == 0:
            logger.info("没有剩余音频需要处理")
            return
        
        logger.info(f"🔚 处理尾部音频: {remaining_ms/1000/60:.1f}分钟")
        
        if remaining_ms < self.min_tail_ms and self.file_index > 1:
            # 尾部不足10分钟，追加到上一个文件
            self._merge_with_previous(ambient)
        else:
            # 独立导出为新的一卷
            self.export_volume(chime)
    
    def _merge_with_previous(self, ambient: Optional[AudioSegment] = None):
        """
        将尾部音频合并到上一个文件
        
        Args:
            ambient: 环境音背景（可选）
        """
        try:
            prev_index = self.file_index - 1
            prev_file = os.path.join(self.output_dir, f"Audiobook_Part_{prev_index:03d}.mp3")
            
            if not os.path.exists(prev_file):
                logger.warning(f"前一个文件不存在: {prev_file}，独立导出尾部")
                self.export_volume()
                return
            
            logger.info(f"🔗 尾部合并: {len(self.buffer)/1000/60:.1f}分钟追加到 {prev_file}")
            
            # 加载前一个文件
            prev_audio = AudioSegment.from_file(prev_file, format="mp3")
            
            # 处理尾部音频（如有环境音则混入）
            tail_audio = self.buffer
            if ambient:
                tail_audio = self.mix_ambient(tail_audio, ambient)
            
            # 合并音频
            merged = prev_audio + tail_audio
            
            # 重新导出
            merged.export(prev_file, format="mp3", bitrate="128k")
            
            # 清空缓冲区
            self.buffer = AudioSegment.empty()
            
            logger.info("✅ 尾部合并完成")
            
        except Exception as e:
            logger.error(f"❌ 尾部合并失败: {e}")
            # 失败时仍然独立导出
            self.export_volume()
    
    def get_buffer_status(self) -> dict:
        """
        获取当前缓冲区状态
        
        Returns:
            dict: 包含缓冲区信息的字典
        """
        return {
            "buffer_length_ms": len(self.buffer),
            "buffer_length_min": len(self.buffer) / 1000 / 60,
            "current_file_index": self.file_index,
            "target_duration_min": self.target_duration_ms / 1000 / 60,
            "remaining_until_target": (self.target_duration_ms - len(self.buffer)) / 1000 / 60
        }

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    packager = CinematicPackager("./test_output")
    
    # 创建测试音频
    test_audio = AudioSegment.silent(duration=5000)  # 5秒静音
    
    # 测试添加音频
    print("测试添加音频...")
    packager.add_audio(test_audio)
    
    # 检查状态
    status = packager.get_buffer_status()
    print(f"缓冲区状态: {status}")
    
    # 测试最终化
    print("测试最终化...")
    packager.finalize()
    
    print("✅ 测试完成")