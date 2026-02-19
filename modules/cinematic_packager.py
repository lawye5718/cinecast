#!/usr/bin/env python3
"""
CineCast 混音与发行打包器
阶段三：电影级混音发版 (Cinematic Post-Processing)
流水线第三阶段：从干音缓存组装成电影级有声书
"""

import os
import logging
import zipfile
from pydub import AudioSegment
from typing import Optional, List, Dict
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Dynamic pause constants (milliseconds)
CROSS_SPEAKER_PAUSE_MS = 500   # 不同角色之间的停顿
SAME_SPEAKER_PAUSE_MS = 250    # 同一角色连续说话的停顿


class CinematicPackager:
    FADE_IN_MS = 3000   # 淡入时长（毫秒）
    FADE_OUT_MS = 2000  # 淡出时长（毫秒）

    def __init__(self, output_dir="output", target_duration_min=30):
        """
        初始化电影级混音台
        
        Args:
            output_dir: 输出目录
            target_duration_min: 目标分卷时长（分钟），默认30分钟
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.target_duration_ms = target_duration_min * 60 * 1000
        self.min_tail_ms = 10 * 60 * 1000         # 10分钟尾部阈值
        self.sample_rate = 24000                  # Qwen3-TTS 1.7B 高保真采样率
        self.crossfade_ms = 18                    # 交叉淡化补偿 (15-20ms 范围，18ms 为 1.7B 情感波动最佳平衡点)
        
        self.buffer = AudioSegment.empty()
        self.file_index = 1
        
        # Track per-speaker audio for multi-track export
        self._speaker_tracks: dict = {}
        self._labels: list = []  # [{"start_ms", "end_ms", "speaker", "text"}]
        self._timeline_ms = 0  # current position on the global timeline
        
        logger.info(f"🎛️ 启动后期混音台 (Pydub)，输出目录: {output_dir}")
    
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
    
    def process_from_cache(self, micro_script: List[Dict], cache_dir: str, assets, 
                          ambient_bgm=None, chime=None):
        """
        流水线第三阶段：从干音缓存组装成电影级有声书
        
        Uses dynamic pauses: CROSS_SPEAKER_PAUSE_MS between different speakers,
        SAME_SPEAKER_PAUSE_MS for consecutive lines by the same speaker.
        """
        # 🌟 前置全量跳过：如果当前分卷已存在，直接跳过整个剧本的混音计算
        output_filename = f"Audiobook_Part_{self.file_index:03d}.mp3"
        output_path = os.path.join(self.output_dir, output_filename)
        if os.path.exists(output_path):
            logger.info(f"⏭️  检测到分卷已完全覆盖当前剧本，直接跳过混音计算: {output_filename}")
            # 快进 file_index 跳过所有已存在的分卷
            while os.path.exists(os.path.join(self.output_dir, f"Audiobook_Part_{self.file_index:03d}.mp3")):
                self.file_index += 1
            return

        logger.info("🎛️ 启动后期混音台 (Pydub)...")
        
        prev_speaker = None
        
        for item in tqdm(micro_script, desc="混音组装中"):
            wav_path = os.path.join(cache_dir, f"{item['chunk_id']}.wav")
            if not os.path.exists(wav_path):
                logger.warning(f"⚠️ 找不到干音缓存: {wav_path}，跳过该句。")
                continue
                
            # 加载干音
            segment = AudioSegment.from_file(wav_path, format="wav")
            
            # 应用语速与音调变化 (如标题的 0.8 倍速一字一顿)
            voice_cfg = assets.get_voice_for_role(
                item["type"], 
                item.get("speaker"), 
                item.get("gender")
            )
            speed_factor = voice_cfg.get("speed", 1.0)
            
            # 🌟 注意：调速应在 TTS 生成时控制，不在混音阶段通过修改帧率实现
            # 直接修改 frame_rate 会导致音调失真（变调变声），因此此处跳过速度调整
            
            # 🌟 动态停顿：同角色连续对白用短停顿，跨角色切换用长停顿
            current_speaker = item.get("speaker", "narrator")
            script_pause = item.get("pause_ms", 0)
            if prev_speaker is not None and current_speaker == prev_speaker:
                # Same speaker: use the shorter of script pause and cap
                pause_ms = SAME_SPEAKER_PAUSE_MS
            else:
                # Different speaker: ensure at least CROSS_SPEAKER_PAUSE_MS
                pause_ms = max(script_pause, CROSS_SPEAKER_PAUSE_MS)
            prev_speaker = current_speaker
            
            # Record label for multi-track export
            seg_start = self._timeline_ms
            seg_end = seg_start + len(segment)
            self._labels.append({
                "start_ms": seg_start,
                "end_ms": seg_end,
                "speaker": current_speaker,
                "text": item.get("content", "")[:80],
            })
            
            # Accumulate per-speaker track data
            if current_speaker not in self._speaker_tracks:
                # Pad with silence up to this point
                self._speaker_tracks[current_speaker] = AudioSegment.silent(
                    duration=seg_start
                )
            else:
                # Pad any gap since the last segment from this speaker
                current_len = len(self._speaker_tracks[current_speaker])
                if current_len < seg_start:
                    self._speaker_tracks[current_speaker] += AudioSegment.silent(
                        duration=seg_start - current_len
                    )
            self._speaker_tracks[current_speaker] += segment
            
            # 拼接入缓冲区
            self.buffer += segment + AudioSegment.silent(duration=pause_ms)
            self._timeline_ms += len(segment) + pause_ms
            
            # 满 30 分钟则导出
            if len(self.buffer) >= self.target_duration_ms:
                self.export_volume(ambient=ambient_bgm, chime=chime)
                
        # 结尾兜底
        self.finalize(ambient=ambient_bgm, chime=chime)
    
    def add_audio(self, audio: AudioSegment, ambient: Optional[AudioSegment] = None, 
                  chime: Optional[AudioSegment] = None):
        """
        向缓冲区添加音频，满30分钟则发版
        保留向后兼容性
        """
        # 如果有环境音，先进行混音
        if ambient:
            audio = self.mix_ambient(audio, ambient)
        
        self.buffer += audio
        
        # 检查是否达到目标时长
        if len(self.buffer) >= self.target_duration_ms:
            self.export_volume(chime=chime)
    
    def export_volume(self, ambient: Optional[AudioSegment] = None,
                     chime: Optional[AudioSegment] = None):
        """
        导出一卷（一个完整的MP3）

        当通过 process_from_cache 调用时，ambient 在此处混入整卷音频。
        当通过 add_audio 调用时，ambient 已在 add_audio 中按片段混入，
        此处不应再传入 ambient 以避免重复混音。
        
        Args:
            ambient: 环境音背景（可选，仅在 process_from_cache 流程中使用）
            chime: 开头过渡音效（可选）
        """
        if len(self.buffer) == 0:
            logger.warning("缓冲区为空，跳过导出")
            return
        
        # 🌟 断点续传：如果分卷文件已存在，跳过压制
        file_name = f"Audiobook_Part_{self.file_index:03d}.mp3"
        save_path = os.path.join(self.output_dir, file_name)
        if os.path.exists(save_path):
            logger.info(f"⏭️  检测到分卷已存在，跳过压制: {file_name}")
            self.buffer = AudioSegment.empty()
            self.file_index += 1
            return
        
        try:
            final_audio = self.buffer
            
            # 0. 混入环境音（如果有）
            if ambient:
                final_audio = self.mix_ambient(final_audio, ambient)
            
            # 1. 睡眠唤醒防惊跳：添加Chime，并对主干开头做淡入
            fade_in_ms = min(self.FADE_IN_MS, len(final_audio))
            final_audio = final_audio.fade_in(fade_in_ms)
            if chime and len(chime) > 500:
                final_audio = chime + final_audio
                
            # 2. 尾部淡出，防止突兀结束
            fade_out_ms = min(self.FADE_OUT_MS, len(final_audio))
            final_audio = final_audio.fade_out(fade_out_ms)
            
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
            self._merge_with_previous(ambient, chime)
        else:
            # 独立导出为新的一卷
            self.export_volume(ambient=ambient, chime=chime)
    
    def _merge_with_previous(self, ambient: Optional[AudioSegment] = None,
                             chime: Optional[AudioSegment] = None):
        """
        将尾部音频合并到上一个文件
        
        Args:
            ambient: 环境音背景（可选）
            chime: 过渡音效（可选）
        """
        try:
            prev_index = self.file_index - 1
            prev_file = os.path.join(self.output_dir, f"Audiobook_Part_{prev_index:03d}.mp3")
            
            if not os.path.exists(prev_file):
                logger.warning(f"前一个文件不存在: {prev_file}，独立导出尾部")
                self.export_volume(chime=chime)
                return
            
            logger.info(f"🔗 尾部合并: {len(self.buffer)/1000/60:.1f}分钟追加到 {prev_file}")
            
            # 加载前一个文件
            prev_audio = AudioSegment.from_file(prev_file, format="mp3")
            
            # 处理尾部音频（如有环境音则混入）
            tail_audio = self.buffer
            if ambient:
                tail_audio = self.mix_ambient(tail_audio, ambient)
            
            # 使用交叉淡化合并，避免前卷 fade_out 与尾部音频之间产生音量断层
            crossfade_ms = min(2000, len(prev_audio), len(tail_audio))
            merged = prev_audio.append(tail_audio, crossfade=crossfade_ms)
            
            # 重新导出
            merged.export(prev_file, format="mp3", bitrate="128k")
            
            # 清空缓冲区
            self.buffer = AudioSegment.empty()
            
            logger.info("✅ 尾部合并完成")
            
        except Exception as e:
            logger.error(f"❌ 尾部合并失败: {e}")
            # 失败时仍然独立导出
            self.export_volume(chime=chime)
    
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

    def export_audacity(self, output_path: Optional[str] = None) -> Optional[str]:
        """Export a multi-track Audacity project as a ZIP archive.

        The archive contains:
        - One WAV file per speaker (stem), named ``{speaker}.wav``
        - A ``labels.txt`` with tab-separated Audacity label format:
          ``start_seconds\tend_seconds\tspeaker: text``

        This allows professional producers to import into a DAW (Audacity,
        Logic Pro, etc.) for fine-grained per-line editing.

        Args:
            output_path: Path for the ZIP file.  Defaults to
                ``<output_dir>/audacity_export.zip``.

        Returns:
            The path to the created ZIP file, or ``None`` on failure.
        """
        if not self._speaker_tracks and not self._labels:
            logger.warning("No multi-track data collected; call process_from_cache first.")
            return None

        if output_path is None:
            output_path = os.path.join(self.output_dir, "audacity_export.zip")

        try:
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Write per-speaker stem WAVs
                for speaker, track in self._speaker_tracks.items():
                    safe_name = speaker.replace("/", "_").replace("\\", "_")
                    wav_name = f"{safe_name}.wav"
                    tmp_wav = os.path.join(self.output_dir, f"_tmp_{wav_name}")
                    try:
                        track.export(tmp_wav, format="wav")
                        zf.write(tmp_wav, wav_name)
                    finally:
                        if os.path.exists(tmp_wav):
                            os.unlink(tmp_wav)

                # Write Audacity labels
                label_lines = []
                for lbl in self._labels:
                    start_s = lbl["start_ms"] / 1000.0
                    end_s = lbl["end_ms"] / 1000.0
                    text = f"{lbl['speaker']}: {lbl['text']}"
                    label_lines.append(f"{start_s:.3f}\t{end_s:.3f}\t{text}")
                zf.writestr("labels.txt", "\n".join(label_lines))

            logger.info(f"✅ Audacity 多轨工程已导出: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"❌ Audacity 导出失败: {e}")
            return None

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