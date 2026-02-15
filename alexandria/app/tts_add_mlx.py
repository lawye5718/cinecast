    def _mlx_generate_voice(self, text, instruct_text, speaker, voice_config, output_path):
        """使用MLX Qwen3-TTS模型生成语音 - 基于CineCast中验证的实现"""
        if not MLX_AVAILABLE:
            print("Warning: MLX not available, falling back to other methods")
            return False

        try:
            import mlx.core as mx
            import numpy as np
            from scipy.io import wavfile
            
            # 获取语音配置
            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            # 加载参考音频（如果存在）
            ref_audio_path = voice_data.get("ref_audio")
            ref_text = voice_data.get("ref_text", "参考音频文本")

            # 文本预处理
            import re
            # 清理文本，移除可能引起问题的字符
            cleaned_text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】《》、]', ' ', text)
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

            if len(cleaned_text) < 2:
                print(f"Text too short after cleaning: '{text}'. Creating silence.")
                # 创建静音文件
                sample_rate = 22050
                duration = 0.5  # 0.5秒静音
                silence = np.zeros(int(sample_rate * duration), dtype=np.float32)
                wavfile.write(output_path, sample_rate, silence)
                return True

            # 确保模型已加载
            if not hasattr(self, '_mlx_model') or self._mlx_model is None:
                from mlx_audio.tts.utils import load_model
                model_path = voice_data.get("model_path", "./models/Qwen3-TTS-MLX-0.6B")
                self._mlx_model = load_model(model_path)
            
            # 使用MLX模型生成音频
            print(f"MLX TTS generating: {cleaned_text[:50]}...")
            
            # 根据CineCast实现，使用适当的参数调用
            result = self._mlx_model.generate(
                text=cleaned_text,
                language=self._language
            )
            
            # 处理结果
            if hasattr(result, '__iter__'):
                audio_arrays = list(result)
                if audio_arrays:
                    # 合并音频片段
                    final_audio = np.concatenate(audio_arrays) if len(audio_arrays) > 1 else audio_arrays[0]
                    
                    # 确保音频数据格式正确
                    if final_audio.dtype != np.float32:
                        final_audio = final_audio.astype(np.float32)
                    
                    # 保存为WAV文件
                    sample_rate = 22050  # Qwen-TTS标准采样率
                    wavfile.write(output_path, sample_rate, final_audio)
                    
                    print(f"✅ MLX TTS audio saved to: {output_path}, shape: {final_audio.shape}")
                    return True
                else:
                    print("❌ MLX TTS returned no audio data")
                    return False
            else:
                print("❌ Unexpected result format from MLX TTS")
                return False
                
        except Exception as e:
            print(f"❌ MLX TTS generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            # 清理MLX缓存
            if 'mx' in globals():
                try:
                    mx.metal.clear_cache()
                except:
                    pass


    def _should_use_mlx(self):
        """检查是否应该使用MLX模式"""
        return self._mode == "mlx" and MLX_AVAILABLE


    def generate_voice(self, text, instruct_text, speaker, voice_config, output_path):
        """Generate audio using the appropriate method based on voice type config and mode."""
        # 串行执行以避免内存冲突
        with self._serial_lock:
            # 如果是MLX模式，直接使用MLX方法
            if self._should_use_mlx():
                return self._mlx_generate_voice(text, instruct_text, speaker, voice_config, output_path)
            
            voice_data = voice_config.get(speaker)
            if not voice_data:
                print(f"Warning: No voice configuration for '{speaker}'. Skipping.")
                return False

            voice_type = voice_data.get("type", "custom")

            if voice_type == "clone":
                return self.generate_clone_voice(text, speaker, voice_config, output_path)
            elif voice_type in ("lora", "builtin_lora"):
                return self.generate_lora_voice(text, instruct_text, voice_data, output_path)
            elif voice_type == "design":
                return self.generate_design_voice(text, instruct_text, voice_data, output_path)
            else:
                return self.generate_custom_voice(text, instruct_text, speaker, voice_config, output_path)