"""
角色处理节点模块
Character Processing Nodes Module
"""

import json
import numpy as np
import torch
from typing import List, Dict, Any

class CharacterVocalExtractor_ASAdv:
    """
    固定角色名多音轨输出节点，对原声音进行剪裁，将非目标角色的片段静音，仅保留目标角色语音进行输出。
    输出端口为 narrator_audio、character1_audio、character2_audio、character3_audio、character4_audio、character5_audio。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "description": "输入音频文件\nInput audio file"
                }),
                "subtitle_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "description": "字幕JSON数据\nSubtitle JSON data"
                }),
                "offset_seconds": ("FLOAT", {"default": 0, "min": 0, "max": 99999999999999999.0}),
                "duration_seconds": ("FLOAT", {"default": 0, "min": 0, "max": 99999999999999999.0}),
                "fill_silence": ("BOOLEAN", {
                    "default": True,
                    "description": "无片段时是否输出静音\nFill silence if no segment for character"
                }),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = (
        "narrator_audio",
        "character1_audio",
        "character2_audio",
        "character3_audio",
        "character4_audio",
        "character5_audio",
        "extraction_info"
    )
    FUNCTION = "extract_fixed_characters"
    CATEGORY = "AudioSuiteAdvanced"
    DESCRIPTION = "对原声音进行剪裁，将非目标角色的片段静音，仅保留目标角色语音。输出端口固定为 narrator_audio、character1_audio、character2_audio、character3_audio、character4_audio、character5_audio。每个端口输出对应角色音频，无片段则输出静音。"

    def extract_fixed_characters(self, audio: Dict[str, Any], subtitle_json: str, offset_seconds: float, duration_seconds: float, fill_silence: bool = True):
        import json
        import numpy as np
        import torch
        from typing import List, Dict, Any

        print(f"[CharacterVocalExtractor_ASAdv] 开始执行角色语音剪裁...")
        print(f"[CharacterVocalExtractor_ASAdv] 输入参数: offset_seconds={offset_seconds}, duration_seconds={duration_seconds}")
        print(f"[CharacterVocalExtractor_ASAdv] 输入音频类型: {type(audio)}")
        print(f"[CharacterVocalExtractor_ASAdv] 输入音频键: {list(audio.keys()) if isinstance(audio, dict) else 'N/A'}")

        # 获取原始音频并应用裁剪逻辑（遵循UTK的duration处理方式）
        waveform = audio["waveform"]  # [B, C, N]
        sample_rate = int(audio["sample_rate"])
        
        if duration_seconds > 0:
            # 只有当duration大于0时才进行裁剪
            start = int(offset_seconds * sample_rate)
            end = int(start + duration_seconds * sample_rate)
            waveform = waveform[:, :, start:end]
            print(f"[CharacterVocalExtractor_ASAdv] 应用时间段裁剪: {offset_seconds}s - {offset_seconds + duration_seconds}s")
        elif offset_seconds > 0:
            # 如果duration为0但offset大于0，只应用offset裁剪
            start = int(offset_seconds * sample_rate)
            waveform = waveform[:, :, start:]
            print(f"[CharacterVocalExtractor_ASAdv] 应用offset裁剪: 从{offset_seconds}s开始到结尾")
        # 如果duration和offset都为0，则不进行任何裁剪
        else:
            print(f"[CharacterVocalExtractor_ASAdv] 未应用任何裁剪，使用完整音频")
        
        # 更新audio字典
        audio = {"waveform": waveform, "sample_rate": sample_rate}

        # 固定角色名列表
        role_keys = [
            ("Narrator", "narrator_audio"),
            ("Character1", "character1_audio"),
            ("Character2", "character2_audio"),
            ("Character3", "character3_audio"),
            ("Character4", "character4_audio"),
            ("Character5", "character5_audio"),
        ]
        
        # 解析字幕JSON
        try:
            print(f"[CharacterVocalExtractor_ASAdv] 开始解析字幕JSON...")
            subtitles = json.loads(subtitle_json)
            print(f"[CharacterVocalExtractor_ASAdv] 字幕JSON解析成功，共 {len(subtitles)} 条字幕")
        except Exception as e:
            print(f"[CharacterVocalExtractor_ASAdv] 字幕JSON解析失败: {str(e)}")
            raise Exception(f"字幕JSON格式错误: {str(e)}")
        
        # 获取音频数据
        print(f"[CharacterVocalExtractor_ASAdv] 开始处理音频数据...")
        print(f"[CharacterVocalExtractor_ASAdv] 原始音频waveform类型: {type(audio['waveform'])}")
        print(f"[CharacterVocalExtractor_ASAdv] 原始音频waveform形状: {audio['waveform'].shape}")
        print(f"[CharacterVocalExtractor_ASAdv] 原始音频sample_rate: {audio['sample_rate']}")
        
        waveform = audio["waveform"].cpu().numpy().squeeze()
        sample_rate = audio["sample_rate"]
        
        print(f"[CharacterVocalExtractor_ASAdv] squeeze后音频形状: {waveform.shape}")
        print(f"[CharacterVocalExtractor_ASAdv] squeeze后音频数据类型: {waveform.dtype}")
        
        # 修复：正确处理各种维度的音频输入
        if len(waveform.shape) == 3:
            print(f"[CharacterVocalExtractor_ASAdv] 检测到3D张量输入: {waveform.shape}")
            # 3D 张量 [batch, channels, samples] -> [samples]
            if waveform.shape[0] == 1:
                waveform = waveform[0]  # 移除 batch 维度
                print(f"[CharacterVocalExtractor_ASAdv] 移除batch维度后形状: {waveform.shape}")
            else:
                waveform = waveform[0]  # 取第一个 batch
                print(f"[CharacterVocalExtractor_ASAdv] 取第一个batch后形状: {waveform.shape}")
            if waveform.shape[0] == 1:
                waveform = waveform[0]  # 单声道
                print(f"[CharacterVocalExtractor_ASAdv] 单声道处理后形状: {waveform.shape}")
            else:
                waveform = waveform.mean(axis=0)  # 多声道取平均
                print(f"[CharacterVocalExtractor_ASAdv] 多声道取平均后形状: {waveform.shape}")
        elif len(waveform.shape) == 2:
            print(f"[CharacterVocalExtractor_ASAdv] 检测到2D张量输入: {waveform.shape}")
            # 2D 张量 [channels, samples] 或 [samples, channels]
            if waveform.shape[0] == 1:
                waveform = waveform[0]  # [1, samples] -> [samples]
                print(f"[CharacterVocalExtractor_ASAdv] [1, samples]处理后形状: {waveform.shape}")
            elif waveform.shape[1] == 1:
                waveform = waveform[:, 0]  # [samples, 1] -> [samples]
                print(f"[CharacterVocalExtractor_ASAdv] [samples, 1]处理后形状: {waveform.shape}")
            else:
                waveform = waveform.mean(axis=0)  # 多声道取平均
                print(f"[CharacterVocalExtractor_ASAdv] 多声道取平均后形状: {waveform.shape}")
        else:
            print(f"[CharacterVocalExtractor_ASAdv] 检测到1D张量输入: {waveform.shape}")
        # 1D 张量 [samples] 保持不变
        
        print(f"[CharacterVocalExtractor_ASAdv] 最终音频形状: {waveform.shape}")
        print(f"[CharacterVocalExtractor_ASAdv] 最终音频长度: {len(waveform)} 样本")
        print(f"[CharacterVocalExtractor_ASAdv] 最终音频时长: {len(waveform) / sample_rate:.2f} 秒")
        
        # 时间转换函数
        def time_to_seconds(time_str: str) -> float:
            parts = time_str.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            else:
                return float(time_str)
        
        # 统计信息
        extraction_info = []
        outputs = []
        
        print(f"[CharacterVocalExtractor_ASAdv] 开始处理 {len(role_keys)} 个角色...")
        
        for i, (role, port_name) in enumerate(role_keys):
            print(f"[CharacterVocalExtractor_ASAdv] 处理角色 {i+1}/{len(role_keys)}: {role}")
            
            # 收集该角色所有片段
            segments = []
            for sub in subtitles:
                if str(sub.get("id", "")).strip().lower() == role.lower():
                    start = time_to_seconds(sub["start"])
                    end = time_to_seconds(sub["end"])
                    segments.append((start, end))
            
            print(f"[CharacterVocalExtractor_ASAdv] 角色 '{role}' 找到 {len(segments)} 个语音片段")
            
            # 生成音频
            if segments:
                print(f"[CharacterVocalExtractor_ASAdv] 开始为角色 '{role}' 剪裁音频...")
                # 计算总时长
                total_duration = max(e for s, e in segments)
                total_samples = int(total_duration * sample_rate)
                print(f"[CharacterVocalExtractor_ASAdv] 角色 '{role}' 总时长: {total_duration:.2f}秒, 总样本数: {total_samples}")
                
                # 创建与原音频等长的数组，初始化为静音
                full_audio = np.zeros(len(waveform), dtype=np.float32)
                print(f"[CharacterVocalExtractor_ASAdv] 创建剪裁音频数组: 形状={full_audio.shape}, 数据类型={full_audio.dtype}")
                
                for j, (s, e) in enumerate(segments):
                    print(f"[CharacterVocalExtractor_ASAdv] 处理片段 {j+1}/{len(segments)}: {s:.2f}s - {e:.2f}s")
                    
                    start_frame = int(s * sample_rate)
                    end_frame = int(e * sample_rate)
                    start_frame = max(0, min(start_frame, len(full_audio)))
                    end_frame = max(start_frame, min(end_frame, len(full_audio)))
                    
                    orig_start = int(s * sample_rate)
                    orig_end = int(e * sample_rate)
                    orig_start = max(0, min(orig_start, len(waveform)))
                    orig_end = max(orig_start, min(orig_end, len(waveform)))
                    
                    print(f"[CharacterVocalExtractor_ASAdv] 片段 {j+1} 目标位置: {start_frame} - {end_frame}")
                    print(f"[CharacterVocalExtractor_ASAdv] 片段 {j+1} 原始位置: {orig_start} - {orig_end}")
                    
                    if start_frame < end_frame and orig_start < orig_end:
                        seg_len = min(end_frame - start_frame, orig_end - orig_start)
                        print(f"[CharacterVocalExtractor_ASAdv] 片段 {j+1} 实际长度: {seg_len} 样本")
                        
                        # 从原始音频复制目标角色的语音片段
                        segment_data = waveform[orig_start:orig_start+seg_len].astype(np.float32)
                        print(f"[CharacterVocalExtractor_ASAdv] 片段 {j+1} 剪裁数据形状: {segment_data.shape}, 数据类型: {segment_data.dtype}")
                        
                        full_audio[start_frame:start_frame+seg_len] = segment_data
                        print(f"[CharacterVocalExtractor_ASAdv] 片段 {j+1} 数据已复制到目标位置（保留语音，其他部分静音）")
                    else:
                        print(f"[CharacterVocalExtractor_ASAdv] 片段 {j+1} 跳过（无效范围）")
                
                print(f"[CharacterVocalExtractor_ASAdv] 角色 '{role}' 音频剪裁完成，开始转换为张量...")
                
                # 修复：确保输出格式为 [channels, samples]，符合 ComfyUI 标准格式
                audio_tensor = torch.tensor(full_audio, dtype=torch.float32).cpu().unsqueeze(0)
                # 保证输出为 [B, C, N]
                if audio_tensor.dim() == 2:
                    audio_tensor = audio_tensor.unsqueeze(0)
                outputs.append({
                    "waveform": audio_tensor,  # [1, samples] -> [1, 1, samples]
                    "sample_rate": sample_rate
                })
                
                total_segments_duration = sum(e-s for s, e in segments)
                extraction_info.append(f"{role}: {len(segments)}段, {total_segments_duration:.2f}秒")
                print(f"[CharacterVocalExtractor_ASAdv] 角色 '{role}' 剪裁完成: {len(segments)}段, {total_segments_duration:.2f}秒")
            else:
                print(f"[CharacterVocalExtractor_ASAdv] 角色 '{role}' 无语音片段，生成全静音...")
                # 无片段，输出全静音（与原音频等长）
                silence = np.zeros(len(waveform), dtype=np.float32)
                print(f"[CharacterVocalExtractor_ASAdv] 角色 '{role}' 静音数组: 形状={silence.shape}, 数据类型={silence.dtype}")
                
                # 修复：确保输出格式为 [channels, samples]，符合 ComfyUI 标准格式
                silence_tensor = torch.tensor(silence, dtype=torch.float32).cpu().unsqueeze(0)
                # 保证输出为 [B, C, N]
                if silence_tensor.dim() == 2:
                    silence_tensor = silence_tensor.unsqueeze(0)
                outputs.append({
                    "waveform": silence_tensor,  # [1, samples] -> [1, 1, samples]
                    "sample_rate": sample_rate
                })
                extraction_info.append(f"{role}: 无片段")
                print(f"[CharacterVocalExtractor_ASAdv] 角色 '{role}' 全静音生成完成")
        
        print(f"[CharacterVocalExtractor_ASAdv] 所有角色剪裁完成，准备返回结果...")
        print(f"[CharacterVocalExtractor_ASAdv] 输出数量: {len(outputs)}")
        print(f"[CharacterVocalExtractor_ASAdv] 剪裁信息: {' | '.join(extraction_info)}")
        print(f"[CharacterVocalExtractor_ASAdv] 执行完成！")
        
        return (*outputs, " | ".join(extraction_info))


class CharacterVocalExtractorMultiTrack:
    """
    基于 UTK Audio Crop Process 的多轨角色音频切分节点
    使用 UTK 的音频切分逻辑，支持多角色音频分离和批量处理
    """
    CATEGORY = "AudioSuiteAdvanced"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "subtitle_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "description": "字幕JSON数据，包含角色和时间信息\nSubtitle JSON data with character and timing info"
                }),
                "gain_db": ("FLOAT", {"default": 0, "min": -100, "max": 100}),
                "offset_seconds": ("FLOAT", {"default": 0, "min": 0, "max": 99999999999999999.0}),
                "duration_seconds": ("FLOAT", {"default": 0, "min": 0, "max": 99999999999999999.0}),
                "resample_to_hz": ("FLOAT", {"default": 0, "min": 0, "max": 99999999999999999.0}),
                "make_stereo": ("BOOLEAN", {"default": True}),
                "fill_silence": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = (
        "narrator_audio",
        "character1_audio", 
        "character2_audio",
        "character3_audio",
        "character4_audio",
        "character5_audio",
        "extraction_info"
    )
    FUNCTION = "execute"
    DESCRIPTION = "基于 UTK Audio Crop Process 的多轨角色音频切分。使用 UTK 的音频处理逻辑，支持增益调整、重采样、立体声转换等功能。"

    @classmethod
    def IS_CHANGED(cls, *args):
        return args

    def execute(
        self,
        audio,
        subtitle_json: str,
        gain_db: float,
        offset_seconds: float,
        duration_seconds: float,
        resample_to_hz: float,
        make_stereo: bool,
        fill_silence: bool,
    ):
        import json
        import numpy as np
        import torch
        from typing import List, Dict, Any

        print(f"[CharacterVocalExtractorMultiTrack] 开始执行多轨角色音频切分...")
        print(f"[CharacterVocalExtractorMultiTrack] 输入参数: gain_db={gain_db}, offset_seconds={offset_seconds}, duration_seconds={duration_seconds}, resample_to_hz={resample_to_hz}, make_stereo={make_stereo}")

        # 获取原始音频
        waveform = audio["waveform"]  # [B, C, N]
        sample_rate = int(audio["sample_rate"])
        
        # 应用音频裁剪逻辑（遵循UTK的duration处理方式）
        if duration_seconds > 0:
            # 只有当duration大于0时才进行裁剪
            start = int(offset_seconds * sample_rate)
            end = int(start + duration_seconds * sample_rate)
            waveform = waveform[:, :, start:end]
            print(f"[CharacterVocalExtractorMultiTrack] 应用时间段裁剪: {offset_seconds}s - {offset_seconds + duration_seconds}s")
        elif offset_seconds > 0:
            # 如果duration为0但offset大于0，只应用offset裁剪
            start = int(offset_seconds * sample_rate)
            waveform = waveform[:, :, start:]
            print(f"[CharacterVocalExtractorMultiTrack] 应用offset裁剪: 从{offset_seconds}s开始到结尾")
        # 如果duration和offset都为0，则不进行任何裁剪
        else:
            print(f"[CharacterVocalExtractorMultiTrack] 未应用任何裁剪，使用完整音频")
        
        # 更新audio字典
        audio = {"waveform": waveform, "sample_rate": sample_rate}

        # 固定角色名列表
        role_keys = [
            ("Narrator", "narrator_audio"),
            ("Character1", "character1_audio"),
            ("Character2", "character2_audio"), 
            ("Character3", "character3_audio"),
            ("Character4", "character4_audio"),
            ("Character5", "character5_audio"),
        ]
        
        # 解析字幕JSON
        try:
            print(f"[CharacterVocalExtractorMultiTrack] 开始解析字幕JSON...")
            subtitles = json.loads(subtitle_json)
            print(f"[CharacterVocalExtractorMultiTrack] 字幕JSON解析成功，共 {len(subtitles)} 条字幕")
        except Exception as e:
            print(f"[CharacterVocalExtractorMultiTrack] 字幕JSON解析失败: {str(e)}")
            raise Exception(f"字幕JSON格式错误: {str(e)}")
        
        # 获取音频数据 - 完全对齐 UTK 的方式
        print(f"[CharacterVocalExtractorMultiTrack] 开始处理音频数据...")
        waveform = audio["waveform"]  # [B, C, N] - 与 UTK 完全一致
        sample_rate = int(audio["sample_rate"])  # 与 UTK 完全一致
        
        print(f"[CharacterVocalExtractorMultiTrack] 原始音频waveform形状: {waveform.shape}")
        print(f"[CharacterVocalExtractorMultiTrack] 原始音频sample_rate: {sample_rate}")
        
        # 时间转换函数
        def time_to_seconds(time_str: str) -> float:
            parts = time_str.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
            elif len(parts) == 3:
                hours = int(parts[0])
                minutes = int(parts[1])
                seconds = float(parts[2])
                return hours * 3600 + minutes * 60 + seconds
            else:
                return float(time_str)
        
        # 基于 UTK Audio Crop Process 的音频处理函数 - 完全对齐 UTK 逻辑
        def process_audio_segment(segment_waveform, segment_sample_rate, gain_db, resample_to_hz, make_stereo):
            """基于 UTK Audio Crop Process 的音频处理逻辑 - 完全对齐"""
            processed_waveform = segment_waveform.clone()
            processed_sample_rate = segment_sample_rate
            
            # 重采样 - 完全对齐 UTK 逻辑
            if resample_to_hz > 0 and int(resample_to_hz) != processed_sample_rate:
                try:
                    import torchaudio
                    processed_waveform = torchaudio.functional.resample(
                        processed_waveform, 
                        processed_sample_rate, 
                        int(resample_to_hz)
                    )
                    processed_sample_rate = int(resample_to_hz)
                    print(f"[CharacterVocalExtractorMultiTrack] 重采样完成: {segment_sample_rate}Hz -> {processed_sample_rate}Hz")
                except ImportError:
                    print(f"[CharacterVocalExtractorMultiTrack] 警告: torchaudio 未安装，跳过重采样")
            
            # 增益调整 - 完全对齐 UTK 逻辑
            if gain_db != 0.0:
                gain_scalar = 10 ** (gain_db / 20)
                processed_waveform = processed_waveform * gain_scalar
                print(f"[CharacterVocalExtractorMultiTrack] 增益调整完成: {gain_db}dB")
            
            # 强制立体声 - 完全对齐 UTK 逻辑
            if make_stereo and processed_waveform.shape[1] == 1:
                processed_waveform = torch.cat([processed_waveform, processed_waveform], dim=1)
                print(f"[CharacterVocalExtractorMultiTrack] 转换为立体声完成")
            elif make_stereo and processed_waveform.shape[1] != 2:
                raise ValueError(f"输入音频有 {processed_waveform.shape[1]} 个声道，无法转换为立体声 (2声道)")
            
            return processed_waveform, processed_sample_rate
        
        # 统计信息
        extraction_info = []
        outputs = []
        
        print(f"[CharacterVocalExtractorMultiTrack] 开始处理 {len(role_keys)} 个角色...")
        
        for i, (role, port_name) in enumerate(role_keys):
            print(f"[CharacterVocalExtractorMultiTrack] 处理角色 {i+1}/{len(role_keys)}: {role}")
            
            # 收集该角色所有片段
            segments = []
            for sub in subtitles:
                if str(sub.get("id", "")).strip().lower() == role.lower():
                    start = time_to_seconds(sub["start"])
                    end = time_to_seconds(sub["end"])
                    segments.append((start, end))
            
            print(f"[CharacterVocalExtractorMultiTrack] 角色 '{role}' 找到 {len(segments)} 个语音片段")
            
            # 生成音频
            if segments:
                print(f"[CharacterVocalExtractorMultiTrack] 开始为角色 '{role}' 切分音频...")
                
                # 计算总时长
                total_duration = max(e for s, e in segments)
                total_samples = int(total_duration * sample_rate)
                print(f"[CharacterVocalExtractorMultiTrack] 角色 '{role}' 总时长: {total_duration:.2f}秒, 总样本数: {total_samples}")
                
                # 创建与原音频等长的张量，初始化为静音 - 与 UTK 格式一致
                full_audio = torch.zeros_like(waveform)
                print(f"[CharacterVocalExtractorMultiTrack] 创建切分音频张量: 形状={full_audio.shape}")
                
                for j, (s, e) in enumerate(segments):
                    print(f"[CharacterVocalExtractorMultiTrack] 处理片段 {j+1}/{len(segments)}: {s:.2f}s - {e:.2f}s")
                    
                    # 基于 UTK Audio Crop Process 的切分逻辑 - 完全对齐
                    start = int(s * sample_rate)
                    end = int(e * sample_rate) if e > 0 else waveform.shape[2]
                    
                    # 边界检查 - 与 UTK 一致
                    start = max(0, min(start, waveform.shape[2]))
                    end = max(start, min(end, waveform.shape[2]))
                    
                    if start < end:
                        # 切分音频片段 - 与 UTK 完全一致
                        segment_waveform = waveform[:, :, start:end]
                        print(f"[CharacterVocalExtractorMultiTrack] 片段 {j+1} 切分完成: 形状={segment_waveform.shape}")
                        
                        # 应用 UTK 音频处理
                        processed_waveform, processed_sample_rate = process_audio_segment(
                            segment_waveform, sample_rate, gain_db, resample_to_hz, make_stereo
                        )
                        
                        # 将处理后的片段放回原位置
                        # 注意：如果重采样了，需要调整目标位置
                        if processed_sample_rate != sample_rate:
                            # 重新计算目标位置
                            target_start = int(s * processed_sample_rate)
                            target_end = int(e * processed_sample_rate)
                            target_start = max(0, min(target_start, full_audio.shape[2]))
                            target_end = max(target_start, min(target_end, full_audio.shape[2]))
                            
                            # 调整 full_audio 大小以匹配新的采样率
                            if full_audio.shape[2] != processed_waveform.shape[2]:
                                full_audio = torch.zeros(
                                    processed_waveform.shape[0], 
                                    processed_waveform.shape[1], 
                                    max(full_audio.shape[2], target_end),
                                    dtype=processed_waveform.dtype,
                                    device=processed_waveform.device
                                )
                        else:
                            target_start = start
                            target_end = end
                        
                        # 确保目标位置有效
                        target_start = max(0, min(target_start, full_audio.shape[2]))
                        target_end = max(target_start, min(target_end, full_audio.shape[2]))
                        seg_len = min(target_end - target_start, processed_waveform.shape[2])
                        
                        if seg_len > 0:
                            full_audio[:, :, target_start:target_start+seg_len] = processed_waveform[:, :, :seg_len]
                            print(f"[CharacterVocalExtractorMultiTrack] 片段 {j+1} 数据已复制到目标位置")
                        else:
                            print(f"[CharacterVocalExtractorMultiTrack] 片段 {j+1} 跳过（无效范围）")
                    else:
                        print(f"[CharacterVocalExtractorMultiTrack] 片段 {j+1} 跳过（无效范围）")
                
                print(f"[CharacterVocalExtractorMultiTrack] 角色 '{role}' 音频切分完成")
                
                # 确保输出格式正确 - 与 UTK 一致
                if make_stereo and full_audio.shape[1] == 1:
                    full_audio = torch.cat([full_audio, full_audio], dim=1)
                # 保证输出为 [B, C, N]
                if full_audio.dim() == 2:
                    full_audio = full_audio.unsqueeze(0)
                outputs.append({
                    "waveform": full_audio,
                    "sample_rate": processed_sample_rate if 'processed_sample_rate' in locals() else sample_rate
                })
                
                total_segments_duration = sum(e-s for s, e in segments)
                extraction_info.append(f"{role}: {len(segments)}段, {total_segments_duration:.2f}秒")
                print(f"[CharacterVocalExtractorMultiTrack] 角色 '{role}' 切分完成: {len(segments)}段, {total_segments_duration:.2f}秒")
            else:
                print(f"[CharacterVocalExtractorMultiTrack] 角色 '{role}' 无语音片段，生成全静音...")
                # 无片段，输出全静音（与原音频等长）
                silence = torch.zeros_like(waveform)
                
                # 应用立体声转换 - 与 UTK 一致
                if make_stereo and silence.shape[1] == 1:
                    silence = torch.cat([silence, silence], dim=1)
                # 保证输出为 [B, C, N]
                if silence.dim() == 2:
                    silence = silence.unsqueeze(0)
                outputs.append({
                    "waveform": silence,
                    "sample_rate": sample_rate
                })
                extraction_info.append(f"{role}: 无片段")
                print(f"[CharacterVocalExtractorMultiTrack] 角色 '{role}' 全静音生成完成")
        
        print(f"[CharacterVocalExtractorMultiTrack] 所有角色切分完成，准备返回结果...")
        print(f"[CharacterVocalExtractorMultiTrack] 输出数量: {len(outputs)}")
        print(f"[CharacterVocalExtractorMultiTrack] 切分信息: {' | '.join(extraction_info)}")
        print(f"[CharacterVocalExtractorMultiTrack] 执行完成！")
        
        return (*outputs, " | ".join(extraction_info)) 