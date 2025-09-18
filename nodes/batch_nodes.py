"""
批量处理节点模块
Batch Processing Nodes Module
"""

import os
import re
import torch
from typing import List, Dict, Any

class MakeAudioBatchNode:
    """
    音频批量队列节点，将两个输入音频顺序加入队列，自动处理空输入。
    Make Audio Batch Node: Add two input audios to a batch queue in order, handle empty input gracefully.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio1": ("AUDIO", {"description": "第一个音频输入\nFirst audio input"}),
                "audio2": ("AUDIO", {"description": "第二个音频输入\nSecond audio input"}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("audio_batch",)
    FUNCTION = "make_batch"
    CATEGORY = "AudioSuiteAdvanced"

    def make_batch(self, audio1, audio2):
        """
        将两个音频顺序加入队列，自动处理空输入。
        Add two audios to a batch queue in order, handle empty input.
        """
        batch = []
        if audio1 is not None:
            batch.append(audio1)
        if audio2 is not None:
            batch.append(audio2)
        return (batch,)

class CombineAudioFromList:
    """
    从音频列表/批量合并为一段音频，可选根据字幕时间戳对齐。
    Combine Audio From List/Batch: Merge audio batch/list into one audio, optionally align with subtitle timestamps.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_batch": ("LIST", {"description": "音频队列，AUDIO类型列表\nAudio batch/list, list of AUDIO objects"}),
                "gap_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "description": "片段间隔时间（秒）\nGap duration (seconds) between audio segments"
                }),
                "use_timestamps": ("BOOLEAN", {
                    "default": False,
                    "description": "是否根据字幕时间戳合并\nWhether to align with subtitle timestamps"
                }),
            },
            "optional": {
                "srt_file": ("STRING", {"default": "", "description": "可选字幕文件路径\nOptional subtitle file path"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("combined_audio",)
    FUNCTION = "combine_audio"
    CATEGORY = "AudioSuiteAdvanced"

    def combine_audio(self, audio_batch, gap_duration, use_timestamps, srt_file=None):
        """
        合并音频队列为一段音频，可选根据字幕时间戳对齐。
        Merge audio batch/list into one audio, optionally align with subtitle timestamps.
        """
        import torch
        import os
        import re
        # srt_file路径清洗
        if srt_file is not None and isinstance(srt_file, str):
            srt_file = srt_file.strip().strip('"').strip("'")
        # 拍平嵌套列表
        def flatten(lst):
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result
        flat_batch = flatten(audio_batch)
        if not flat_batch or not isinstance(flat_batch[0], dict) or "waveform" not in flat_batch[0] or "sample_rate" not in flat_batch[0]:
            raise Exception("音频队列格式错误，需为AUDIO对象列表。\nAudio batch format error, must be a list of AUDIO objects.")
        sample_rate = flat_batch[0]["sample_rate"]
        for audio in flat_batch:
            if audio["sample_rate"] != sample_rate:
                raise Exception("所有音频采样率必须一致\nAll audio sample rates must be the same.")
        waveforms = [audio["waveform"] for audio in flat_batch]
        # 如果不使用时间戳，普通拼接
        if not use_timestamps or not srt_file or not os.path.exists(srt_file):
            silence = None
            if gap_duration > 0:
                num_channels = waveforms[0].shape[0]
                silence_len = int(sample_rate * gap_duration)
                silence = torch.zeros((num_channels, 1, silence_len), dtype=waveforms[0].dtype, device=waveforms[0].device)
            merged = waveforms[0]
            for w in waveforms[1:]:
                if silence is not None:
                    merged = torch.cat((merged, silence, w), dim=2)
                else:
                    merged = torch.cat((merged, w), dim=2)
            return ({"waveform": merged, "sample_rate": sample_rate},)
        # 解析字幕时间戳
        ext = os.path.splitext(srt_file)[1].lower()
        with open(srt_file, 'r', encoding='utf-8') as f:
            content = f.read()
        # 支持srt, ass, ssa, vtt, lrc, sub
        timestamps = []
        if ext == '.srt':
            # SRT格式
            for match in re.finditer(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})', content):
                start = int(match.group(1))*3600 + int(match.group(2))*60 + int(match.group(3)) + int(match.group(4))/1000
                end = int(match.group(5))*3600 + int(match.group(6))*60 + int(match.group(7)) + int(match.group(8))/1000
                timestamps.append((start, end))
        elif ext in ['.ass', '.ssa']:
            # ASS/SSA格式
            for line in content.splitlines():
                if line.startswith('Dialogue:'):
                    parts = line.split(',', 3)
                    if len(parts) >= 3:
                        start = self._parse_ass_time(parts[1])
                        end = self._parse_ass_time(parts[2])
                        timestamps.append((start, end))
        elif ext == '.vtt':
            # VTT格式
            for match in re.finditer(r'(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})', content):
                start = int(match.group(1))*3600 + int(match.group(2))*60 + int(match.group(3)) + int(match.group(4))/1000
                end = int(match.group(5))*3600 + int(match.group(6))*60 + int(match.group(7)) + int(match.group(8))/1000
                timestamps.append((start, end))
        elif ext == '.lrc':
            # LRC格式（只用时间点，假设每段音频间隔均等）
            for match in re.finditer(r'\[(\d{1,2}):(\d{2})(?:\.(\d{1,2}))?\]', content):
                start = int(match.group(1))*60 + int(match.group(2)) + (int(match.group(3)) if match.group(3) else 0)/100
                timestamps.append((start, start+1))  # 假设每段1秒
        elif ext == '.sub':
            # SUB格式 {start}{end}
            for match in re.finditer(r'\{(\d+)\}\{(\d+)\}', content):
                start = int(match.group(1))/25  # 假设25fps
                end = int(match.group(2))/25
                timestamps.append((start, end))
        # 对齐音频到时间轴
        timeline = []
        current_time = 0
        for i, (start, end) in enumerate(timestamps):
            if i >= len(waveforms):
                break
            if start > current_time:
                gap_len = int((start - current_time) * sample_rate)
                if gap_len > 0:
                    num_channels = waveforms[0].shape[0]
                    silence = torch.zeros((num_channels, 1, gap_len), dtype=waveforms[0].dtype, device=waveforms[0].device)
                    timeline.append(silence)
            timeline.append(waveforms[i])
            current_time = end
        merged = torch.cat(timeline, dim=2) if timeline else waveforms[0]
        return ({"waveform": merged, "sample_rate": sample_rate},)

    def _parse_ass_time(self, t):
        # ASS/SSA时间格式: H:MM:SS.cs
        parts = t.strip().split(':')
        if len(parts) == 3:
            h, m, s = parts
            if '.' in s:
                s, cs = s.split('.')
                return int(h)*3600 + int(m)*60 + int(s) + int(cs)/100
            else:
                return int(h)*3600 + int(m)*60 + int(s)
        return 0 