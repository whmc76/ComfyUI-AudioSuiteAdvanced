"""
音频处理节点模块
Audio Processing Nodes Module
"""

import torch
import numpy as np
from typing import List, Dict, Any, Union, Tuple
import comfy.model_management
import torchaudio
from torchaudio.transforms import Fade, Resample
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
import librosa
import math
import torchaudio.functional as F

class AudioConcatenateFree:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio1": ("AUDIO",),
            "audio2": ("AUDIO",),
            "direction": (
                ['right', 'left'],
                {"default": 'right'}),
            "gap_duration": ("FLOAT", {
                "default": 0.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "description": "Gap duration in seconds between audio1 and audio2."
            }),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concanate"
    CATEGORY = "AudioSuiteAdvanced"
    DESCRIPTION = """
Concatenates audio1 and audio2 in the specified direction, with optional silence gap.
"""

    def concanate(self, audio1, audio2, direction, gap_duration):
        import torch
        # 容错处理：如果只有一个有效输入，直接返回该音频
        if audio1 is None and audio2 is None:
            raise Exception("Both audio1 and audio2 are None, cannot concatenate.")
        if audio1 is None:
            return (audio2,)
        if audio2 is None:
            return (audio1,)
        sample_rate_1 = audio1["sample_rate"]
        sample_rate_2 = audio2["sample_rate"]
        if sample_rate_1 != sample_rate_2:
            raise Exception("Sample rates of the two audios do not match.")
        waveform_1 = audio1["waveform"]
        waveform_2 = audio2["waveform"]
        # 生成静音段
        silence = None
        if gap_duration > 0:
            num_channels = waveform_1.shape[0]
            silence_len = int(sample_rate_1 * gap_duration)
            silence = torch.zeros((num_channels, 1, silence_len), dtype=waveform_1.dtype, device=waveform_1.device)
        # 拼接顺序
        if direction == 'right':
            parts = [waveform_1]
            if silence is not None:
                parts.append(silence)
            parts.append(waveform_2)
        else:  # left
            parts = [waveform_2]
            if silence is not None:
                parts.append(silence)
            parts.append(waveform_1)
        concatenated_audio = torch.cat(parts, dim=2)
        return ({"waveform": concatenated_audio, "sample_rate": sample_rate_1},)

def ensure_stereo(audio: torch.Tensor) -> torch.Tensor:
    """
    Ensures the input audio is stereo. Supports [channels, frames] and [batch, channels, frames].
    Handles mono by duplicating channels and multi-channel by downmixing to stereo.

    Args:
        audio (torch.Tensor): Audio data with shape [channels, frames] or [batch, channels, frames].

    Returns:
        torch.Tensor: Stereo audio with the same dimensional format as the input.
    """
    if audio.ndim not in (2, 3):
        raise ValueError(
            "Audio input must have 2 or 3 dimensions: [channels, frames] or [batch, channels, frames]."
        )

    is_batched = audio.ndim == 3
    channels_dim = 1 if is_batched else 0

    # Already stereo
    if audio.shape[channels_dim] == 2:
        return audio

    # Mono audio
    elif audio.shape[channels_dim] == 1:
        return audio.repeat(1, 2, 1) if is_batched else audio.repeat(2, 1)

    # Multi-channel audio
    audio = audio.narrow(channels_dim, 0, 2).mean(dim=channels_dim, keepdim=True)
    return audio.repeat(1, 2, 1) if is_batched else audio.repeat(2, 1)

class AudioSeparationNode:
    """
    音频分离节点，将音频分离为四个音轨：低音、鼓点、其他、人声。
    Audio Separation Node: Separate audio into four sources: bass, drums, other, and vocals.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "description": "输入音频，将被分离为四个音轨\nInput audio to be separated into four tracks"
                }),
            },
            "optional": {
                "chunk_fade_shape": (
                    [
                        "linear",
                        "half_sine",
                        "logarithmic",
                        "exponential",
                    ],
                    {
                        "default": "linear",
                        "description": "音频分块重叠区域的淡入淡出效果。线性为均匀淡入淡出，半正弦为平滑曲线，对数为先快后慢，指数为先慢后快。\nFade effect at chunk overlaps. Linear for even fading, Half-Sine for smooth curve, Logarithmic for quick fade out and slow fade in, or Exponential for slow fade out and quick fade in."
                    },
                ),
                "chunk_length": (
                    "FLOAT",
                    {
                        "default": 10.0,
                        "min": 1.0,
                        "max": 30.0,
                        "step": 0.5,
                        "description": "每个音频块的长度（秒）。较长的块可能需要更多内存，但可能产生更好的结果。\nLength of each audio chunk in seconds. Longer chunks may require more memory but might produce better results."
                    },
                ),
                "chunk_overlap": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "description": "音频块之间的重叠时间（秒）。如果块太短或音频变化很快，可能需要更高的重叠。\nOverlap between audio chunks in seconds. Higher overlap may be necessary if chunks are too short or audio changes rapidly."
                    },
                ),
            },
        }

    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO")
    RETURN_NAMES = ("bass", "drums", "other", "vocals")
    FUNCTION = "separate_audio"
    CATEGORY = "AudioSuiteAdvanced"
    DESCRIPTION = "将音频分离为四个音轨：低音、鼓点、其他、人声。\nSeparate audio into four sources: bass, drums, other, and vocals."

    def separate_audio(
        self,
        audio: Dict[str, Any],
        chunk_fade_shape: str = "linear",
        chunk_length: float = 10.0,
        chunk_overlap: float = 0.1,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """分离音频为四个音轨
        
        Args:
            audio: 输入音频字典，包含waveform和sample_rate
            chunk_fade_shape: 淡入淡出形状
            chunk_length: 块长度（秒）
            chunk_overlap: 块重叠时间（秒）
            
        Returns:
            四个分离的音轨：低音、鼓点、其他、人声
        """
        device: torch.device = comfy.model_management.get_torch_device()
        waveform: torch.Tensor = audio["waveform"]
        waveform = waveform.squeeze(0).to(device)
        input_sample_rate: int = audio["sample_rate"]

        bundle = HDEMUCS_HIGH_MUSDB_PLUS
        model: torch.nn.Module = bundle.get_model().to(device)
        model_sample_rate = bundle.sample_rate

        waveform = ensure_stereo(waveform)

        # 重采样到模型期望的采样率
        if input_sample_rate != model_sample_rate:
            resample = Resample(input_sample_rate, model_sample_rate).to(device)
            waveform = resample(waveform)

        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()  # 标准化

        sources = self._separate_sources(
            model,
            waveform[None],
            model_sample_rate,
            segment=chunk_length,
            overlap=chunk_overlap,
            device=device,
            chunk_fade_shape=chunk_fade_shape,
        )[0]
        sources = sources * ref.std() + ref.mean()
        sources_list = model.sources
        sources = list(sources)

        return self._sources_to_tuple(dict(zip(sources_list, sources)), model_sample_rate)

    def _sources_to_tuple(
        self, sources: Dict[str, torch.Tensor], sample_rate: int
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """将分离的音源转换为输出格式
        
        Args:
            sources: 分离的音源字典
            sample_rate: 采样率
            
        Returns:
            四个音轨的元组
        """
        output_order = ["bass", "drums", "other", "vocals"]
        outputs = []
        for source in output_order:
            if source not in sources:
                raise ValueError(f"Missing source {source} in the output")
            outputs.append(
                {
                    "waveform": sources[source].cpu().unsqueeze(0),
                    "sample_rate": sample_rate,
                }
            )
        return tuple(outputs)

    def _separate_sources(
        self,
        model: torch.nn.Module,
        mix: torch.Tensor,
        sample_rate: int,
        segment: float = 10.0,
        overlap: float = 0.1,
        device: torch.device = None,
        chunk_fade_shape: str = "linear",
    ) -> torch.Tensor:
        """
        将模型应用于给定的混合音频。使用淡入淡出，并将片段相加以逐段添加模型。
        
        Args:
            model: 分离模型
            mix: 混合音频张量
            sample_rate: 采样率
            segment: 片段长度（秒）
            overlap: 重叠时间（秒）
            device: 计算设备
            chunk_fade_shape: 淡入淡出形状
            
        Returns:
            分离后的音源张量
        """
        if device is None:
            device = mix.device
        else:
            device = torch.device(device)

        batch, channels, length = mix.shape

        chunk_len = int(sample_rate * segment * (1 + overlap))
        start = 0
        end = chunk_len
        overlap_frames = overlap * sample_rate
        fade = Fade(
            fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape=chunk_fade_shape
        )

        final = torch.zeros(batch, len(model.sources), channels, length, device=device)

        while start < length - overlap_frames:
            chunk = mix[:, :, start:end]
            with torch.no_grad():
                out = model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out
            if start == 0:
                fade.fade_in_len = int(overlap_frames)
                start += int(chunk_len - overlap_frames)
            else:
                start += chunk_len
            end += chunk_len
            if end >= length:
                fade.fade_out_len = 0
        return final 