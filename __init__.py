"""
ComfyUI-LongTextTTSSuite
Author: CyberDickLang
Description: 用于处理长文本文件并生成语音的ComfyUI插件
"""

import os
import json
from typing import List, Dict, Any
import torch
import numpy as np
from pydub import AudioSegment
import re
from .nodes import LongTextSplitterNode, AudioConcatenateFree, IndexSelectFromListNode, ListLengthNode, SubtitleFileLoaderNode, MakeAudioBatchNode, CombineAudioFromList

class TextSplitterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_file": ("STRING", {"default": ""}),
                "max_length": ("INT", {"default": 200, "min": 50, "max": 1000}),
                "overlap": ("INT", {"default": 20, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("text_chunks",)
    FUNCTION = "split_text"
    CATEGORY = "text/audio"

    def split_text(self, text_file: str, max_length: int, overlap: int) -> List[str]:
        if not os.path.exists(text_file):
            raise ValueError(f"File not found: {text_file}")
            
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_length
            if end >= len(text):
                chunks.append(text[start:])
                break
                
            # 在最大长度处寻找句子结束符
            last_period = text.rfind('.', start, end)
            last_question = text.rfind('?', start, end)
            last_exclamation = text.rfind('!', start, end)
            
            # 找到最后一个句子结束符
            last_sentence_end = max(last_period, last_question, last_exclamation)
            
            if last_sentence_end > start:
                end = last_sentence_end + 1
            else:
                # 如果没有找到句子结束符，就在空格处分割
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunks.append(text[start:end])
            start = end - overlap
            
        return (chunks,)

class AudioMergerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio_segments": ("LIST",),
                "output_path": ("STRING", {"default": "output/merged_audio.wav"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("merged_audio_path",)
    FUNCTION = "merge_audio"
    CATEGORY = "text/audio"

    def merge_audio(self, audio_segments: List[str], output_path: str) -> str:
        if not audio_segments:
            raise ValueError("No audio segments provided")
            
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 合并音频片段
        combined = AudioSegment.empty()
        for segment_path in audio_segments:
            if os.path.exists(segment_path):
                segment = AudioSegment.from_file(segment_path)
                combined += segment
                
        # 导出合并后的音频
        combined.export(output_path, format="wav")
        return (output_path,)

NODE_CLASS_MAPPINGS = {
    "LongTextSplitter": LongTextSplitterNode,
    "AudioConcatenateFree": AudioConcatenateFree,
    "IndexSelectFromList": IndexSelectFromListNode,
    "ListLength": ListLengthNode,
    "SubtitleFileLoader": SubtitleFileLoaderNode,
    "MakeAudioBatch": MakeAudioBatchNode,
    "CombineAudioFromList": CombineAudioFromList,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LongTextSplitter": "Long Text Splitter",
    "AudioConcatenateFree": "Audio Concatenate Free",
    "IndexSelectFromList": "Index Select From List",
    "ListLength": "List Length",
    "SubtitleFileLoader": "Subtitle File Loader",
    "MakeAudioBatch": "Make Audio Batch",
    "CombineAudioFromList": "Combine Audio From List/Batch",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 