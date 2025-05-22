"""
ComfyUI-LongTextTTSSuite
Author: CyberDickLang
Description: 用于处理长文本文件并生成语音的ComfyUI插件
"""

import os
import re
from typing import List, Dict, Any, Union, Tuple
import torch
import numpy as np
from pydub import AudioSegment

class LongTextSplitterNode:
    """长文本拆分节点，支持多种文本切分模式"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "description": "输入要处理的文本内容。支持多行文本输入。"
                }),
                "split_mode": (["sentence", "paragraph", "custom"], {
                    "default": "sentence",
                    "description": "选择文本切分模式：\n- sentence: 按句子切分（使用中英文标点）\n- paragraph: 按段落切分（使用空行）\n- custom: 使用自定义分隔符切分"
                }),
                "custom_delimiters": ("STRING", {
                    "default": "。！？.!?",
                    "description": "自定义分隔符列表。仅在custom模式下生效。例如：'。！？.!?' 或 '|'"
                }),
                "max_length": ("INT", {
                    "default": 200, 
                    "min": 50, 
                    "max": 1000,
                    "description": "每段文本的最大长度。如果文本超过此长度，将在合适的位置进行切分。"
                }),
                "overlap": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 100,
                    "description": "相邻文本段的重叠长度。用于保持上下文的连贯性，避免切分导致的语义断裂。"
                }),
                "filter_chars": ("STRING", {
                    "default": "",
                    "description": "要过滤的字符列表。例如：'#$%' 将过滤掉所有 #、$、% 字符。留空表示不过滤任何字符。"
                }),
            },
        }

    RETURN_TYPES = ("LIST", "INT")
    RETURN_NAMES = ("text_chunks", "text_chunks_length")
    FUNCTION = "process_text"
    CATEGORY = "text/audio"

    def process_text(self, text: str, split_mode: str, custom_delimiters: str, 
                    max_length: int, overlap: int, filter_chars: str) -> Tuple[List[str], int]:
        """处理文本，根据指定模式进行切分
        
        Args:
            text: 输入文本
            split_mode: 切分模式（sentence/paragraph/custom）
            custom_delimiters: 自定义分隔符（仅在custom模式下使用）
            max_length: 每段文本的最大长度
            overlap: 相邻文本段的重叠长度
            filter_chars: 要过滤的字符列表
            
        Returns:
            切分后的文本段列表和文本段数量
        """
        # 移除多余的空白字符，但保留换行符
        text = re.sub(r'[ \t]+', ' ', text).strip()
        
        if not text:
            return ([], 0)
            
        # 过滤指定字符
        if filter_chars:
            # 创建字符过滤正则表达式
            filter_pattern = f'[{re.escape(filter_chars)}]'
            text = re.sub(filter_pattern, '', text)
            
        # 根据不同的切分模式处理文本
        if split_mode == "sentence":
            # 按句子切分（支持中英文标点、换行符）
            delimiters = r'[。！？.!?\n]'
            chunks = self._split_by_delimiters(text, delimiters, max_length, overlap)
        elif split_mode == "paragraph":
            # 按段落切分（空行作为分隔符）
            chunks = self._split_by_paragraphs(text, max_length, overlap)
        else:  # custom mode
            # 按自定义分隔符切分
            delimiters = f'[{re.escape(custom_delimiters)}]'
            chunks = self._split_by_delimiters(text, delimiters, max_length, overlap)
            
        return (chunks, len(chunks))
        
    def _split_by_delimiters(self, text: str, delimiters: str, 
                           max_length: int, overlap: int) -> List[str]:
        """按分隔符切分文本
        
        Args:
            text: 输入文本
            delimiters: 分隔符正则表达式
            max_length: 最大长度
            overlap: 重叠长度
            
        Returns:
            切分后的文本段列表
        """
        # 匹配网址的正则表达式
        url_pattern = r'https?://[^\s]+'
        urls = re.findall(url_pattern, text)
        for i, url in enumerate(urls):
            text = text.replace(url, f'URL_PLACEHOLDER_{i}')
        
        # 匹配数字后面的点号（包括序号和小数）
        number_dot_pattern = r'\d+\.(?=\d|\s|$)'
        number_dots = re.findall(number_dot_pattern, text)
        for i, dot in enumerate(number_dots):
            text = text.replace(dot, f'NUMBER_DOT_{i}')
        
        # 以句末标点或换行符分句（无论前面是否有标点）
        split_pattern = r'(?<=[。！？.!?])(?=\s*\n|\s*$)|(?<=\n)'
        sentences = re.split(split_pattern, text)
        
        # 恢复数字点号
        for i, dot in enumerate(number_dots):
            sentences = [s.replace(f'NUMBER_DOT_{i}', dot) for s in sentences]
        # 恢复网址
        for i, url in enumerate(urls):
            sentences = [s.replace(f'URL_PLACEHOLDER_{i}', url) for s in sentences]
        
        # 合并过短的句子，按最大长度和重叠处理
        chunks = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(s) > max_length:
                chunks.extend(self._split_by_length(s, max_length, overlap))
            else:
                chunks.append(s)
        return chunks
        
    def _split_by_paragraphs(self, text: str, max_length: int, 
                           overlap: int) -> List[str]:
        """按段落切分文本
        
        Args:
            text: 输入文本
            max_length: 最大长度
            overlap: 重叠长度
            
        Returns:
            切分后的文本段列表
        """
        # 预处理：将分隔线替换为特殊标记
        text = re.sub(r'-{3,}', '---SEPARATOR---', text)
        
        # 按回车分割段落
        paragraphs = text.split('\n')
        chunks = []
        current_paragraph = []
        current_title = None
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                # 如果遇到空行且当前段落不为空，处理当前段落
                if current_paragraph:
                    combined = ' '.join(current_paragraph)
                    if len(combined) > max_length:
                        chunks.extend(self._split_by_length(combined, max_length, overlap))
                    else:
                        chunks.append(combined)
                    current_paragraph = []
                continue
                
            # 处理分隔线
            if para == '---SEPARATOR---':
                if current_paragraph:
                    combined = ' '.join(current_paragraph)
                    if len(combined) > max_length:
                        chunks.extend(self._split_by_length(combined, max_length, overlap))
                    else:
                        chunks.append(combined)
                    current_paragraph = []
                continue
                
            # 检查是否是标题行（不包含标点符号且长度较短）
            if not any(c in para for c in '，。！？,.!?') and len(para) < 30:
                # 如果当前段落不为空，先处理它
                if current_paragraph:
                    combined = ' '.join(current_paragraph)
                    if len(combined) > max_length:
                        chunks.extend(self._split_by_length(combined, max_length, overlap))
                    else:
                        chunks.append(combined)
                    current_paragraph = []
                # 保存标题
                current_title = para
                chunks.append(para)
                continue
                
            # 处理普通段落
            if len(para) > max_length:
                # 如果当前段落不为空，先处理它
                if current_paragraph:
                    combined = ' '.join(current_paragraph)
                    if len(combined) > max_length:
                        chunks.extend(self._split_by_length(combined, max_length, overlap))
                    else:
                        chunks.append(combined)
                    current_paragraph = []
                # 处理超长段落
                chunks.extend(self._split_by_length(para, max_length, overlap))
            else:
                current_paragraph.append(para)
        
        # 处理最后一个段落
        if current_paragraph:
            combined = ' '.join(current_paragraph)
            if len(combined) > max_length:
                chunks.extend(self._split_by_length(combined, max_length, overlap))
            else:
                chunks.append(combined)
                
        return chunks
        
    def _split_by_length(self, text: str, max_length: int, 
                        overlap: int) -> List[str]:
        """按长度切分文本
        
        Args:
            text: 输入文本
            max_length: 最大长度
            overlap: 重叠长度
            
        Returns:
            切分后的文本段列表
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_length
            if end >= len(text):
                chunks.append(text[start:])
                break
                
            # 在最大长度处寻找空格
            last_space = text.rfind(' ', start, end)
            if last_space > start:
                end = last_space
                
            chunks.append(text[start:end])
            start = end - overlap if end - overlap > 0 else end
            
        return chunks

class IndexSelectFromListNode:
    """索引选择节点，用于从列表中提取指定索引的元素"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_list": ("LIST", {
                    "description": "输入列表。可以是文本列表、音频文件路径列表等。"
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "description": "要提取的元素索引。从0开始计数。如果索引超出列表范围，将返回最后一个元素。"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected_item",)
    FUNCTION = "select_item"
    CATEGORY = "text/audio"

    def select_item(self, input_list: List[Any], index: int) -> str:
        """从列表中提取指定索引的元素
        
        Args:
            input_list: 输入列表
            index: 要提取的元素索引
            
        Returns:
            选中的元素
        """
        if not input_list:
            return ("",)
            
        # 确保索引在有效范围内
        index = min(index, len(input_list) - 1)
        index = max(0, index)
        
        selected = input_list[index]
        return (str(selected),)

class ListLengthNode:
    """列表长度节点，用于输出输入列表的长度"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_list": ("LIST", {
                    "description": "输入列表。可以是文本列表、音频文件路径列表等。"
                })
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("length",)
    FUNCTION = "get_length"
    CATEGORY = "text/audio"

    def get_length(self, input_list: List[Any]) -> int:
        """返回输入列表的长度
        Args:
            input_list: 输入列表
        Returns:
            列表长度
        """
        return (len(input_list),)

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
                "description": "拼接间隔时间（秒）"
            }),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concanate"
    CATEGORY = "text/audio"
    DESCRIPTION = """
Concatenates the audio1 to audio2 in the specified direction, with optional silence gap.
"""

    def concanate(self, audio1, audio2, direction, gap_duration):
        import torch
        # 容错处理：如果只有一个有效输入，直接返回该音频
        if audio1 is None and audio2 is None:
            raise Exception("audio1 和 audio2 都为空，无法拼接")
        if audio1 is None:
            return (audio2,)
        if audio2 is None:
            return (audio1,)
        sample_rate_1 = audio1["sample_rate"]
        sample_rate_2 = audio2["sample_rate"]
        if sample_rate_1 != sample_rate_2:
            raise Exception("Sample rates of the two audios do not match")
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