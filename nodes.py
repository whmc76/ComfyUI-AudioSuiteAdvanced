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
    """
    长文本拆分节点，支持多种文本切分模式。
    Long Text Splitter Node: Supports various text splitting modes.
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, 
                    "default": "",
                    "description": "输入要处理的文本内容。支持多行文本输入。\nInput the text to be processed. Supports multi-line input."
                }),
                "split_mode": (["sentence", "paragraph", "custom"], {
                    "default": "sentence",
                    "description": "选择文本切分模式：\n- sentence: 按句子切分（使用中英文标点）\n- paragraph: 按段落切分（使用空行）\n- custom: 使用自定义分隔符切分\nSelect text splitting mode:\n- sentence: Split by sentence (Chinese/English punctuation)\n- paragraph: Split by paragraph (empty line)\n- custom: Use custom delimiters"
                }),
                "custom_delimiters": ("STRING", {
                    "default": "。！？.!?",
                    "description": "自定义分隔符列表。仅在custom模式下生效。例如：'。！？.!?' 或 '|'\nCustom delimiter list. Only effective in custom mode. E.g.: '。！？.!?' or '|'"
                }),
                "max_length": ("INT", {
                    "default": 200, 
                    "min": 50, 
                    "max": 1000,
                    "description": "每段文本的最大长度。如果文本超过此长度，将在合适的位置进行切分。\nMaximum length of each text chunk. If exceeded, will split at suitable position."
                }),
                "overlap": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 100,
                    "description": "相邻文本段的重叠长度。用于保持上下文的连贯性，避免切分导致的语义断裂。\nOverlap length between adjacent chunks. Helps keep context continuity."
                }),
                "filter_chars": ("STRING", {
                    "default": "",
                    "description": "要过滤的字符列表。例如：'#$%' 将过滤掉所有 #、$、% 字符。留空表示不过滤任何字符。\nCharacters to filter out. E.g.: '#$%' will remove all #, $, %. Leave empty for no filtering."
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
    """Index Select From List Node: Extracts the element at the specified index from a list."""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_list": ("LIST", {
                    "description": "Input list. Can be a list of texts, audio file paths, etc."
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999,
                    "description": "Index of the element to extract. Starts from 0. If out of range, returns the last element."
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
    """List Length Node: Outputs the length of the input list."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_list": ("LIST", {
                    "description": "Input list. Can be a list of texts, audio file paths, etc."
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
                "description": "Gap duration in seconds between audio1 and audio2."
            }),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "concanate"
    CATEGORY = "text/audio"
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

class SubtitleFileLoaderNode:
    """
    字幕文件加载节点，支持多种主流字幕格式，输出文本内容。
    Subtitle File Loader Node: Supports various subtitle formats, outputs text content.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "",
                    "description": "Subtitle file path (can drag file here). 支持拖拽文件到此输入框。\n支持txt、srt、ass、ssa、vtt、lrc、sub等格式。"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "filepath")
    FUNCTION = "load_subtitle"
    CATEGORY = "text/audio"

    def load_subtitle(self, file_path: str):
        """
        加载字幕文件，自动识别格式并输出纯文本内容和文件路径。
        Load subtitle file, auto-detect format, output plain text and file path.
        """
        import os
        import re
        original_path = file_path
        file_path = file_path.strip().strip('"“"\'')
        if not os.path.exists(file_path):
            return ("", original_path)
        ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        text_lines = []
        if ext == '.srt':
            lines = content.splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.isdigit():
                    continue
                if re.match(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', line):
                    continue
                text_lines.append(line)
            content = '\n'.join(text_lines)
        elif ext in ['.ass', '.ssa']:
            # ASS/SSA: 提取Dialogue行的字幕文本
            lines = content.splitlines()
            for line in lines:
                if line.startswith('Dialogue:'):
                    parts = line.split(',', 9)
                    if len(parts) >= 10:
                        text_lines.append(parts[9].strip())
            content = '\n'.join(text_lines)
        elif ext == '.vtt':
            # VTT: 去除WEBVTT头和时间戳
            lines = content.splitlines()
            for line in lines:
                line = line.strip()
                if not line or line.upper() == 'WEBVTT':
                    continue
                if re.match(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}', line):
                    continue
                text_lines.append(line)
            content = '\n'.join(text_lines)
        elif ext == '.lrc':
            # LRC: 去除时间标签
            lines = content.splitlines()
            for line in lines:
                text = re.sub(r'\[\d{1,2}:\d{2}(?:\.\d{1,2})?\]', '', line).strip()
                if text:
                    text_lines.append(text)
            content = '\n'.join(text_lines)
        elif ext == '.sub':
            # SUB: 简单去除时间戳，保留文本
            lines = content.splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if re.match(r'\{\d+\}\{\d+\}', line):
                    continue
                text_lines.append(line)
            content = '\n'.join(text_lines)
        # 其他格式直接输出原始内容
        return (content, original_path)

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
    CATEGORY = "text/audio"

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
    CATEGORY = "text/audio"

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
            srt_file = srt_file.strip().strip('"“"\'')
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