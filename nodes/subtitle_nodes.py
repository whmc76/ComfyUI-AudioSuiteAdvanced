"""
字幕处理节点模块
Subtitle Processing Nodes Module
"""

import os
import re
from typing import List, Dict, Any

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
    CATEGORY = "AudioSuiteAdvanced"

    def load_subtitle(self, file_path: str):
        """
        加载字幕文件，自动识别格式并输出纯文本内容和文件路径。
        Load subtitle file, auto-detect format, output plain text and file path.
        """
        import os
        import re
        original_path = file_path
        file_path = file_path.strip().strip('"').strip("'")
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