"""
文本处理节点模块
Text Processing Nodes Module
"""

import re
from typing import List, Dict, Any, Union, Tuple

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
    CATEGORY = "AudioSuiteAdvanced"

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
    CATEGORY = "AudioSuiteAdvanced"

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
    CATEGORY = "AudioSuiteAdvanced"

    def get_length(self, input_list: List[Any]) -> int:
        """返回输入列表的长度
        Args:
            input_list: 输入列表
        Returns:
            列表长度
        """
        return (len(input_list),) 