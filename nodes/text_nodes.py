"""
文本处理节点模块
Text Processing Nodes Module
"""

import re
import os
import torch
import tempfile
import json
from typing import List, Dict, Any, Union, Tuple
from pathlib import Path

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

import numpy as np
import torchaudio
from faster_whisper import WhisperModel
import soundfile as sf
import tempfile
import whisperx

class MultiSpeakerSpeechToText:
    CATEGORY = "AudioSuiteAdvanced"
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "description": "输入音频文件 (支持多人说话识别)"
                }),
                "language": ([
                    "auto", "zh", "en", "ja", "fr", "de", "ru", "es", "it", "ko"
                ], {
                    "default": "auto",
                    "description": "识别语言代码，auto为自动检测"
                }),
                "whisper_model": ([
                    "tiny", "base", "small", "medium", "large-v2"
                ], {
                    "default": "medium",
                    "description": "faster-whisper模型类型"
                }),
                "use_whisperx": ("BOOLEAN", {
                    "default": True,
                    "description": "使用 WhisperX 进行说话人分离（推荐）"
                }),
                "auth_token": ("STRING", {
                    "default": "",
                    "description": "HuggingFace auth token (仅在使用 pyannote 时需要)"
                }),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("subtitle_json",)
    FUNCTION = "execute"
    DESCRIPTION = "多人语音转文字，输出标准 JSON，供角色分离节点使用。"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return args, kwargs

    def execute(self, audio: Dict[str, Any], language: str, whisper_model: str, use_whisperx: bool, auth_token: str):
        import shutil
        # 1. 保存音频为临时 wav 文件
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        if waveform.dim() == 3:
            waveform = waveform[0]  # [B, C, N] -> [C, N]
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # 多通道转单通道
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)  # [1, N] -> [N]
        waveform_np = waveform.cpu().numpy()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            sf.write(tmp_wav.name, waveform_np, sample_rate)
            wav_path = tmp_wav.name

        try:
            if use_whisperx:
                # 使用 WhisperX 进行说话人分离和转写
                results = self._process_with_whisperx(wav_path, language, whisper_model)
            else:
                # 使用 pyannote + faster-whisper
                results = self._process_with_pyannote(wav_path, language, whisper_model, auth_token)
        except Exception as e:
            print(f"[ERROR] 处理失败: {e}")
            results = []
        finally:
            os.remove(wav_path)
        
        # 输出标准 JSON
        return (json.dumps(results, ensure_ascii=False, indent=2),)
    
    def _process_with_whisperx(self, wav_path: str, language: str, whisper_model: str) -> List[Dict]:
        """使用 WhisperX 进行说话人分离和转写"""
        print("[INFO] 使用 WhisperX 进行说话人分离和转写")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16"
        
        # 1. 加载 WhisperX 模型
        if whisper_model == "large-v2":
            model_type = "large-v2"
        elif whisper_model == "medium":
            model_type = "medium"
        elif whisper_model == "small":
            model_type = "small"
        elif whisper_model == "base":
            model_type = "base"
        elif whisper_model == "tiny":
            model_type = "tiny"
        else:
            model_type = "medium"
        
        model = whisperx.load_model(model_type, device, compute_type=compute_type)
        audio = whisperx.load_audio(wav_path)
        
        # 2. 转写
        result = model.transcribe(audio, batch_size=4)
        language_code = result["language"]
        
        # 3. 时间对齐
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        
        # 清理模型内存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        del model_a, model
        
        # 4. 说话人分离
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=None, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # 清理内存
        gc.collect()
        torch.cuda.empty_cache()
        del diarize_model
        
        # 5. 格式化输出
        results = []
        for i, segment in enumerate(result["segments"]):
            try:
                speaker_name = segment.get("speaker", f"SPEAKER_{i:02d}")
            except:
                speaker_name = f"SPEAKER_{i:02d}"
            
            # 时间格式化
            def sec2str(sec):
                m, s = divmod(int(sec), 60)
                return f"{m}:{s:02d}"
            
            results.append({
                "id": speaker_name,
                "start": sec2str(segment["start"]),
                "end": sec2str(segment["end"]),
                "text": segment["text"].strip()
            })
        
        return results
    
    def _process_with_pyannote(self, wav_path: str, language: str, whisper_model: str, auth_token: str) -> List[Dict]:
        """使用 pyannote + faster-whisper 进行说话人分离和转写"""
        print("[INFO] 使用 pyannote + faster-whisper 进行说话人分离和转写")
        
        from pyannote.audio import Pipeline
        diarization_model = "pyannote/speaker-diarization@2.1"
        
        # 2. 说话人分离（使用绝对路径避免反斜杠问题）
        comfy_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        local_model_dir = os.path.join(comfy_root, 'models', 'speaker-diarization')
        print(f"[DEBUG] pyannote local_model_dir (absolute): {local_model_dir}")
        
        # 检查本地模型是否存在
        config_file = os.path.join(local_model_dir, "config.yaml")
        pipeline = None
        
        if os.path.exists(config_file):
            print(f"[DEBUG] 使用本地 pyannote 模型: {local_model_dir}")
            try:
                pipeline = Pipeline.from_pretrained(local_model_dir)
            except Exception as e:
                print(f"[WARN] 本地模型加载失败: {e}")
                pipeline = None
        
        if pipeline is None:
            print(f"[DEBUG] 从远程下载 pyannote 模型到: {local_model_dir}")
            try:
                # 使用用户提供的 auth_token 或 None
                use_token = auth_token if auth_token.strip() else None
                pipeline = Pipeline.from_pretrained(diarization_model, cache_dir=local_model_dir, use_auth_token=use_token)
            except Exception as e:
                print(f"[ERROR] 远程下载失败: {e}")
                if not auth_token.strip():
                    print("[INFO] 请访问 https://hf.co/pyannote/speaker-diarization 接受使用条款")
                    print("[INFO] 然后在 auth_token 参数中输入你的 HuggingFace token")
                else:
                    print("[INFO] 请检查你的 auth_token 是否正确")
                return []
        
        if pipeline is None:
            print("[ERROR] pipeline 加载失败")
            return []
        
        diarization = pipeline(wav_path)
        # diarization.itertracks(yield_label=True) -> (segment, track, label)
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })

        # 3. 用 faster-whisper 对每个说话人片段做转写
        model = WhisperModel(whisper_model, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")
        # 语言auto时传None
        lang_arg = None if language == "auto" else language
        results = []
        
        # 重新加载音频用于裁剪
        waveform_np, sample_rate = sf.read(wav_path)
        
        for seg in segments:
            # 裁剪音频片段
            start_sample = int(seg["start"] * sample_rate)
            end_sample = int(seg["end"] * sample_rate)
            seg_audio = waveform_np[start_sample:end_sample]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as seg_wav:
                sf.write(seg_wav.name, seg_audio, sample_rate)
                seg_wav_path = seg_wav.name
            # 识别
            segments_gen, _ = model.transcribe(seg_wav_path, language=lang_arg, beam_size=5, word_timestamps=True)
            text = "".join([s.text for s in segments_gen])
            # 时间格式化
            def sec2str(sec):
                m, s = divmod(int(sec), 60)
                return f"{m}:{s:02d}"
            results.append({
                "id": seg["speaker"],
                "start": sec2str(seg["start"]),
                "end": sec2str(seg["end"]),
                "text": text.strip()
            })
            os.remove(seg_wav_path)
        
        return results 