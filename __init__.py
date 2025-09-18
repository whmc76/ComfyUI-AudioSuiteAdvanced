"""
ComfyUI-AudioSuiteAdvanced v1.0.2
Author: CyberDickLang
Description: 用于处理长文本文件并生成语音的ComfyUI插件，支持音频分离、文本分割、音频拼接等功能
"""

# 从nodes文件夹导入所有节点
from .nodes import (
    LongTextSplitterNode,
    IndexSelectFromListNode,
    ListLengthNode,
    AudioConcatenateFree,
    AudioSeparationNode,
    SubtitleFileLoaderNode,
    MakeAudioBatchNode,
    CombineAudioFromList,
    CharacterVocalExtractor_ASAdv,
    CharacterVocalExtractorMultiTrack,
    MultiSpeakerSpeechToText
)

# 注册所有节点
NODE_CLASS_MAPPINGS = {
    "LongTextSplitter": LongTextSplitterNode,
    "IndexSelectFromList": IndexSelectFromListNode,
    "ListLength": ListLengthNode,
    "AudioConcatenateFree": AudioConcatenateFree,
    "SubtitleFileLoader": SubtitleFileLoaderNode,
    "MakeAudioBatch": MakeAudioBatchNode,
    "CombineAudioFromList": CombineAudioFromList,
    "AudioSeparation": AudioSeparationNode,
    "CharacterVocalExtractor": CharacterVocalExtractor_ASAdv,
    "CharacterVocalExtractorMultiTrack": CharacterVocalExtractorMultiTrack,
    "MultiSpeakerSpeechToText": MultiSpeakerSpeechToText,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LongTextSplitter": "Long Text Splitter (ASA)",
    "IndexSelectFromList": "Index Select From List (ASA)",
    "ListLength": "List Length (ASA)",
    "AudioConcatenateFree": "Audio Concatenate (ASA)",
    "SubtitleFileLoader": "Subtitle File Loader (ASA)",
    "MakeAudioBatch": "Audio Batch Queue (ASA)",
    "CombineAudioFromList": "Combine Audio From List (ASA)",
    "AudioSeparation": "Audio Separation (ASA)",
    "CharacterVocalExtractor": "Character Vocal Extractor (ASA)",
    "CharacterVocalExtractorMultiTrack": "Character Vocal Extractor (Multi-Track) (ASA)",
    "MultiSpeakerSpeechToText": "Multi-Speaker Speech To Text (ASA)",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 