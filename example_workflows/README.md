# 示例工作流

作者：CyberDickLang

本文件夹包含了一些使用 ComfyUI-LongTextTTSSuite 的示例工作流。

## 示例文件说明

1. `basic_workflow.json` - 基础文本转语音工作流
   - 使用 Text Splitter 节点分割文本
   - 使用 TTS 节点生成语音
   - 使用 Audio Merger 节点合并音频

2. `srt_workflow.json` - 字幕文件转语音工作流
   - 处理 SRT 格式字幕文件
   - 保持字幕时间轴信息
   - 生成带时间戳的音频文件

## 使用方法

1. 在 ComfyUI 中加载工作流文件（.json）
2. 确保已安装所需依赖
3. 根据示例修改输入文件路径和参数
4. 运行工作流

## 注意事项

- 示例中的文件路径需要根据您的实际环境进行修改
- 确保输入文件使用 UTF-8 编码
- 建议先使用小文件测试工作流 