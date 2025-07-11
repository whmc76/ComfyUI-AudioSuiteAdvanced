# ComfyUI-AudioSuiteAdvanced

作者：CyberDickLang

版本号：1.0.1

本插件为 ComfyUI 提供长文本处理与音频合成相关的多功能节点，支持文本分割、音频拼接、音频合并、字幕时间戳对齐、音频分离、说话人分离等多种场景。

---

## 主要功能节点

### 1. Long Text Splitter
- **功能**：支持按句子、段落或自定义分隔符切分文本，支持字符过滤。
- **输出**：分割后的文本列表及其长度。
- **典型用途**：TTS前的文本预处理。

### 2. Subtitle File Loader
- **功能**：加载字幕文件，自动识别并提取文本内容，支持多种主流字幕格式（txt、srt、ass、ssa、vtt、lrc、sub等）。
- **输出**：字幕文本内容、原始文件路径。
- **用法**：可直接拖拽字幕文件到输入框。

### 3. Make Audio Batch
- **功能**：将两个AUDIO类型音频顺序加入队列，自动跳过空输入。
- **输出**：音频队列（LIST）

### 4. Combine Audio From List/Batch
- **功能**：将音频队列合并为一段音频，支持按字幕文件时间戳对齐（支持srt、ass、ssa、vtt、lrc、sub等格式），自动插入静音。
- **参数**：
  - `audio_batch`：AUDIO类型音频队列
  - `gap_duration`：片段间隔（秒）
  - `use_timestamps`：是否根据字幕时间戳对齐
  - `srt_file`：字幕文件路径（支持多格式）
- **输出**：合并后的AUDIO
- **用法示例**：
  1. 用 Subtitle File Loader 加载字幕文件，获取路径
  2. 用 Make Audio Batch 生成音频队列
  3. 用 Combine Audio From List/Batch 合成，勾选 use_timestamps 并输入字幕路径

### 5. Audio Concatenate Free
- **功能**：将两个AUDIO类型音频按指定方向拼接，支持插入静音间隔。
- **参数**：
  - `audio1`、`audio2`：AUDIO类型音频
  - `direction`：拼接方向（right/left）
  - `gap_duration`：间隔时间（秒）

### 6. Audio Separation
- **功能**：将音频分离为四个音轨：低音、鼓点、其他、人声。基于Hybrid Demucs模型。
- **参数**：
  - `audio`：输入音频
  - `chunk_fade_shape`：音频分块重叠区域的淡入淡出效果（linear/half_sine/logarithmic/exponential）
  - `chunk_length`：每个音频块的长度（秒，1.0-30.0）
  - `chunk_overlap`：音频块之间的重叠时间（秒，0.0-1.0）
- **输出**：四个分离的音轨（bass、drums、other、vocals）
- **典型用途**：音乐制作、音频处理、人声提取等。

### 7. Index Select From List
- **功能**：按索引提取列表元素。

### 8. List Length
- **功能**：输出列表长度。

### 9. Speaker Separation_ASAdv
- **功能**：使用 WhisperX 将多说话人音频分离为最多4个人物的独立音轨（如双人对话分别输出两个人的音频）。
- **参数**：
  - `audio`：输入音频（建议16kHz单声道）
  - `max_speakers`：最大说话人数（1~4）
  - `language`：音频语言代码，如'zh'、'en'、'ja'等，'auto'为自动检测
  - `compute_type`：计算精度，float16更快但精度稍低
- **输出**：audio1、audio2、audio3、audio4（不足人数补静音）
- **优势**：基于 WhisperX，兼容性更好，无需额外模型下载，支持多语言。

---

## 安装方法

1. 将本项目克隆到 ComfyUI 的 `custom_nodes` 目录下：
   ```bash
   git clone <本项目地址> ComfyUI/custom_nodes/ComfyUI-AudioSuiteAdvanced
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

---

## WhisperX 说话人分离说明

### 功能特点
- **无需额外配置**：WhisperX 会自动下载所需的模型文件
- **多语言支持**：支持中文、英文、日文等多种语言
- **自动检测**：可自动检测音频语言，也可手动指定
- **高兼容性**：支持 Python 3.8+，无需降级 Python 版本

### 使用建议
- 建议使用 16kHz 单声道音频以获得最佳效果
- 对于较长的音频，处理时间可能较长，请耐心等待
- 可通过 `compute_type` 参数调整计算精度和速度的平衡

---

## 兼容性与说明
- 支持主流音频格式（wav、mp3、ogg、flac、m4a等）
- 支持主流字幕格式（txt、srt、ass、ssa、vtt、lrc、sub等）
- 所有AUDIO类型节点需保证采样率一致
- 节点参数均有详细说明，支持中英文双语（自动跟随系统）
- Audio Separation节点需要torchaudio>=2.3.0和librosa>=0.10.0
- Speaker Separation节点基于WhisperX，兼容性更好

---

## 版本信息
- 当前版本：1.0.1
- 2024-06-09：支持多格式字幕时间戳对齐音频合成，Subtitle File Loader支持多格式字幕解析。
- 2024-12-19：新增Audio Separation节点，支持音频分离为四个音轨。
- 2024-12-20：新增Speaker Separation节点，支持多说话人音频分离，支持本地模型优先。
- 2025-01-10：升级Speaker Separation节点，使用WhisperX替代pyannote.audio，提升兼容性和易用性。

---

## 参考与致谢
- 部分音频节点参考自 [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)
- Audio Separation节点基于 [audio-separation-nodes-comfyui](https://github.com/your-repo/audio-separation-nodes-comfyui)
- Speaker Separation节点基于 [WhisperX](https://github.com/m-bain/whisperX)

---

## 联系方式
- B站/小红书/RunningHub/赛博迪克朗
- YouTube/OpenArt/cyberdicklang
- 邮箱：286878701@qq.com

如有问题欢迎在评论区或issue区留言反馈！
