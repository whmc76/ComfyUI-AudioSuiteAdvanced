# ComfyUI-LongTextTTSSuite

作者：CyberDickLang

版本号：1.0.1

本插件为 ComfyUI 提供长文本处理与音频合成相关的多功能节点，支持文本分割、音频拼接、音频合并、字幕时间戳对齐等多种场景。

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

### 6. Index Select From List
- **功能**：按索引提取列表元素。

### 7. List Length
- **功能**：输出列表长度。

---

## 安装方法

1. 将本项目克隆到 ComfyUI 的 `custom_nodes` 目录下：
   ```bash
   git clone <本项目地址> ComfyUI/custom_nodes/ComfyUI-LongTextTTSSuite
   ```
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

---

## 兼容性与说明
- 支持主流音频格式（wav、mp3、ogg、flac、m4a等）
- 支持主流字幕格式（txt、srt、ass、ssa、vtt、lrc、sub等）
- 所有AUDIO类型节点需保证采样率一致
- 节点参数均有详细说明，支持中英文双语（自动跟随系统）

---

## 版本信息
- 当前版本：1.0.1
- 2024-06-09：支持多格式字幕时间戳对齐音频合成，Subtitle File Loader支持多格式字幕解析。

---

## 参考与致谢
- 部分音频节点参考自 [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

---

## 联系方式
- B站/小红书/RunningHub/赛博迪克朗
- YouTube/OpenArt/cyberdicklang
- 邮箱：286878701@qq.com

如有问题欢迎在评论区或issue区留言反馈！
