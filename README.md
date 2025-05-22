# ComfyUI-LongTextTTSSuite

作者：CyberDickLang

本插件为 ComfyUI 提供长文本处理与音频合成相关的多功能节点，支持文本分割、音频拼接、音频合并等多种场景。

---

## 主要功能节点

### 1. 文本处理节点（TextProcessor）
- **功能**：支持按句子、段落或自定义分隔符切分文本，支持字符过滤。
- **输出**：分割后的文本列表及其长度。
- **典型用途**：TTS前的文本预处理。

### 2. 音频拼接free（AudioConcatenateFree）
- **功能**：将两个AUDIO类型音频按指定方向拼接。
- **参数**：
  - `audio1`：第一个音频（AUDIO类型）
  - `audio2`：第二个音频（AUDIO类型）
  - `direction`：拼接方向（right=audio1+audio2，left=audio2+audio1）
- **容错**：如有一个输入为空，直接输出另一个音频。
- **用途**：灵活拼接任意两段音频。

### 3. 音频合并（AUDIO数据）（AudioMerger）
- **功能**：将多个AUDIO类型音频按顺序拼接，支持方向选择。
- **参数**：
  - `audio_list`：AUDIO类型音频列表
  - `direction`：拼接方向（right/left）
- **用途**：批量合并多段AUDIO音频。

### 4. 音频文件合并（路径）（AudioFileMerger）
- **功能**：按文件路径批量合并音频文件，支持主流格式，支持插入间隔。
- **参数**：
  - `audio_segments`：音频文件路径列表
  - `output_path`：输出文件路径
  - `gap_duration`：片段间隔（秒）
- **用途**：适合已有音频文件批量合并场景。

### 5. 创建音频列表（MakeAudioList）
- **功能**：将多行音频文件路径转为列表。
- **用途**：配合音频文件合并节点使用。

### 6. 列表相关节点
- **索引选择（IndexSelectFromList）**：按索引提取列表元素。
- **列表长度（ListLength）**：输出列表长度。

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
- 所有AUDIO类型节点需保证采样率一致
- 节点参数均有详细说明，支持中英文双语（自动跟随系统）

---

## 参考与致谢
- 部分音频节点参考自 [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)

---

如有问题欢迎在评论区或issue区留言反馈！
