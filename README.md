# 钢琴卷帘音序器 3.0

一个用 Python + Tkinter 写的钢琴卷帘音序器：在时间轴上点画音符，直接合成波形播放，能导出 WAV。
所有声音都是**实时算出来的**（加法谐波 / 滤波 / 卷积混响），不依赖采样音源库，所以整个项目只有两个第三方依赖。

## 功能

**编辑**

- 钢琴卷帘画布：单击加音符，拖动改起点，Shift 拖动改时长，Ctrl 拖动改音高
- 按住 Alt 拖动框选多个音符，可整体移动、变调、复制
- 按时间段选中 / 复制粘贴：填「第几个 16 分音符」圈定一段，整体复制到新起点
- 多音轨，每轨独立音色与音量，切换音轨时非当前轨自动变暗
- 撤销 / 重做

**合成**

- 谐波类音色：triangle / square / saw / sine / harmonics / harmonica / violin / string / ocarina / organ / steel-tongue-drum
- 打击类与噪声类音色，以及 `audio_sample` 音频采样事件
- **organ 音色的谐波幅值随基波频率变化**：`A_n(f0) = a_n · (f0/261.63)^(-p_n)`，系数由真实管风琴采样拟合而来（详见 `organ_harmonics.py`）
- 谐波类音色统一以基波为 1 归一，改音色不会让整体音量跳变

**效果器块**

- 在画布上拖出一个色块，框住的音符统一加效果：一阶低通、延音、卷积混响
- 混响用真实 IR 卷积，自带三个合成 IR（环绕声 / 浴室 / 大堂）
- 色块可按音色过滤，高度自动贴合所选音色的音域

**其他**

- 多进程并行合成（默认 4 进程，可在设置里调整）
- 导入 MIDI（`midi2seq.py`，纯标准库解析 SMF）
- 导出 WAV、批注、节拍设置（4/4、3/4、6/8）

## 环境要求

- Python 3.9+
- `numpy`、`pygame`（Tkinter 随 Python 自带）

```bash
pip install -r requirements.txt
```

可选依赖（只有辅助脚本用到）：

- `scipy` —— `make_ir_presets.py` 重新生成混响 IR
- `librosa` —— `PianoFFT.py` 频谱分析

## 运行

```bash
cd Software
python 音乐编辑器3.0.0.py
```

## 目录结构

```
Software/
  音乐编辑器3.0.0.py    主程序：界面、编辑、播放、导出
  sound_generator.py    波形合成 / 效果器 DSP
  midi2seq.py            MIDI → 音序文本（纯标准库）
  organ_harmonics.py    管风琴谐波分析与拟合工具
  make_ir_presets.py    合成三个预置混响 IR
  PianoFFT.py           频谱分析（需要 librosa）
  settings.json         用户设置
impulse_responses/      预置混响 IR（由脚本合成，随仓库提供）
examples/               示例音序文件
```

## 关于素材

仓库**不包含**任何第三方音频与曲谱：

- `Music/`（导出的成品）、`txtFiles/`（曲谱）、`Samples/`、`audio_samples/`、`audio_time_stretch/`

程序本身可以只用合成音色运行。想用 `audio_sample` 事件，需要自备采样并放在 `audio_samples/` 下，
音序文件里的 `sample_file` 用相对路径引用即可。

`impulse_responses/` 下的三个 IR 由 `make_ir_presets.py` 用房间声学模型合成，不是外部采样。

## 许可

**尚未声明开源许可。** 在加上 LICENSE 之前，默认保留所有权利。
