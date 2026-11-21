# -*- coding: utf-8 -*-
"""
简谱 -> 音序文件（音乐编辑器 2.1.1 / sound_generator.py 可读）

文件格式（UTF-8 纯文本）：
    TimeSignature;4;4                    拍号，; 分隔
    Annotation;小节号;文本                可选批注
    音色,频率Hz,起始,时长,振幅,振幅变化率,{JSON}

时间单位：
    「起始」「时长」的单位是十六分音符。
    4/4 拍下 units_per_measure = 16，因此：
        四分音符(1拍) = 4.00    八分音符 = 2.00    十六分音符 = 1.00
        附点四分      = 6.00    二分音符 = 8.00    附点二分   = 12.00
    播放时 time_unit = 60 / (bpm * 4) 秒。

音高：
    程序内部按 midi = round(12*log2(f/440)+69) 反算音高（再减 12 得内部 pitch），
    所以频率必须填标准十二平均律值，范围 C0(16.35Hz) ~ B7(3951.99Hz)。
    保存时 pitch 被夹在 0~95，对应 MIDI 12~107。
"""

import json
import math
from pathlib import Path

# ---------------------------------------------------------------- 可调参数

DO_FREQ = 392.00        # 简谱 "1" 的频率：1=G 时 do=G4=392.00Hz（1=C 则填 261.63）
WAVETYPE = "triangle"   # 每段统一音色（振荡器波形）
AMP = 0.50              # 每段统一起始振幅（0.0~1.0）
AMP_RATE = 0.0000       # 振幅变化率，0 表示振幅恒定
UNITS_PER_BEAT = 4      # 1 拍 = 4 个十六分音符
BEATS_PER_MEASURE = 4   # 4/4 拍

# ---------------------------------------------------------------- 乐谱数据

# 贝多芬《欢乐颂》——《第九交响曲》第四乐章主题
# 简谱（1=G 4/4）：
#   3 3 4 5 | 5 4 3 2 | 1 1 2 3 | 3. 2 2 -
#   3 3 4 5 | 5 4 3 2 | 1 1 2 3 | 2. 1 1 -
#   2 2 3 1 | 2 34 3 1 | 2 34 3 2 | 1 2 5 3
#   3 3 4 5 | 5 4 3 4 2 | 1 1 2 3 | 2. 1 1 -
# 每项为 (音级, 拍数)：音级 1~7，0 表示休止；8 即高八度的 1。
MELODY = [
    # 第 1 小节   3 3 4 5
    (3, 1), (3, 1), (4, 1), (5, 1),
    # 第 2 小节   5 4 3 2
    (5, 1), (4, 1), (3, 1), (2, 1),
    # 第 3 小节   1 1 2 3
    (1, 1), (1, 1), (2, 1), (3, 1),
    # 第 4 小节   3. 2 2 -
    (3, 1.5), (2, 0.5), (2, 2),

    # 第 5 小节   3 3 4 5
    (3, 1), (3, 1), (4, 1), (5, 1),
    # 第 6 小节   5 4 3 2
    (5, 1), (4, 1), (3, 1), (2, 1),
    # 第 7 小节   1 1 2 3
    (1, 1), (1, 1), (2, 1), (3, 1),
    # 第 8 小节   2. 1 1 -
    (2, 1.5), (1, 0.5), (1, 2),

    # 第 9 小节   2 2 3 1
    (2, 1), (2, 1), (3, 1), (1, 1),
    # 第 10 小节  2 34 3 1   （34 为八分音符）
    (2, 1), (3, 0.5), (4, 0.5), (3, 1), (1, 1),
    # 第 11 小节  2 34 3 2
    (2, 1), (3, 0.5), (4, 0.5), (3, 1), (2, 1),
    # 第 12 小节  1 2 5 3
    (1, 1), (2, 1), (5, 1), (3, 1),

    # 第 13 小节  3 3 4 5
    (3, 1), (3, 1), (4, 1), (5, 1),
    # 第 14 小节  5 4 3 42  （42 为八分音符）
    (5, 1), (4, 1), (3, 1), (4, 0.5), (2, 0.5),
    # 第 15 小节  1 1 2 3
    (1, 1), (1, 1), (2, 1), (3, 1),
    # 第 16 小节  2. 1 1 -
    (2, 1.5), (1, 0.5), (1, 2),
]

# (起始小节, 批注文字)，只影响可读性，不参与发声
ANNOTATIONS = [
    (0, "A段 1-8小节：3345 5432 1123 3.22"),
    (8, "B段 9-12小节：2231 2_34_31 2_34_32 1253"),
    (12, "A'段 13-16小节：3345 543_42_ 1123 2.11"),
]

# ---------------------------------------------------------------- 和声配置

# 教材版和声：全曲只用 I(G) 与 V(D) 两个正三和弦，且只在乐句终止处换和弦。
# 每行为一个小节，元素为 (根音级数, 拍数)，同小节内可换和弦；1=I 5=V。
# 四个乐句的终止：第 4 小节半终止(I->V)、第 8 小节完满终止(V->I)、
#                 第 12 小节半终止(I->V)、第 16 小节完满终止(V->I)。
# 其余小节一律保持主和弦，旋律里的经过音(C、A)按和弦外音处理。
HARMONY = [
    [(1, 4)],              # m01  3 3 4 5
    [(1, 4)],              # m02  5 4 3 2
    [(1, 4)],              # m03  1 1 2 3
    [(1, 1.5), (5, 2.5)],  # m04  3. 2 2-   半终止：3 在 I，2 2 落 V
    [(1, 4)],              # m05  3 3 4 5
    [(1, 4)],              # m06  5 4 3 2
    [(1, 4)],              # m07  1 1 2 3
    [(5, 1.5), (1, 2.5)],  # m08  2. 1 1-   完满终止：V 解决到 I
    [(1, 4)],              # m09  2 2 3 1
    [(1, 4)],              # m10  2 34 3 1
    [(1, 4)],              # m11  2 34 3 2
    [(1, 2), (5, 2)],      # m12  1 2 5 3  半终止
    [(1, 4)],              # m13  3 3 4 5
    [(1, 4)],              # m14  5 4 3 42
    [(1, 4)],              # m15  1 1 2 3
    [(5, 1.5), (1, 2.5)],  # m16  2. 1 1-   完满终止
]

CHORD_SEMITONES = (-12, -8, -5)  # 相对根音的 MIDI 半音：低八度起的大三和弦(0/4/7)
BASS_OFFSET = -24                # 低音再低一个八度（合计比旋律根音低两个八度）

# 振幅：旋律沿用上方 AMP；和弦与低音分设，避免盖过旋律
CHORD_AMP = 0.15      # 和弦内单个音的振幅（三音同时起音会叠加，故明显低于旋律）
BASS_AMP = 0.20       # 低音振幅（低频等响补偿增益更高，故再压低）

OUT_PATH = Path(__file__).resolve().parent.parent / "txtFiles" / "欢乐颂.txt"
OUT_CHORD_PATH = Path(__file__).resolve().parent.parent / "txtFiles" / "欢乐颂-和弦.txt"

# ---------------------------------------------------------------- 转换逻辑


# 简谱 1~7 相对 do 的半音数（大调音阶：全全半全全全半）
MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]

# do 的 MIDI 音高（G4 = 67），由 DO_FREQ 推出，保证频率取十二平均律标准值
DO_MIDI = int(round(12 * math.log2(DO_FREQ / 440.0) + 69))


def degree_to_midi(degree):
    """简谱音级 -> MIDI 音高。degree=1 为 do，可跨八度（8 为高八度 do，负数下行）。"""
    octave, index = divmod(degree - 1, 7)
    return DO_MIDI + MAJOR_STEPS[index] + 12 * octave


def midi_to_freq(midi):
    """MIDI 音高 -> 频率(Hz)，十二平均律。"""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def degree_to_freq(degree):
    """简谱音级 -> 频率(Hz)。degree=0 为休止，返回 None。"""
    if degree == 0:
        return None
    return midi_to_freq(degree_to_midi(degree))


def collect_melody(melody):
    """旋律 -> 事件列表 [(MIDI, 起始, 时长, 振幅)]，时间单位均为十六分音符。

    音高统一用 MIDI 表示，避免和弦音落在 degree=0（简谱休止符）上。
    """
    events = []
    cursor = 0.0
    for degree, beats in melody:
        duration = beats * UNITS_PER_BEAT
        if degree:
            events.append((degree_to_midi(degree), cursor, duration, AMP))
        cursor += duration
    return events, cursor


def collect_chords(harmony):
    """按拍生成柱式和弦与低音。

    harmony 为逐小节的 [(根音级数, 拍数), ...]，同一小节可分段换和弦；
    级数 1/5 在本调内都是大三和弦，故统一取 根音+0/+4/+7 半音；
    和弦声部比旋律低一个八度，低音再低一个八度。
    """
    measure_units = UNITS_PER_BEAT * BEATS_PER_MEASURE
    events = []
    for index, segments in enumerate(harmony):
        total = sum(beats for _, beats in segments)
        if abs(total - BEATS_PER_MEASURE) > 1e-9:
            raise ValueError("第 {} 小节的和声拍数合计为 {}，应为 {}".format(
                index + 1, total, BEATS_PER_MEASURE))

        cursor = index * measure_units
        for root, beats in segments:
            duration = beats * UNITS_PER_BEAT
            root_midi = degree_to_midi(root)
            events.append((root_midi + BASS_OFFSET, cursor, duration, BASS_AMP))
            for semitone in CHORD_SEMITONES:
                events.append((root_midi + semitone, cursor, duration, CHORD_AMP))
            cursor += duration
    return events


def build_sequence(melody, annotations=None, chords=None, time_signature=(4, 4)):
    """把旋律（可选加入和弦层）转换成音序文件的文本行。

    返回 (文本行列表, 总时长/十六分音符)。
    """
    lines = ["TimeSignature;{};{}".format(*time_signature)]

    for measure, text in annotations or []:
        if ";" in text or "\n" in text:
            raise ValueError("批注不能包含分号或换行: " + text)
        lines.append("Annotation;{};{}".format(measure, text))

    events, total_units = collect_melody(melody)
    if chords:
        events += collect_chords(chords)

    # 按起始时间排序；同一时刻先旋律后和弦（sort 稳定，旋律先入表）
    events.sort(key=lambda item: item[1])

    for midi, start, duration, amp in events:
        lines.append("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.4f},{}".format(
            WAVETYPE,
            midi_to_freq(midi),
            start,
            duration,
            amp,
            AMP_RATE,
            json.dumps({}),
        ))

    return lines, total_units


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    melody_lines, total_units = build_sequence(MELODY, ANNOTATIONS)
    OUT_PATH.write_text("\n".join(melody_lines) + "\n", encoding="utf-8")
    print("已写入: {}".format(OUT_PATH))
    print("  单旋律 {} 个音符".format(len(MELODY)))

    # 和弦版沿用同一批批注，避免同一小节叠字
    chord_lines, _ = build_sequence(MELODY, ANNOTATIONS, HARMONY)
    OUT_CHORD_PATH.write_text("\n".join(chord_lines) + "\n", encoding="utf-8")
    chord_notes = sum(len(segments) for segments in HARMONY) * (1 + len(CHORD_SEMITONES))
    print("已写入: {}".format(OUT_CHORD_PATH))
    print("  旋律 {} + 和弦 {} = {} 个音符".format(
        len(MELODY), chord_notes, len(MELODY) + chord_notes))
    melody_measures = total_units / (UNITS_PER_BEAT * BEATS_PER_MEASURE)
    if len(HARMONY) != int(round(melody_measures)):
        print("  警告: 和声小节数 {} 与旋律小节数 {} 不一致".format(
            len(HARMONY), int(round(melody_measures))))

    measures = total_units / (UNITS_PER_BEAT * BEATS_PER_MEASURE)
    print("总长: {:.0f} 个十六分音符 = {} 小节，BPM 90 时约 {:.1f} 秒".format(
        total_units, int(measures), total_units * 60.0 / (90 * 4)))


if __name__ == "__main__":
    main()
