# -*- coding: utf-8 -*-
"""
MIDI -> 音序文件（音乐编辑器 3.x / sound_generator.py 可读）

只用标准库（struct），不依赖 mido。用法：

    python midi2seq.py 输入.mid -o 输出.txt --voices "0:旋律:0.55,2:低音:0.15"

参数说明：
    --voices     要导出的 MIDI 轨道，格式 "轨道号:音轨名:起始振幅"，逗号分隔。
                 轨道号是文件里的实际顺序（0 起，第 0 轨通常是速度/拍号，不含音符）。
    --bpm        写进文件的时间基准。每单位 = 60/(bpm*4) 秒（即 16 分音符）。
                 编辑器里把 BPM 设成同一个值，速度才和原 MIDI 一致。
    --time-sig   拍号 "分子,分母"，默认沿用 MIDI 的拍号。
    --wavetype   音色，默认 square。
    --lowpass    每个音符开一阶低通（谐波类效果器）。
    --reverb-ir  卷积混响的冲激响应文件；给了就开混响。
    --reverb-mix 混响湿声比例，默认 0.3。
    --quantize   起点/时长的对齐栅格（单位，默认 0.25；给 0 表示不量化）。
    --transpose  整体移调半音数。
    --max-seconds 只导出前 N 秒（调试用）。

时间换算：tick -> 秒 走 MIDI 的 tempo 表（支持中途变速），再乘 bpm 换成单位。
音符按"同音高前一个没停就先掐掉"处理，避免同一个音叠两层听起来变响。
"""

import argparse
import json
import math
import os
import struct


# ---------------------------------------------------------------- MIDI 解析

def _read_vlq(buf, i):
    """读可变长度整数，返回 (值, 新下标)"""
    value = 0
    while True:
        c = buf[i]
        i += 1
        value = (value << 7) | (c & 0x7F)
        if not c & 0x80:
            return value, i


def _split_chunks(data):
    """拆出 MThd 头与所有 MTrk 数据体"""
    if data[:4] != b'MThd':
        raise ValueError('不是 MIDI 文件（缺少 MThd）')
    fmt, ntrk, division = struct.unpack('>HHH', data[8:14])
    if division & 0x8000:
        raise ValueError('不支持 SMPTE 时间格式的 MIDI')
    tracks, i = [], 14
    while i + 8 <= len(data):
        if data[i:i + 4] != b'MTrk':
            i += 1
            continue
        length = struct.unpack('>I', data[i + 4:i + 8])[0]
        tracks.append(data[i + 8:i + 8 + length])
        i += 8 + length
    return fmt, division, tracks


def _events(body):
    """把一条轨道展开成 [(绝对tick, 类型, 参数...)]，处理 running status"""
    out, i, tick, running = [], 0, 0, None
    while i < len(body):
        delta, i = _read_vlq(body, i)
        tick += delta
        if i >= len(body):
            break
        status = body[i]
        if status == 0xFF:                       # meta
            kind = body[i + 1]
            length, j = _read_vlq(body, i + 2)
            out.append((tick, 'meta', kind, body[j:j + length]))
            i = j + length
            continue
        if status in (0xF0, 0xF7):               # sysex
            length, j = _read_vlq(body, i + 1)
            i = j + length
            continue
        if status & 0x80:
            running = status
            i += 1
        else:
            status = running
        high = status & 0xF0
        size = 1 if high in (0xC0, 0xD0) else 2
        out.append((tick, 'ch', status, tuple(body[i:i + size])))
        i += size
    return out


def _tempo_map(tracks, division):
    """收集 (tick, 秒/tick) 的速度表，返回 [(tick, 该点起生效的秒/tick)]"""
    marks = []
    for body in tracks:
        for ev in _events(body):
            if ev[1] == 'meta' and ev[2] == 0x51:
                usec = int.from_bytes(ev[3], 'big')
                marks.append((ev[0], usec / 1e6 / division))
    marks.sort()
    if not marks or marks[0][0] > 0:
        marks.insert(0, (0, 500000 / 1e6 / division))   # 默认 120 BPM
    merged = []
    for tick, sec_per_tick in marks:
        if merged and merged[-1][0] == tick:
            merged[-1] = (tick, sec_per_tick)
        else:
            merged.append((tick, sec_per_tick))
    return merged


def _ticks_to_seconds(tempo, tick):
    """按速度表把绝对 tick 换算成秒"""
    total = 0.0
    for idx, (mark, sec_per_tick) in enumerate(tempo):
        end = tempo[idx + 1][0] if idx + 1 < len(tempo) else None
        if tick <= mark:
            break
        stop = tick if end is None else min(tick, end)
        total += (stop - mark) * sec_per_tick
    return total


def _time_signature(tracks):
    for body in tracks:
        for ev in _events(body):
            if ev[1] == 'meta' and ev[2] == 0x58 and len(ev[3]) >= 2:
                return ev[3][0], 2 ** ev[3][1]
    return 4, 4


def track_notes(body):
    """取出一条轨道的音符 [(起始tick, 结束tick, 音高, 力度)]"""
    open_notes, notes = {}, []
    for tick, kind, status, params in _events(body):
        if kind != 'ch':
            continue
        high = status & 0xF0
        if high in (0xC0, 0xD0):          # 音色切换 / 通道压力，只有一个参数，跳过
            continue
        pitch, vel = params[0], params[1]
        if high == 0x90 and vel > 0:
            open_notes.setdefault(pitch, []).append((tick, vel))
        elif high in (0x80, 0x90):
            queue = open_notes.get(pitch)
            if queue:
                start, velocity = queue.pop(0)
                notes.append((start, tick, pitch, velocity))
    # 没有 note-off 的收尾：给 1 拍长度
    for pitch, queue in open_notes.items():
        for start, velocity in queue:
            notes.append((start, start + 120, pitch, velocity))
    notes.sort()
    return notes


# ---------------------------------------------------------------- 输出

def midi2note_freq(pitch):
    return 440.0 * (2.0 ** ((pitch - 69) / 12.0))


def fmt_json(obj):
    return json.dumps(obj, ensure_ascii=True)


def convert(args):
    data = open(args.midi, 'rb').read()
    fmt, division, tracks = _split_chunks(data)
    tempo = _tempo_map(tracks, division)
    num, den = _time_signature(tracks)
    if args.time_sig:
        num, den = [int(x) for x in args.time_sig.split(',')]

    unit_seconds = 60.0 / (args.bpm * 4.0)
    snap = args.quantize

    voices = []
    for spec in args.voices.split(','):
        parts = spec.split(':')
        idx = int(parts[0])
        name = parts[1] if len(parts) > 1 and parts[1] else '轨道 %d' % (len(voices) + 1)
        amp = float(parts[2]) if len(parts) > 2 else 0.5
        if idx >= len(tracks):
            raise ValueError('轨道 %d 不存在（共 %d 轨）' % (idx, len(tracks)))
        voices.append((idx, name, amp))

    lines = []
    total_units = 0.0
    rows = []
    for track_index, name, amp in voices:
        notes = track_notes(tracks[track_index])
        # 同一个音高前后叠着，就掐掉前一个
        last_end = {}
        kept = []
        for start, end, pitch, velocity in notes:
            start_s = _ticks_to_seconds(tempo, start)
            end_s = _ticks_to_seconds(tempo, end)
            if args.max_seconds and start_s > args.max_seconds:
                continue
            pitch += args.transpose
            if not (0 <= pitch <= 127):
                continue
            start_u = start_s / unit_seconds
            end_u = end_s / unit_seconds
            if snap > 0:
                start_u = round(start_u / snap) * snap
                end_u = round(end_u / snap) * snap
            if pitch in last_end and start_u < last_end[pitch]:
                start_u = last_end[pitch]
            if end_u - start_u < 0.5:
                end_u = start_u + 0.5
            last_end[pitch] = end_u
            rows.append((start_u, pitch, end_u - start_u, amp, name))
            total_units = max(total_units, end_u)

    order = {name: i for i, (_idx, name, _amp) in enumerate(voices)}
    rows.sort(key=lambda r: (r[0], order[r[4]], r[1]))

    lines.append('TimeSignature;%d;%d' % (num, den))
    for i, (_idx, name, _amp) in enumerate(voices):
        lines.append('Track;%d;%s;%s;1.00;0.00;0;0' % (i, name, args.wavetype))

    if args.lowpass or args.reverb_ir:
        fx = {
            'lowpass_1st': {'on': bool(args.lowpass)},
            'sustain': {'on': False, 'value': 0.2},
            'reverb': {'on': bool(args.reverb_ir),
                       'ir': args.reverb_ir or '',
                       'mix': args.reverb_mix},
        }
        block = {'color': '#7E57C2', 'timbres': [args.wavetype], 'fx': fx}
        lines.append('EffectBlock;0.0000;%.4f;0;95;%s' % (total_units + 4, fmt_json(block)))

    if args.annotation:
        lines.append('Annotation;0;%s' % args.annotation)

    for start_u, pitch, dur_u, amp, name in rows:
        lines.append('%s,%.2f,%.2f,%.2f,%.3f,0.0000,{},%d'
                     % (args.wavetype, midi2note_freq(pitch), start_u, dur_u, amp,
                        order[name]))

    text = '\r\n'.join(lines) + '\r\n'
    with open(args.out, 'w', encoding='utf-8', newline='') as fp:
        fp.write(text)

    seconds = total_units * unit_seconds
    print('写出 %s' % args.out)
    print('  音符 %d 条 / 音轨 %d 条 / 时长 %.1f 秒（%.1f 小节）'
          % (len(rows), len(voices), seconds, total_units / (num * 16.0 / den)))
    print('  播放请把 BPM 设为 %g，拍号 %d/%d' % (args.bpm, num, den))
    return 0


def main():
    ap = argparse.ArgumentParser(description='MIDI 转音序文件')
    ap.add_argument('midi')
    ap.add_argument('-o', '--out', required=True)
    ap.add_argument('--voices', default='1:旋律:0.55')
    ap.add_argument('--bpm', type=float, default=90.0)
    ap.add_argument('--time-sig')
    ap.add_argument('--wavetype', default='square')
    ap.add_argument('--lowpass', action='store_true')
    ap.add_argument('--reverb-ir')
    ap.add_argument('--reverb-mix', type=float, default=0.3)
    ap.add_argument('--quantize', type=float, default=0.25)
    ap.add_argument('--transpose', type=int, default=0)
    ap.add_argument('--max-seconds', type=float)
    ap.add_argument('--annotation')
    args = ap.parse_args()
    if args.reverb_ir:
        args.reverb_ir = os.path.abspath(args.reverb_ir)
    return convert(args)


if __name__ == '__main__':
    raise SystemExit(main())
