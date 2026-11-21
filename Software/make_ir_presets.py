# -*- coding: utf-8 -*-
"""生成三个预置混响脉冲响应（IR）：环绕声 / 浴室 / 大堂

不是从网上下载的采样，而是按“房间声学”模型合成出来的：
前延迟(delay) + 早期反射(early reflections) + 扩散尾巴(diffuse tail)，
尾巴按频率分两段做不同的 RT60（高频衰减更快 = 空气/软表面吸收），
这样听感上才像真实空间，而不是一段加了包络的白噪声。

用法：python make_ir_presets.py
输出：项目根目录 impulse_responses/*.wav（44.1kHz / 16bit / 单声道）
"""
import os
import wave

import numpy as np
from scipy import signal

SR = 44100


def _decay_env(n, rt60, sr=SR):
    """-60 dB 包络：t=rt60 时正好衰减到 1/1000"""
    t = np.arange(n) / sr
    return np.exp(-6.907755 * t / rt60)


def band_noise(n, lo, hi, seed):
    """带限噪声（用 butter 带通，边界放软一些）"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    if hi >= sr_limit:
        sos = signal.butter(2, lo / (SR / 2), btype='high', output='sos')
    elif lo <= 0:
        sos = signal.butter(2, hi / (SR / 2), btype='low', output='sos')
    else:
        sos = signal.butter(2, [lo / (SR / 2), hi / (SR / 2)], btype='band', output='sos')
    return signal.sosfilt(sos, x)


sr_limit = SR / 2


def room_ir(rt60=1.0, predelay_ms=10.0, length=None, early=(),
            hf_ratio=0.55, low_gain=1.0, density=1.0, seed=1,
            flutter=None, swirl=0.0, build_ms=12.0):
    """合成一条 IR

    rt60        混响时间（低频段）
    predelay_ms 直达声与第一次反射之间的空隙
    early       [(毫秒, 幅度), ...] 离散早期反射
    hf_ratio    高频 RT60 相对低频的比例，<1 表示偏暗（软装房间），接近 1 偏亮（瓷砖）
    low_gain    低频段增益（大堂有明显低频堆积）
    density     尾巴噪声的平滑程度，越大越密
    flutter     (间隔毫秒, 强度) 规律性颤动回声，浴室瓷砖墙的典型特征
    swirl       长尾的慢速起伏深度，做出“包围感”
    """
    if length is None:
        length = rt60 * 1.35
    n = int(length * SR)
    ir = np.zeros(n, dtype=np.float64)

    # ---- 早期反射：离散脉冲，带一点点自身衰减 ----
    for delay_ms, amp in early:
        idx = int((predelay_ms + delay_ms) / 1000.0 * SR)
        if 0 <= idx < n:
            ir[idx] += amp

    # ---- 扩散尾巴：低频段 / 高频段分别按不同 RT60 衰减 ----
    low = band_noise(n, 0.0, 900, seed)
    mid = band_noise(n, 900, 4000, seed + 1)
    high = band_noise(n, 4000, 16000, seed + 2)

    low *= _decay_env(n, rt60 * 1.08) * 5.0 * low_gain
    mid *= _decay_env(n, rt60 * 0.85)
    high *= _decay_env(n, rt60 * hf_ratio) * 0.13

    tail = low + mid + high

    # 密度：轻微低通把噪声“揉”得更连贯（density 越大越平滑）
    if density > 0:
        sos = signal.butter(2, min(0.95, 6000.0 / sr_limit), btype='low', output='sos')
        tail = signal.sosfilt(sos, tail) * (1.0 + 0.25 * density)

    # 尾巴不能从 0 突然起来：给 build_ms 的上升，模拟声场建立
    build = np.clip(np.arange(n) / max(1.0, build_ms / 1000.0 * SR), 0.0, 1.0)
    build = build ** 2
    tail *= build

    # 环绕感：长尾做慢速幅度起伏（0.7 Hz 左右），听上去声场在“转”
    if swirl > 0:
        t = np.arange(n) / SR
        tail *= (1.0 + swirl * np.sin(2 * np.pi * 0.7 * t))

    # ---- 颤动回声（浴室）----
    if flutter:
        step_ms, strength = flutter
        idx = int((predelay_ms + step_ms) / 1000.0 * SR)
        k = 0
        sign = 1.0
        while idx < n:
            ir[idx] += sign * strength * np.exp(-3.0 * (idx / SR) / max(rt60, 0.05))
            sign = -sign
            k += 1
            idx = int((predelay_ms + step_ms * (k + 1)) / 1000.0 * SR)

    ir += tail

    # ---- 去掉直流、峰值归一 ----
    ir -= np.mean(ir[:min(len(ir), 200)])
    peak = np.max(np.abs(ir))
    if peak > 1e-9:
        ir /= peak
    return ir


def save_wav(path, data, sr=SR):
    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return len(pcm) / sr


PRESETS = {
    # 浴室：小空间、瓷砖硬表面 → 前延迟极短、早期反射密而亮、颤动回声明显、尾巴短
    "浴室.wav": dict(
        rt60=0.62, predelay_ms=3.0, early=(
            (6, 0.80), (11, 0.55), (17, 0.42), (23, 0.34),
            (31, 0.28), (38, 0.22), (46, 0.18)),
        hf_ratio=0.95, low_gain=0.85, density=0.35, seed=11,
        flutter=(11.0, 0.22),
    ),
    # 大堂：大体积、长混响、低频堆积、空气吸收让高频先没 → 听上去“厚而远”
    "大堂.wav": dict(
        rt60=2.60, predelay_ms=26.0, early=(
            (20, 0.55), (37, 0.40), (54, 0.34), (73, 0.26),
            (95, 0.22), (120, 0.16), (150, 0.12)),
        hf_ratio=0.42, low_gain=1.25, density=0.80, seed=22,
    ),
    # 环绕声：超长前延迟 + 极密极长尾巴 + 慢速起伏 → 声场把乐器包住
    "环绕声.wav": dict(
        rt60=3.40, predelay_ms=46.0, early=(
            (18, 0.30), (29, 0.28), (41, 0.26), (58, 0.24),
            (77, 0.22), (99, 0.20), (128, 0.18), (160, 0.16)),
        hf_ratio=0.62, low_gain=1.10, density=1.20, seed=33,
        swirl=0.18,
    ),
}


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "impulse_responses")
    os.makedirs(out_dir, exist_ok=True)
    for name, kwargs in PRESETS.items():
        ir = room_ir(**kwargs)
        path = os.path.join(out_dir, name)
        dur = save_wav(path, ir)
        # 简单的 RT60 复核（Schroeder 反向积分到 -60 dB 的时间）
        energy = np.cumsum((ir ** 2)[::-1])[::-1]
        energy_db = 10 * np.log10(np.maximum(energy, 1e-20) / max(energy[0], 1e-20))
        idx = np.argmax(energy_db <= -60.0)
        rt = idx / SR if idx > 0 else float('nan')
        centroid = float(np.sum(np.abs(np.fft.rfft(ir)) * np.fft.rfftfreq(len(ir), 1 / SR))
                         / max(np.sum(np.abs(np.fft.rfft(ir))), 1e-9))
        print(f"{name:10s} {dur:5.2f}s  实测RT60≈{rt:4.2f}s  频谱重心≈{centroid:6.0f}Hz  "
              f"{os.path.getsize(path) // 1024} KB")


if __name__ == "__main__":
    main()
