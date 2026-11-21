# -*- coding: utf-8 -*-
"""管风琴谐波分析 + 拟合工具

用途
----
1. 读一批管风琴采样（本工具默认配合 VSCO 2 CE 的 Keys/Organ/Loud/Rode_Man3Open_*.wav），
   逐个估计基频 f0，然后在 n·f0 处量出第 n 次谐波的幅值；
2. 把每个谐波的幅值拟合成基频的函数：A_n(f0) = a_n · (f0 / f_ref)^(-p_n)，
   即在 log-log 坐标下做一元线性回归（最小二乘）；
3. 打印系数表，并输出可直接粘进 sound_generator.py 的 Python 字面量。

为什么用幂律
------------
管风琴是气鸣乐器，管口辐射相当于一个大致固定的低通/共振包络。基频升高时，
同样的“绝对频率上限”对应的谐波次数变少，相对幅值随 n 下降得更快。
实测各次谐波在 log A_n - log f0 上接近直线，幂律 a·f0^(-p) 用一个尺度 + 一个
指数就能同时描述低音区“谐波丰富”和高音区“接近正弦”，且 p 有明确物理含义：
p 越大，该次谐波随音高衰减得越快。

用法
----
  python organ_harmonics.py <采样目录> [--harmonics 16] [--emit]
  python organ_harmonics.py <采样目录> --download   # 顺便从 jsdelivr 拉样本
"""

import argparse
import glob
import os
import sys
import wave

import numpy as np

# 分析参考频率：C4
F_REF = 261.6256


# ---------------------------------------------------------------- 读写
def read_wav_mono(path):
    """读 wav（16/24/32bit，单/双声道），返回 (单声道 float64 归一化信号, 采样率)"""
    with wave.open(path, "rb") as w:
        n_ch, width, sr, n = (w.getnchannels(), w.getsampwidth(),
                              w.getframerate(), w.getnframes())
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    elif width == 3:
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        v = np.where(v >= 1 << 23, v - (1 << 24), v)
        x = v.astype(np.float64)
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64)
    else:
        raise ValueError("不支持的位深: %d" % width)
    if n_ch > 1:
        x = x[: len(x) // n_ch * n_ch].reshape(-1, n_ch).mean(axis=1)
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak > 0:
        x = x / peak
    return x, sr


# ---------------------------------------------------------------- 基频
def estimate_f0(x, sr, fmin=25.0, fmax=2200.0, start=0.6, length=0.8):
    """自相关估基频（取稳态段，避免起音/混响尾巴）

    自相关用 FFT 算（np.correlate 直接算是 O(n²)，1.2 s 的段要跑几十秒）
    """
    a = int(start * sr)
    b = min(len(x), a + int(length * sr))
    seg = x[a:b]
    if len(seg) < 1024:
        seg = x[: len(x)]
    seg = seg - seg.mean()
    n = len(seg)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(seg, nfft)
    ac = np.fft.irfft(spec * np.conj(spec), nfft)[:n]
    lo = max(2, int(sr / fmax))
    hi = min(len(ac) - 2, int(sr / fmin))
    if hi <= lo:
        raise ValueError("信号太短，无法估计基频")
    k = lo + int(np.argmax(ac[lo:hi]))
    # 抛物线插值，把周期细化到亚采样点
    if 0 < k < len(ac) - 1:
        y0, y1, y2 = ac[k - 1], ac[k], ac[k + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            k = k + 0.5 * (y0 - y2) / denom
    return sr / k


def refine_f0(x, sr, f_hint, n_probe=12, span=0.035, start=0.6, length=1.0):
    """在 f_hint 附近微调基频：让 n·f 上的谐波能量和最大

    自相关在管风琴上很容易锁到低八度（高音区基波弱、二次谐波最响，周期上看就是
    二倍/四倍周期），所以先用“已知音高”给个提示，再按谐波能量找精确值。
    """
    a = int(start * sr)
    b = min(len(x), a + int(length * sr))
    seg = x[a:b] * np.hanning(b - a)
    nfft = 1 << int(np.ceil(np.log2(max(4096, (b - a) * 2))))
    spec = np.abs(np.fft.rfft(seg, nfft))
    df = sr / nfft

    def score(f):
        s = 0.0
        for i in range(1, n_probe + 1):
            fh = f * i
            if fh >= sr / 2 - 2 * df:
                break
            lo = max(1, int((fh * 0.98) / df))
            hi = min(len(spec) - 1, int((fh * 1.02) / df))
            if hi > lo:
                s += np.sum(spec[lo:hi + 1] ** 2)
        return s

    best_f, best_s = f_hint, -1.0
    for f in np.linspace(f_hint * (1 - span), f_hint * (1 + span), 141):
        s = score(f)
        if s > best_s:
            best_s, best_f = s, f
    return best_f


# ---------------------------------------------------------------- 谐波
def harmonic_amplitudes(x, sr, f0, n_harm=16, start=0.6, length=1.0):
    """在 n·f0 处量谐波幅值（±3% 带宽内的频谱 RMS）

    为什么用带宽 RMS 而不是取单个峰：管风琴采样有轻微颤音/合唱（多个音管略微失谐），
    一条谐波会散成几条相近的谱线，取单峰会被“正好落在缝里”的频点骗到（相邻谐波差几十倍）。
    带宽内平方和开根号，相当于把这条谐波的功率整个收进来，稳定得多。

    返回 (幅值数组, 是否有效数组)
    """
    a = int(start * sr)
    b = min(len(x), a + int(length * sr))
    seg = x[a:b] * np.hanning(b - a)
    nfft = 1 << int(np.ceil(np.log2(max(4096, (b - a) * 4))))
    spec = np.abs(np.fft.rfft(seg, nfft))
    df = sr / nfft
    amps = np.zeros(n_harm, dtype=np.float64)
    valid = np.zeros(n_harm, dtype=bool)
    for i in range(n_harm):
        f = f0 * (i + 1)
        if f >= sr / 2 - 2 * df:
            break
        half = max(f * 0.03, 3.0)             # 至少 ±3 Hz，低频谐波也够宽
        lo = max(1, int((f - half) / df))
        hi = min(len(spec) - 1, int((f + half) / df))
        if hi <= lo + 1:
            continue
        amps[i] = np.sqrt(np.sum(spec[lo:hi + 1] ** 2)) / np.sqrt(2.0)
        valid[i] = True
    return amps, valid


# ---------------------------------------------------------------- 拟合
def fit_power_law(f0s, amps, n_harm=16, min_snr=0.004):
    """对每个谐波做 log A = log a - p·log(f0/F_REF) 最小二乘

    amps: (n_sample, n_harm)，已按整条谱的 L2 范数归一化（不是按 A1）
    按 A1 归一化在这里会炸：低音区（C1）管风琴的基波本来就弱到几乎测不到，
    除一下就放大成几十倍，拟合会被一个不可靠的数带跑。
    """
    f0s = np.asarray(f0s, dtype=np.float64)
    out = []
    lx = np.log10(f0s / F_REF)
    for i in range(n_harm):
        y = amps[:, i]
        mask = (y > min_snr) & np.isfinite(y)
        if mask.sum() < 3:
            out.append((0.0, 0.0, 0.0, int(mask.sum())))
            continue
        ly = np.log10(y[mask])
        xx = lx[mask]
        # 两轮：第一轮拟合后剔掉残差 > 2.5σ 的点（个别采样受混响/音栓切换影响会跳）
        keep = np.ones(len(xx), dtype=bool)
        for _ in range(2):
            p, q = np.polyfit(xx[keep], ly[keep], 1)          # ly = p·xx + q
            resid = ly - (p * xx + q)
            sd = resid[keep].std()
            if sd > 0:
                new_keep = np.abs(resid) <= 2.5 * sd
                if new_keep.sum() >= 3:
                    keep = new_keep
        pred = p * xx[keep] + q
        ss_res = np.sum((ly[keep] - pred) ** 2)
        ss_tot = np.sum((ly[keep] - ly[keep].mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        out.append((10 ** q, -p, r2, int(keep.sum())))
    return out


def harmonic_amps_at(coeffs, f0, n_harm=16):
    """按拟合结果算某个基频下的谐波幅值（按 L2 范数归一化，与拟合时同一口径）"""
    f0 = max(1.0, float(f0))
    ratio = f0 / F_REF
    amps = np.array([a * ratio ** (-p) for a, p, _r2, _n in coeffs[:n_harm]],
                    dtype=np.float64)
    norm = np.sqrt(np.sum(amps ** 2))
    return amps / norm if norm > 0 else amps


# ---------------------------------------------------------------- 下载
URL_TMPL = ("https://cdn.jsdelivr.net/gh/sgossner/VSCO-2-CE@master/"
            "Keys/Organ/Loud/Rode_Man3Open_%s.wav")
TAKES = ["01", "04", "07", "10", "13", "16", "19", "22", "25", "28", "31",
         "34", "37", "40", "43", "46", "49", "52", "55", "58", "61"]


def download(out_dir):
    import urllib.request
    os.makedirs(out_dir, exist_ok=True)
    for t in TAKES:
        path = os.path.join(out_dir, "Man_%s.wav" % t)
        if os.path.exists(path) and os.path.getsize(path) > 500000:
            continue
        print("下载", t, flush=True)
        with urllib.request.urlopen(URL_TMPL % t, timeout=120) as r, open(path, "wb") as f:
            f.write(r.read())


# ---------------------------------------------------------------- 主流程
def analyze(files, n_harm=16, verbose=True, expect_start=None, expect_step=3):
    """逐个采样分析

    expect_start/expect_step：VSCO 的 Man3Open 取样按 3 个半音步进，
    给出起始 MIDI 音高后就能给 refine_f0 一个靠谱的提示，避开八度误判。
    """
    f0s, rows, names = [], [], []
    for idx, path in enumerate(files):
        x, sr = read_wav_mono(path)
        if expect_start is not None:
            midi = expect_start + expect_step * idx
            f_hint = 440.0 * 2 ** ((midi - 69) / 12.0)
            f0 = refine_f0(x, sr, f_hint)
        else:
            f0 = estimate_f0(x, sr)
        amps, valid = harmonic_amplitudes(x, sr, f0, n_harm)
        if not valid[0] or amps[0] <= 0:
            if verbose:
                print("  跳过（基频未检出）:", os.path.basename(path))
            continue
        norm = np.sqrt(np.sum(amps[valid] ** 2))
        if norm <= 0:
            continue
        amps = amps / norm
        amps[~valid] = 0.0
        f0s.append(f0)
        rows.append(amps)
        names.append(os.path.basename(path))
        if verbose:
            midi = 69 + 12 * np.log2(f0 / 440.0)
            print("  %-16s f0=%7.2f Hz (MIDI %5.1f)  前6次谐波: %s"
                  % (names[-1], f0, midi,
                     " ".join("%.3f" % v for v in amps[:6])))
    return np.array(f0s), np.array(rows), names


def main():
    ap = argparse.ArgumentParser(description="管风琴谐波分析与幂律拟合")
    ap.add_argument("dir", help="采样目录（Rode_Man3Open_*.wav 或 Man_*.wav）")
    ap.add_argument("--harmonics", type=int, default=16)
    ap.add_argument("--emit", action="store_true", help="输出可粘贴的 Python 系数表")
    ap.add_argument("--download", action="store_true", help="从 jsdelivr 下载 VSCO 2 CE 管风琴样本")
    ap.add_argument("--expect-start", type=int, default=24,
                    help="第一个采样的 MIDI 音高（VSCO Man3Open 为 24=C1）")
    ap.add_argument("--expect-step", type=int, default=3, help="相邻采样的半音数")
    ap.add_argument("--no-expect", action="store_true", help="不用音高提示，纯自相关估 f0")
    args = ap.parse_args()

    if args.download:
        download(args.dir)

    files = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    if not files:
        print("目录里没有 wav:", args.dir)
        return 1
    print("分析 %d 个采样" % len(files))
    f0s, rows, _names = analyze(
        files, args.harmonics,
        expect_start=None if args.no_expect else args.expect_start,
        expect_step=args.expect_step)

    coeffs = fit_power_law(f0s, rows, args.harmonics)
    print("\n谐波幂律拟合 A_n(f0) = a_n·(f0/%.2f)^(-p_n)" % F_REF)
    print("%3s %10s %8s %6s %4s" % ("n", "a_n", "p_n", "R²", "样本"))
    for i, (a, p, r2, n) in enumerate(coeffs, 1):
        print("%3d %10.6f %8.4f %6.3f %4d" % (i, a, p, r2, n))

    print("\n核对：几个音高上的谐波幅值（拟合值 / 实测中位）")
    for probe in (65.41, 130.81, 261.63, 523.25, 1046.5):
        amps = harmonic_amps_at(coeffs, probe, args.harmonics)
        print("  f0=%7.2f Hz: %s" % (probe, " ".join("%.3f" % v for v in amps[:8])))

    if args.emit:
        print("\n# ---- 粘进 sound_generator.py ----")
        print("ORGAN_HARMONICS = [")
        for i, (a, p, _r2, _n) in enumerate(coeffs, 1):
            print("    (%.6f, %.4f),   # H%d" % (a, p, i))
        print("]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
