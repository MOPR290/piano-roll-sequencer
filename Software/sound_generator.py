import numpy as np
import json
import wave
from array import array
import os

# pygame / scipy 只在少数分支里用到，改成用的时候才导入：
# 多进程合成时每个子进程都要重新 import 整个模块，而 pygame 约 1.4 s、scipy 约 2.7 s，
# 一上来就导入会让"第一声"凭空多等好几秒。合成主路径只需要 numpy。
_signal = None
_signal_loaded = False


def _scipy_signal():
    """延迟导入 scipy.signal；没有 scipy 时返回 None（调用方走纯 Python 兜底）"""
    global _signal, _signal_loaded
    if not _signal_loaded:
        _signal_loaded = True
        try:
            from scipy import signal as sig
            _signal = sig
        except ImportError:
            _signal = None
    return _signal


def _pygame():
    """延迟导入 pygame（只有解码音频样本时才需要）"""
    import pygame
    return pygame


# ---------------------------------------------------------------- 管风琴音色
# 数据来源：VSCO 2 CE 的 Keys/Organ/Loud/Rode_Man3Open（CC0），C1~C6 共 21 个取样，
# 每个取样在 n·f0 处量谐波幅值，再按  A_n(f0) = a_n · (f0 / 261.63)^(-p_n)  做 log-log 拟合。
# 表里是 (a_n, p_n)，n = 1..16。拟合脚本见 organ_harmonics.py。
#
# 为什么不是一张固定表：管风琴同一排音管在不同音区发出的相对谐波并不一样——
# 低音区谐波丰富、高音区基波变强，整体越来越接近正弦。幂律里的 p_n 就是这个
# 随音高变化的斜率：p>0 表示该次谐波随基频升高而变弱，p≈0 表示几乎不随音高变。
# 偶数次（H2/H4/H6/H8）的 p 接近 0，是因为这排音栓带了 4'、2' 的倍音管，
# 八度关系的那几次谐波在整个音域里都很稳定。
ORGAN_HARMONICS = [
    (0.109389, -0.6543),   # H1
    (0.486419, -0.5020),   # H2
    (0.082251, -0.7385),   # H3
    (0.451713, -0.0348),   # H4
    (0.070196, 0.2275),    # H5
    (0.251942, 0.0562),    # H6
    (0.040564, 1.1179),    # H7
    (0.253570, 0.2669),    # H8
    (0.018787, 1.3822),    # H9
    (0.037657, 1.0998),    # H10
    (0.014881, 0.9543),    # H11
    (0.076920, 0.6128),    # H12
    (0.007501, 1.0299),    # H13
    (0.023390, 1.3235),    # H14
    (0.002863, 1.3583),    # H15
    (0.140018, 0.4221),    # H16
]

ORGAN_F_REF = 261.6256     # C4，拟合时的参考基频

# 3.0.0：谐波类音色统一「基波定幅」——谐波表里第 1 项恒为 1（基波幅值 = 1），
# 各音色只差一个整体增益 HARMONIC_GAIN。增益 = 0.9 / 峰值因数，保证 start_amp=1 时
# 不削顶，同时让所有谐波音色在同一个 start_amp 下电平可比（以前各自拍 0.3 / 1.0）。
# 峰值因数由各自谐波表算：harmonics 1.00 / harmonica 1.70 / violin 1.06 /
# string 1.36 / ocarina 1.00 / steel-tongue-drum 0.88 / organ 11.8（C4）。
HARMONIC_GAIN = {
    'harmonics':         0.90,
    'harmonica':         0.53,
    'violin':            0.85,
    'string':            0.66,
    'ocarina':           0.90,
    'organ':             0.076,
    'steel-tongue-drum': 0.90,
}


def harmonic_wave(harmonics, angle, gain=1.0):
    """按谐波表合成波形：以基波幅值 1 为基准，gain 只调整体电平

    以前每个分支都是“先除以峰值再乘固定增益”，不同音色、不同音高的实际基波电平
    各不相同；现在谐波表第一项就是 1，不再按峰值缩放，同一个 start_amp 下基波幅值
    一致，音色差异只体现在谐波比例上（generate_wave 末尾仍有 max>1 的兜底限幅）。
    """
    wave = sum(h * np.sin((i + 1) * angle) for i, h in enumerate(harmonics))
    return gain * wave


def organ_harmonics_at(f0, n_harm=None):
    """按基频算管风琴的谐波幅值（按基波归一，H1 = 1，其余为基波的倍数）

    f0 超出取样范围（C1~C6）时直接按幂律外推，两端不会突变。
    """
    coeffs = ORGAN_HARMONICS if n_harm is None else ORGAN_HARMONICS[:n_harm]
    ratio = max(1.0, float(f0)) / ORGAN_F_REF
    amps = np.array([a * ratio ** (-p) for a, p in coeffs], dtype=np.float64)
    # 3.0.0：按基波归一（H1 = 1），与其它谐波音色同一把尺子；以前按 L2 归一，
    # 基波只有 0.05~0.23，看表时没法直接读“第 n 次谐波是基波的几倍”
    return amps / amps[0] if amps[0] > 0 else amps


class SimpleAudioSegment:
    def __init__(self, data, frame_rate, sample_width=2, channels=1):
        self.data = data
        self.frame_rate = frame_rate
        self.sample_width = sample_width  # 1 for 8-bit, 2 for 16-bit
        self.channels = channels
    
    def __add__(self, other):
        if not isinstance(other, SimpleAudioSegment):
            return NotImplemented
        if self.sample_width != other.sample_width:
            raise ValueError("Cannot add audio segments with different bit depths")
        if self.channels != other.channels:
            raise ValueError("Cannot add audio segments with different channel counts")
        return SimpleAudioSegment(self.data + other.data, self.frame_rate,
                                  self.sample_width, self.channels)
        
    def export(self, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.frame_rate)
            wf.writeframes(self.data.tobytes())
    
    def overlay(self, other):
        if len(self.data) != len(other.data) or self.frame_rate != other.frame_rate:
            raise ValueError("音频参数不匹配")
        if self.sample_width != other.sample_width:
            raise ValueError("Cannot overlay audio segments with different bit depths")
        if self.channels != other.channels:
            raise ValueError("Cannot overlay audio segments with different channel counts")
        
        if self.sample_width == 1:
            self_float = np.array(self.data, dtype=np.float32) / 255.0 * 2 - 1
            other_float = np.array(other.data, dtype=np.float32) / 255.0 * 2 - 1
        else:
            self_float = np.array(self.data, dtype=np.float32) / 32767.0
            other_float = np.array(other.data, dtype=np.float32) / 32767.0
        
        mixed = self_float + other_float
        
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed /= max_val
        
        if self.sample_width == 1:
            mixed = ((mixed + 1) / 2 * 255).astype(np.uint8)
            return SimpleAudioSegment(array('B', mixed), self.frame_rate,
                                      sample_width=1, channels=self.channels)
        else:
            mixed = (mixed * 32767).astype(np.int16)
            return SimpleAudioSegment(array('h', mixed), self.frame_rate,
                                      sample_width=2, channels=self.channels)
    
    def tobytes(self):
        """转换为bytes，供pygame使用"""
        return self.data.tobytes()
    
    @staticmethod
    def from_stereo_float(left, right, frame_rate):
        """从左右声道 float32 数组创建立体声 SimpleAudioSegment。
        数据存储为交错16位 int16 数组。
        """
        n = min(len(left), len(right))
        stereo_float = np.column_stack([left[:n], right[:n]])
        max_val = np.max(np.abs(stereo_float))
        if max_val > 1.0:
            stereo_float /= max_val
        stereo_int16 = (stereo_float * 32767).astype(np.int16)
        # 交错: [L,R,L,R,...]
        interleaved = stereo_int16.ravel()
        return SimpleAudioSegment(array('h', interleaved), frame_rate,
                                  sample_width=2, channels=2)
    
    @staticmethod
    def from_mono_array(data_float, frame_rate):
        """从 float32 数组创建单声道 SimpleAudioSegment。"""
        max_val = np.max(np.abs(data_float))
        if max_val > 1.0:
            data_float = data_float / max_val
        data_int16 = (data_float * 32767).astype(np.int16)
        return SimpleAudioSegment(array('h', data_int16), frame_rate,
                                  sample_width=2, channels=1)

def first_order_lowpass(waveform, cutoff_hz, sample_rate=44100):
    """一阶 RC 低通：y[n] = y[n-1] + a·(x[n] - y[n-1])

    a = 1 - exp(-2π·fc/fs)，恒在 (0, 1) 内 → 极点 (1-a) 恒在单位圆内，
    所以**永远不会发散**，也不需要额外限幅。

    听感上它接近积分器：高于截止的成分按 -6 dB/oct 衰减、滞后 90°，
    方波会被磨成圆角、锯齿的高次谐波被压掉；但直流增益是 1（不是无穷），
    不像理想积分器那样把直流分量一路累加直到溢出。
    """
    if waveform is None or len(waveform) == 0:
        return waveform
    # 截止频率夹在 (0, 0.49·fs)：越界时 a 会退化到 0 或 1，滤波失去意义
    cutoff_hz = float(max(1.0, min(cutoff_hz, sample_rate * 0.49)))
    a = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / sample_rate)

    sig = _scipy_signal()
    if sig is not None:
        return sig.lfilter([a], [1.0, a - 1.0], waveform)

    y = np.empty_like(np.asarray(waveform, dtype=np.float64))
    prev = 0.0
    for i in range(len(waveform)):
        prev += a * (waveform[i] - prev)
        y[i] = prev
    return y*1.414


_IR_CACHE = {}


def load_impulse_response(ir_path, sample_rate=44100):
    """读取混响脉冲响应（IR），单声道、归一化到峰值 1，按 (路径, mtime, 采样率) 缓存

    找不到文件或解码失败返回 None —— 调用方据此退化为“不处理”，
    不要因为一个 IR 文件坏了就整曲渲染失败。
    """
    try:
        if not ir_path or not os.path.exists(ir_path):
            return None
        mtime = os.path.getmtime(ir_path)
    except OSError:
        return None

    key = (os.path.abspath(ir_path), mtime, sample_rate)
    cached = _IR_CACHE.get(key)
    if cached is not None:
        return cached

    ir = None
    try:
        import soundfile as sf
        ir, ir_sr = sf.read(ir_path, dtype='float32', always_2d=True)
        ir = ir.mean(axis=1) if ir.ndim > 1 else ir
    except Exception:
        ir = None
        ir_sr = sample_rate

    if ir is None:
        # 退回 scipy 的 wav 读取（只支持 wav，够用）
        try:
            from scipy.io import wavfile
            ir_sr, data = wavfile.read(ir_path)
            data = np.asarray(data)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if data.dtype == np.int16:
                data = data.astype(np.float32) / 32768.0
            elif data.dtype == np.int32:
                data = data.astype(np.float32) / 2147483648.0
            elif data.dtype == np.uint8:
                data = (data.astype(np.float32) - 128.0) / 128.0
            ir = data.astype(np.float32)
        except Exception as e:
            print(f"读取混响 IR 失败: {ir_path} ({e})")
            return None

    if ir is None or len(ir) == 0:
        return None

    if int(ir_sr) != int(sample_rate):
        sig = _scipy_signal()
        if sig is not None:
            n = max(1, int(len(ir) * sample_rate / float(ir_sr)))
            ir = sig.resample(ir, n)
        # 没有 scipy 就不重采样，IR 长短略有偏差，听感上可接受

    peak = float(np.max(np.abs(ir))) if len(ir) else 0.0
    if peak > 1e-9:
        ir = ir / peak
    _IR_CACHE[key] = ir
    return ir


def convolve_reverb(waveform, ir_path, mix=0.3, sample_rate=44100, tail_seconds=4.8):
    """卷积混响：干声与 IR 卷积后按 mix 混合，返回带尾音的更长数组

    - 湿信号峰值对齐干声峰值，改了 mix 不会让整体音量跳变；
    - 尾巴最多保留 tail_seconds 秒，避免长 IR（比如 8 秒教堂）把每段拖得很长；
    - 返回长度 > 输入长度，调用方（generate_audio_from_events）本身会把
      超出的尾音追加进时间轴，所以混响尾巴不会被截断。
    """
    if waveform is None or len(waveform) == 0:
        return waveform
    ir = load_impulse_response(ir_path, sample_rate)
    if ir is None:
        return waveform

    try:
        mix = max(0.0, min(1.0, float(mix)))
    except (TypeError, ValueError):
        mix = 0.3

    dry = np.asarray(waveform, dtype=np.float32)
    max_tail = max(1, int(tail_seconds * sample_rate))
    ir = ir[:max_tail]

    sig = _scipy_signal()
    if sig is not None:
        wet = sig.fftconvolve(dry, ir)[:len(dry) + len(ir) - 1]
    else:
        wet = np.convolve(dry, ir)[:len(dry) + len(ir) - 1]

    dry_peak = float(np.max(np.abs(dry))) if len(dry) else 0.0
    wet_peak = float(np.max(np.abs(wet))) if len(wet) else 0.0
    if wet_peak > 1e-9 and dry_peak > 1e-9:
        wet = wet * (dry_peak / wet_peak)

    wet = wet.astype(np.float32) * mix
    out = np.zeros(len(wet), dtype=np.float32)
    out[:len(dry)] += dry * (1.0 - mix)
    out += wet
    return out


def filtered_input(R, C, input_waveform, x, opt):
    '''opt为1对应低通滤波，反之为高通滤波，最大截止频率为3509.36Hz

    原来按样本逐点递推，是纯 Python 循环，单曲渲染里能占一半以上时间。
    该递推式为一阶 IIR： y[n] = a*x[n] + (1-a)*y[n-1]，可直接用 lfilter 一次算完，
    结果与原循环逐样本一致（误差 ~1e-16，仅浮点舍入差异）。
    没有 scipy 时自动退回逐点循环，保证功能不受影响。
    '''
    tau = R * C
    dt = 1 / 44100
    alpha = dt / tau

    if alpha > 0.5:
        raise ValueError(f"dt/tau={alpha:.3f}过大")

    if len(input_waveform) == 0:
        return input_waveform

    sig = _scipy_signal()
    if sig is not None:
        decay = 1.0 - alpha
        # zi 使 y[0] = x[0]，对齐原实现里的 V_c[0] = input_waveform[0]
        V_c = sig.lfilter([alpha], [1.0, -decay], input_waveform,
                          zi=[(1.0 - alpha) * input_waveform[0]])[0]
        return V_c if opt == 1 else input_waveform - V_c

    V_c = np.zeros_like(input_waveform)
    V_r = np.zeros_like(input_waveform)
    V_c[0] = input_waveform[0]

    for i in range(1, len(x)):
        V_c[i] = V_c[i-1] + dt * (input_waveform[i] - V_c[i-1]) / tau
        V_r[i] = input_waveform[i] - V_c[i]

    return V_c if opt == 1 else V_r

def filter_custom(lower, upper, input_waveform, x):
    """lower为下限频率，upper为上限频率"""
    R = 1000
    C1 = 1/(lower*2*np.pi*R)
    C2 = 1/(upper*2*np.pi*R)
    lower_filtered = filtered_input(R, C1, input_waveform, x, False)
    return filtered_input(R, C2, lower_filtered, x, True)

def wave_selection(angle_freq_over_sp_rate, x, start_amp=1, amp_change_rate=0, wavetype='sine', **kwargs):
    sample_rate = 44100
    t = x / sample_rate
    
    base_angle = angle_freq_over_sp_rate * x
    angle = base_angle
    
    if kwargs.get('vibrato_enabled', False) and kwargs.get('spread_period', 0) > 0 and kwargs.get('vibrato_depth', 0) > 0:
        vibrato_freq = 0.5 / kwargs['spread_period']
        modulation = kwargs['vibrato_depth'] * np.sin(2 * np.pi * vibrato_freq * t + kwargs.get('vibrato_phase_shift', 0))
        if kwargs.get('vibrato_continuous', True):
            angle += modulation
        else:
            half_idx = len(x) // 2
            angle[half_idx:] += modulation[half_idx:]
    
    # 修改：振幅计算 - 使用amp_change_rate替代end_amp
    total_duration = t[-1] if len(t) > 0 else 0
    if total_duration > 0:
        # 线性变化：振幅 = 起始振幅 + 变化率 × 时间
        amplitude = start_amp + amp_change_rate * t
    else:
        amplitude = np.maximum(0,amplitude)
    amplitude=np.clip(amplitude,0,1)
    # 新增：应用max_amp和min_amp限制
    max_amp_val = kwargs.get('max_amp', None)
    min_amp_val = kwargs.get('min_amp', None)
    
    if max_amp_val is not None:
        amplitude = np.minimum(amplitude, max_amp_val)
    if min_amp_val is not None:
        amplitude = np.maximum(amplitude, min_amp_val)
    
    def get_duty(t_val):
        start_duty = kwargs.get('start_duty', 0.5)
        duty_change_rate = kwargs.get('duty_change_rate', 0)  # 默认不变化
        max_duty = kwargs.get('max_duty', 1)  # 默认最大占空比
        min_duty = kwargs.get('min_duty', 0)  # 默认最小占空比
        
        # 线性变化：占空比 = 起始占空比 + 变化率 × 时间
        duty = start_duty + duty_change_rate * t_val * 2
        
        # 应用限制
        duty = np.clip(duty, min_duty, max_duty)
        return duty
    
    if wavetype == 'sine':
        return amplitude * np.sin(angle)
    
    elif wavetype == 'harmonics':
        # 钢琴式包络：无淡入（起音即满幅），之后整段按时间常数 TAU 指数衰减。
        # TAU 取钢琴中音区代表值：C4(261.6 Hz) 实测 T60≈9.5 s（不踩延音踏板），
        # 由 T60 = ln(1000)·τ 反推 τ = 9.5/6.908 ≈ 1.38 s。
        # （低音区更慢，C2 约 20 s；高音区更快，C6 约 12 s，此处取通用中音值）
        TAU_PIANO = 0.3
        total_duration = 5
        total_samples = int(total_duration * sample_rate)
        x = np.arange(total_samples)
        t = x / sample_rate
        amplitude = start_amp * np.exp(-t / TAU_PIANO)
        angle = angle_freq_over_sp_rate * x
        
        harmonics = [1.0, 0.0450323, 0.0071143, 0.0004801, 0.0010696, 0.0004060, 
                     0.0008309, 0.0003858, 0.0004831, 0.0007879, 0.0000005, 
                     0.0000007, 0.0000009, 0.0000031, 0.0000063, 0.0000004, 0.0000130]
        
        base_wave = harmonic_wave(harmonics, angle, HARMONIC_GAIN['harmonics'])
        output_wave = amplitude * base_wave
        
        # 指数尾巴到 5 s 仍有约 -31 dB，末尾补 5 ms 微淡出，避免硬切出爆音
        tail_samples = int(0.005 * sample_rate)
        if tail_samples > 0:
            output_wave[-tail_samples:] *= np.linspace(1, 0, tail_samples)

        return filtered_input(1000, 1e-7, output_wave, x, 1)
    
    elif wavetype == 'steel-tongue-drum':
        # 3.0.0：固定包络 —— 前 0.1s 恒定音量，之后按时间常数 tau 指数衰减，
        # 衰减满 4*tau 后截断。
        # tau 随音符名义时长分档：tau = 0.2*(2*floor(主时长/0.5)+1)
        nominal = float(kwargs.get('_note_duration') or 0.0)
        steps = int(np.floor(max(0.0, nominal) / 0.5 + 1e-9))
        hold_sec = 0.1                        # 恒定音量段
        decay_tau = 0.2 * (2 * steps + 1)     # 指数衰减时间常数
        decay_sec = 4 * decay_tau             # 衰减 4 个时间常数后截断（约 -34.7 dB）
        total_duration = hold_sec + decay_sec
        total_samples = int(total_duration * sample_rate)
        x = np.arange(total_samples)
        t = x / sample_rate
        # 前 0.1s 不衰减，之后才起算指数
        decay_t = np.maximum(0.0, t - hold_sec)
        amplitude = start_amp * np.exp(-decay_t / decay_tau)
        angle = angle_freq_over_sp_rate * x
        base_freq = angle_freq_over_sp_rate * sample_rate / (2 * np.pi)
        harmonics = [1/(np.e**(n)) if n%2==0 else 0 for n in range(10)]
        
        base_wave = harmonic_wave(harmonics, angle, HARMONIC_GAIN['steel-tongue-drum'])
        output_wave = amplitude * base_wave
        # 截断处只做极短淡出（10ms），避免咔哒声，同时不破坏指数衰减形状
        fade_out_samples = min(int(0.01 * sample_rate), total_samples)
        if fade_out_samples > 0:
            fade_out = np.linspace(1, 0, fade_out_samples)
            output_wave[-fade_out_samples:] *= fade_out
        return output_wave

    elif wavetype == 'harmonica':
        harmonics = [1.0, 0.1050600, 0.7823245, 0.1064383, 0.3086024, 0.0917113, 
                     0.2716031, 
                     0.0543184, 
                     0.0018590, 
                     0.0423334, 
                     0.0175479, 0.0067693, 0.0135420, 0.0124118, 0.0083355, 
                     0.0117891]
        
        base_wave = harmonic_wave(harmonics, angle, HARMONIC_GAIN['harmonica'])
        return amplitude * base_wave
    elif wavetype == 'violin':
        harmonics = [1.0,
        0.2406495,
        0.0103159,
        0.0392179,
        0.0473385,
        0.0113894,
        0.0138991,
        0.0105276,
        0.0020875,
        0.0036540]
        
        base_wave = harmonic_wave(harmonics, angle, HARMONIC_GAIN['violin'])
        return amplitude * base_wave
    elif wavetype == 'string':
        harmonics = [1.0,
        0.3036827,
        0.3316727,
        0.3182851,
        0.0278974,
        0.0590347,
        0.0111676,
        0.0166846,
        0.0014442]
        
        base_wave = harmonic_wave(harmonics, angle, HARMONIC_GAIN['string'])
        return amplitude * base_wave
    elif wavetype == 'ocarina':
        harmonics = [1.0]
        
        base_wave = harmonic_wave(harmonics, angle, HARMONIC_GAIN['ocarina'])
        return amplitude * base_wave
    elif wavetype == 'organ':
        # 3.0.0：管风琴。谐波幅值随基频变化（拟合自 VSCO 2 CE 的管风琴取样），
        # 所以这里先由角频率反推 f0，再取该音高对应的谐波配比。
        # 各次谐波的相位都用 sin，起音瞬间不会出现相位堆叠的爆音。
        base_freq = angle_freq_over_sp_rate * sample_rate / (2 * np.pi)
        harmonics = organ_harmonics_at(base_freq)
        base_wave = harmonic_wave(harmonics, angle, HARMONIC_GAIN['organ'])
        return amplitude * base_wave

    elif wavetype == 'square':
        normalized_angle = angle / (2 * np.pi)
        phase = normalized_angle % 1
        duties = np.vectorize(get_duty)(t)
        return amplitude * np.where(phase < duties, 1, -1)
    
    elif wavetype == 'filtered_sq':
        R = kwargs.get("R", 500)
        C = kwargs.get("C", 1.59e-7)
        opt = kwargs.get("opt", 1)
        
        # 首先创建方波
        phase_rad = angle % (2 * np.pi)
        duty_cycle = get_duty(t)
        square_wave = np.ones_like(phase_rad)
        square_wave[phase_rad >= 2 * np.pi * duty_cycle] = -1
        
        # 然后对方波进行滤波
        waveform = filtered_input(R, C, square_wave, x, opt)
        
        '''# 计算波形的最大绝对值
        wv_maxval = np.max(np.abs(waveform))
        
        # 归一化处理（如果最大值小于1）
        if wv_maxval > 0 and wv_maxval < 1:
            waveform /= wv_maxval'''
        
        maxval = kwargs.get("maxval", amplitude)
        
        return np.clip(amplitude * waveform, a_min=-maxval, a_max=maxval)
        
        
    elif wavetype == 'triangle':
        R = kwargs.get("R", 500)
        C = kwargs.get("C", 1.59e-7)
        opt = kwargs.get("opt", 1)
        normalized_angle = angle / (2 * np.pi)
        orig_waveform=amplitude * (2 * np.abs(2 * (normalized_angle % 1 - 0.5)) - 1)
        out_wave=filtered_input(R, C, orig_waveform, x, opt)
        return out_wave
    
    elif wavetype == 'sawtooth':
        R = kwargs.get("R", 500)
        C = kwargs.get("C", 1.59e-7)
        opt = kwargs.get("opt", 1)
        normalized_angle = angle / (2 * np.pi)
        orig_waveform=amplitude * (2 * (normalized_angle % 1) - 1)
        out_wave=filtered_input(R, C, orig_waveform, x, opt)
        out_wave=filtered_input(700, C, out_wave, x, 0)
        return out_wave
    
    elif wavetype == 'noise':
        return amplitude * np.random.uniform(-1, 1, len(angle))
    
    elif wavetype == 'filtered_noise':
        R = kwargs.get("R", 2000)
        fc = kwargs.get("fc", 2100)
        C = 1/(fc*2*np.pi*R)
        return amplitude * filtered_input(R, C, np.random.uniform(-1, 1, len(angle)), x, kwargs.get("opt", True))
    
    elif wavetype == 'double_filtered_noise' or wavetype == 'double_filtered_sq':
        lower = kwargs.get("lo", 10)
        upper = kwargs.get("hi", 22050/(np.pi*2)-100)
        
        if wavetype == 'double_filtered_sq':
            phase_rad = angle % (2 * np.pi)
            duty_cycle = get_duty(t)
            input_wave = np.where(phase_rad >= 2 * np.pi * duty_cycle, -1, 1)
        else:
            input_wave = np.random.uniform(-1, 1, len(angle))
        
        return amplitude * filter_custom(lower, upper, input_wave, x)
    
    elif wavetype == 'beat':
        base_freq = angle_freq_over_sp_rate * sample_rate / (2 * np.pi)
        rand_factor = kwargs.get("rand",1)
        freq_variation = base_freq * (1 + rand_factor * (2 * np.random.random(len(x)) - 1))
        dt = 1.0 / sample_rate
        d_angle = 2 * np.pi * freq_variation * dt
        beat_angle = np.cumsum(d_angle)
        normalized_angle = beat_angle / (2 * np.pi)
        phase = normalized_angle % 1.0
        beat_wave = np.sin(phase) * amplitude 
        beat_wave[:500] = 0
        lo = kwargs.get('lo', max(base_freq*0.5,20))
        hi = kwargs.get('hi', min(base_freq*4,3000))
        if lo is not None and hi is not None:
            return filter_custom(lo, hi, beat_wave, x)
        return beat_wave
    
    elif wavetype == 'empty':
        return np.zeros(len(x))
    
    elif wavetype == 'custom_harmonics':
        coeff_path = kwargs.get('coeff_path', '')
        K_custom = int(kwargs.get('K', 0))  # 0=使用全部可用谐波，不再默认限制50
        if not coeff_path or not os.path.exists(coeff_path):
            return np.zeros(len(x))
        try:
            data = np.load(coeff_path, allow_pickle=True)
            has_wave = 'x' in data and 'y' in data
            has_har = 'har_x' in data and 'har_y' in data

            output_len = len(x)

            # 从 angle_freq_over_sp_rate 反算音符频率
            freq = angle_freq_over_sp_rate * sample_rate / (2 * np.pi)
            if freq <= 0:
                freq = 440.0

            # 从 -har.npz 或 -wave.npz 提取复FFT系数
            if has_har:
                coeffs_x = data['har_x'].astype(complex)
                coeffs_y = data['har_y'].astype(complex)
            elif has_wave:
                x_raw = data['x'].astype(float)
                y_raw = data['y'].astype(float)
                # NaN → 相邻平均
                def fill_nan(arr):
                    arr = arr.copy()
                    for idx in np.where(np.isnan(arr))[0]:
                        lv = arr[idx-1] if idx > 0 else 0
                        rv = arr[idx+1] if idx+1 < len(arr) else 0
                        arr[idx] = (lv + rv) / 2
                    return arr
                n_pts = len(x_raw)
                coeffs_x = np.fft.rfft(fill_nan(x_raw)) / n_pts
                coeffs_y = np.fft.rfft(fill_nan(y_raw)) / n_pts
            else:
                left = np.zeros(output_len)
                right = np.zeros(output_len)
                return (left * start_amp, right * start_amp)

            # 按音符频率生成一个周期
            # K=0 表示使用全部可用谐波；Nyquist 自动裁剪超出采样率半频的谐波
            if K_custom <= 0:
                K_eff = len(coeffs_x) - 1
            else:
                K_eff = min(K_custom, len(coeffs_x) - 1)
            # Nyquist安全裁剪：谐波频率不超过 sample_rate/2
            max_k_nyquist = int(sample_rate / (2 * freq))
            if max_k_nyquist > 0:
                K_eff = min(K_eff, max_k_nyquist)
            period_len = max(2, int(sample_rate / freq))  # 每周期样本数
            t_period = np.linspace(0, 1, period_len, endpoint=False)

            left_one = np.zeros(period_len)
            right_one = np.zeros(period_len)
            for k in range(1, K_eff+1):
                ax, px = np.abs(coeffs_x[k]), np.angle(coeffs_x[k])
                ay, py = np.abs(coeffs_y[k]), np.angle(coeffs_y[k])
                left_one += 2 * ax * np.cos(2*np.pi*k*t_period + px)
                right_one += 2 * ay * np.cos(2*np.pi*k*t_period + py)

            # 联合归一化单周期：使用左右声道的共同最大值，
            # 保持 X/Y 比例不变（示波器音乐依赖此比例）
            mx = max(np.max(np.abs(left_one)), np.max(np.abs(right_one)))
            if mx > 0:
                left_one /= mx
                right_one /= mx

            # tile 到目标时长
            repeats = int(np.ceil(output_len / period_len))
            left = np.tile(left_one, repeats)[:output_len]
            right = np.tile(right_one, repeats)[:output_len]

            return (left * start_amp, right * start_amp)
        except Exception as e:
            print(f"自定义谐波加载失败: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros(len(x))
    
    else:
        raise ValueError(f"不支持的音色: {wavetype}")

def load_audio_file(file_path, target_sample_rate=44100):
    """加载音频文件并转换为目标采样率"""
    try:
        # 使用pygame加载音频
        pygame = _pygame()
        sound = pygame.mixer.Sound(file_path)
        
        # 获取音频数据
        array = pygame.sndarray.array(sound)
        
        # 转换为单声道
        if len(array.shape) > 1:
            array = array.mean(axis=1)
        
        # 转换为浮点数
        array = array.astype(np.float32) / 32767.0
        
        # 获取音频长度（秒）
        audio_length = sound.get_length()
        
        # 计算原始采样率（正确的计算方法）
        original_rate = len(array) / audio_length
        
        print(f"音频信息: 长度={audio_length:.2f}秒, 原始采样率={original_rate:.0f}Hz, 目标采样率={target_sample_rate}Hz")
        
        # 如果需要重采样
        if abs(original_rate - target_sample_rate) > 100:
            from scipy import signal
            # 计算重采样比例
            num_samples = int(len(array) * target_sample_rate / original_rate)
            
            print(f"重采样: {len(array)}样本 -> {num_samples}样本")
            
            # 安全检查：避免创建过大的数组
            if num_samples > 10000000:  # 限制最大样本数
                print(f"警告: 重采样样本数过大({num_samples})，跳过重采样")
                return array[:min(len(array), 10000000)]  # 截断前1000万样本
            
            array = signal.resample(array, num_samples)
        
        return array
    except Exception as e:
        print(f"加载音频文件失败: {str(e)}")
        # 返回空数组作为备选
        return np.zeros(1000, dtype=np.float32)

def generate_wave(wavetype, freq, duration, start_amp=1, amp_change_rate=0, 
                 tail_fadeout=False, fadeout_duration=0.5, sustain_ratio=0.0, **kwargs):
    sample_rate = 44100

    # 3.0.0：延音。主时长结束后，用 sustain_ratio×主时长 的时间把振幅衰减到 0
    try:
        sustain_ratio = max(0.0, min(1.0, float(sustain_ratio or 0.0)))
    except (TypeError, ValueError):
        sustain_ratio = 0.0
    sustain_seconds = duration * sustain_ratio

    # 3.0.0：把音符名义时长（不含 +0.02 尾巴和延音）带给音色分支，
    # steel-tongue-drum 的保持段长度要按它算
    kwargs['_note_duration'] = duration

    # 如果是音频样本类型
    if wavetype == "audio_sample":
        sample_id = kwargs.get('sample_id')
        sample_file = kwargs.get('sample_file')
        
        if sample_file and os.path.exists(sample_file):
            try:
                # 使用pygame加载音频文件
                pygame = _pygame()
                sound = pygame.mixer.Sound(sample_file)
                
                # 获取原始音频数据
                raw_array = pygame.sndarray.array(sound)
                
                # 转换为单声道（如果是立体声）
                if len(raw_array.shape) > 1:
                    raw_array = raw_array.mean(axis=1)
                
                # 获取原始音频长度（秒）
                original_duration = sound.get_length()
                
                # 计算原始采样率
                original_sample_rate = int(len(raw_array) / original_duration)
                
                # 计算所需样本数（基于原始采样率）；3.0.0：延音也要占时长
                play_duration = duration + sustain_seconds
                num_samples_needed = int(play_duration * original_sample_rate)
                
                # 如果事件时长超过音频长度，循环播放
                if play_duration > original_duration:
                    repeats = int(np.ceil(play_duration / original_duration))
                    raw_array = np.tile(raw_array, repeats)
                
                # 截取所需长度的样本
                audio_data = raw_array[:num_samples_needed]
                
                # 如果需要重采样到44100Hz
                if original_sample_rate != sample_rate:
                    from scipy import signal
                    # 计算重采样后的样本数
                    resample_num = int(len(audio_data) * sample_rate / original_sample_rate)
                    audio_data = signal.resample(audio_data, resample_num)
                    # 更新样本数
                    num_samples_needed = int(play_duration * sample_rate)
                    # 再次截取确保长度正确
                    audio_data = audio_data[:num_samples_needed]
                
                # 转换为浮点数并进行归一化
                audio_data = audio_data.astype(np.float32) / 32767.0
                
                # 应用振幅变化
                t = np.arange(len(audio_data)) / sample_rate
                amplitude = start_amp + amp_change_rate * t
                audio_data = audio_data * amplitude.reshape(-1, 1) if len(audio_data.shape) > 1 else audio_data * amplitude
                
                # 3.0.0：延音尾衰减到静音
                tail_len = int(sustain_seconds * sample_rate)
                if tail_len > 0:
                    tail_len = min(tail_len, len(audio_data))
                    ramp = np.linspace(1.0, 0.0, tail_len)
                    if audio_data.ndim > 1:
                        audio_data[-tail_len:] *= ramp.reshape(-1, 1)
                    else:
                        audio_data[-tail_len:] *= ramp
                
                # 应用淡入淡出
                fade_in_length = min(300, len(audio_data) // 10)
                if fade_in_length > 0:
                    fade_in = np.linspace(0, 1, fade_in_length)
                    audio_data[:fade_in_length] *= fade_in.reshape(-1, 1) if len(audio_data.shape) > 1 else audio_data[:fade_in_length] * fade_in
                
                if tail_fadeout:
                    fade_out_length = min(int(fadeout_duration * sample_rate), len(audio_data))
                    if fade_out_length > 0:
                        fade_out = np.linspace(1, 0, fade_out_length)
                        audio_data[-fade_out_length:] *= fade_out.reshape(-1, 1) if len(audio_data.shape) > 1 else audio_data[-fade_out_length:] * fade_out
                
                # 确保值在有效范围内
                max_val = np.max(np.abs(audio_data))
                if max_val > 1.0:
                    audio_data /= max_val
                
                # 转换为16位音频
                waveform_16bit = array('h', (int(sample * 32767) for sample in audio_data.flatten()))
                return SimpleAudioSegment(waveform_16bit, sample_rate)
                
            except Exception as e:
                print(f"加载音频样本失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 如果加载失败，返回空音频
        num_samples = int(duration * sample_rate)
        empty_data = np.zeros(num_samples)
        waveform_16bit = array('h', (int(sample * 32767) for sample in empty_data))
        return SimpleAudioSegment(waveform_16bit, sample_rate)

    angle_freq_over_sp_rate = 2 * np.pi * freq / sample_rate

    duration+=0.02
    if wavetype=="beat":
        duration+=0.15
    # 3.0.0：把延音时长接到主时长后面一起生成
    duration += sustain_seconds
    num_samples = int(duration * sample_rate)
    x = np.arange(num_samples)
    #a=-13.2491, b=443.3763, c=0.002702, d=72.9773
    amp_formula = max(min(10**((-13.2491 * np.log10(freq) + 443.3763 / freq + 0.002702 * freq + 72.9773 - 36)/20), 3),1)
    start_amp *= amp_formula
    # 注意：amp_change_rate也需要按比例调整
    amp_change_rate *= amp_formula
    
    result = wave_selection(angle_freq_over_sp_rate, x, start_amp, amp_change_rate, wavetype, **kwargs)
    is_stereo = isinstance(result, tuple) and len(result) == 2

    # 3.0.0：延音尾衰减（harmonics / steel-tongue-drum 是固定长度，忽略延音）
    tail_len = 0
    if sustain_seconds > 0 and wavetype not in ('harmonics', 'steel-tongue-drum'):
        tail_len = int(sustain_seconds * sample_rate)
    
    if is_stereo:
        left_wave, right_wave = result[0], result[1]
        if tail_len > 0:
            n = min(tail_len, len(left_wave))
            ramp = np.linspace(1.0, 0.0, n)
            left_wave[-n:] *= ramp
            right_wave[-n:] *= ramp
        # 3.0.0：立体声音色（custom_harmonics 等）也要能开一阶低通，
        # 两个声道分别滤，补偿用两声道共同的峰值，避免左右音量被改得不一致
        if kwargs.get('lowpass_1st'):
            peak_before = max(float(np.max(np.abs(left_wave))),
                              float(np.max(np.abs(right_wave)))) if len(left_wave) else 0.0
            left_wave = first_order_lowpass(left_wave, freq * 1.0, sample_rate)
            right_wave = first_order_lowpass(right_wave, freq * 1.0, sample_rate)
            peak_after = max(float(np.max(np.abs(left_wave))),
                             float(np.max(np.abs(right_wave)))) if len(left_wave) else 0.0
            if peak_after > 1e-9 and peak_before > 1e-9:
                gain = peak_before / peak_after
                left_wave = left_wave * gain
                right_wave = right_wave * gain
        # 3.1：立体声段也要能过卷积混响（两声道用同一条 IR，尾巴长度取最长的那个）
        if kwargs.get('reverb_ir'):
            _mix = kwargs.get('reverb_mix', 0.3)
            left_wave = convolve_reverb(left_wave, kwargs['reverb_ir'], _mix, sample_rate)
            right_wave = convolve_reverb(right_wave, kwargs['reverb_ir'], _mix, sample_rate)
            _n = max(len(left_wave), len(right_wave))
            if len(left_wave) < _n:
                left_wave = np.pad(left_wave, (0, _n - len(left_wave)))
            if len(right_wave) < _n:
                right_wave = np.pad(right_wave, (0, _n - len(right_wave)))
        # 直接返回立体声段
        max_val = max(np.max(np.abs(left_wave)), np.max(np.abs(right_wave)))
        if max_val > 1.0:
            left_wave /= max_val
            right_wave /= max_val
        return SimpleAudioSegment.from_stereo_float(left_wave, right_wave, sample_rate)
    
    waveform = result

    # 3.0.0：延音尾衰减到静音
    if tail_len > 0:
        n = min(tail_len, len(waveform))
        waveform[-n:] *= np.linspace(1.0, 0.0, n)

    def process_fade(waveform,type,fade_sec,sample_rate):
        """waveform为波形数组 type表示要选择淡入还是淡出 0为淡入 1为淡出 fade_sec表示淡入淡出的时间长短 sample_rate表示采样率"""
        fade_length = min(int(fade_sec * sample_rate), num_samples)
        if fade_length>0:
            if type==0:
                fade = np.linspace(0, 1, fade_length)
                waveform[:fade_length] *= fade
            elif type==1:
                fade = np.linspace(1, 0, fade_length)
                waveform[-fade_length:] *= fade
        return waveform

    # 修改开始：为harmonica添加自定义淡入淡出
    if wavetype == "harmonics":
        # harmonics 的包络已在 wave_selection 内定义好（无淡入 + 钢琴式指数衰减），
        # 这里不再叠加任何淡入淡出，否则会破坏起音和衰减曲线
        pass
    elif wavetype == 'organ':
        # 管风琴：开管发音很快（起音约 20 ms，没有拨弦乐器的“咬字”过程），
        # 松键后风压立刻泄掉，收尾也短。fi/fo 仍可用 kwargs 覆盖。
        fade_in_sec = kwargs.get('fi', 0.01) * 2
        fade_out_sec = kwargs.get('fo', 0.03) * 2
        process_fade(waveform, 0, fade_in_sec, sample_rate)
        process_fade(waveform, 1, fade_out_sec, sample_rate)
    elif wavetype == 'harmonica' or wavetype=="string" or wavetype=="violin" or wavetype=="ocarina" or wavetype=="beat":
        # 处理自定义淡入
        fade_in_sec = kwargs.get('fi', 0.1) if wavetype!="beat" else 0
        fade_in_sec*=2
        # 处理自定义淡出
        fade_out_sec = kwargs.get('fo', 0.1) if wavetype!="beat" else 0.15
        fade_out_sec*=2
        process_fade(waveform,0,fade_in_sec,sample_rate)
        process_fade(waveform,1,fade_out_sec,sample_rate)
    else:
        # 保留原有淡入逻辑
        fade_in_length = min(300, num_samples // 10)
        # 保留原有淡出逻辑
        fade_out_sec = kwargs.get('fo', 300/sample_rate)
        process_fade(waveform,0,fade_in_length/sample_rate,sample_rate)
        process_fade(waveform,1,fade_out_sec,sample_rate)
    # 修改结束

    # 3.0.0：一阶低通选项（截止 = 基波 × 1.5）。
    # 放在包络/淡入淡出之后、归一化之前：先把整段波形磨圆，再统一拉回 ±1。
    if kwargs.get('lowpass_1st'):
        peak_before = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
        waveform = first_order_lowpass(waveform, freq * 1.0, sample_rate)
        # 峰值补偿：低通滤掉的是高频能量，锯齿波这类陡边波形峰值会掉到 1/4。
        # 按低通前后的峰值比补回来，只改音色不改音量（补偿后峰值 = 原峰值，不会过载）。
        peak_after = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
        if peak_after > 1e-9 and peak_before > 1e-9:
            waveform = waveform * (peak_before / peak_after)

    # 3.1：卷积混响（由“效果器块”注入 reverb_ir / reverb_mix）。
    # 放在低通之后、归一化之前：先把干声磨圆，再和 IR 卷积，
    # 混响尾巴会延长数组，generate_audio_from_events 会把它一起混进时间轴。
    if kwargs.get('reverb_ir'):
        waveform = convolve_reverb(waveform, kwargs['reverb_ir'],
                                   kwargs.get('reverb_mix', 0.3), sample_rate)

    # 确保所有值都在[-1, 1]范围内
    max_val = np.max(np.abs(waveform))
    if max_val > 1.0:
        waveform /= max_val
    
    # 转换为16位音频
    # waveform = [int(sample*7)/8 for sample in waveform]
    # 用 numpy 一次性转换再 frombytes，比逐样本生成器快一个数量级（结果逐字节相同）
    waveform_16bit = array('h')
    waveform_16bit.frombytes(
        (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    return SimpleAudioSegment(waveform_16bit, sample_rate)  # 只返回一个对象


def _render_event(job):
    """渲染单个音符事件，返回 (起始采样点, 声道数, int16 数据, 声像)。

    必须是模块级函数、参数必须可 pickle —— 多进程靠它分发任务。
    """
    event, time_unit, tail_fadeout, fadeout_duration = job
    duration_sec = 2.5 if event['wavetype'] == 'harmonics' else event['duration'] * time_unit
    segment = generate_wave(
        wavetype=event['wavetype'],
        freq=event['freq'],
        duration=duration_sec,
        start_amp=event['start_amp'],
        amp_change_rate=event.get('amp_change_rate', 0),
        tail_fadeout=tail_fadeout,
        fadeout_duration=min(fadeout_duration * duration_sec, 0.5),
        **event['kwargs']
    )
    return (int(event['start_time'] * time_unit * 44100),
            segment.channels,
            np.asarray(segment.data, dtype=np.int16),
            float(event.get('pan', 0.0) or 0.0))


def _render_chunk(job):
    """渲染一小批事件（减少进程间往返次数，按块分发才划算）"""
    events, time_unit, tail_fadeout, fadeout_duration = job
    return [_render_event((e, time_unit, tail_fadeout, fadeout_duration)) for e in events]


_RENDER_POOL = None
_RENDER_POOL_WORKERS = 0


def _get_render_pool(workers):
    """按进程数复用一个进程池：反复播放时不用每次重新拉起子进程。"""
    global _RENDER_POOL, _RENDER_POOL_WORKERS
    if _RENDER_POOL is not None and _RENDER_POOL_WORKERS == workers:
        return _RENDER_POOL
    shutdown_render_pool()
    import multiprocessing as mp
    _RENDER_POOL = mp.Pool(processes=workers)
    _RENDER_POOL_WORKERS = workers
    return _RENDER_POOL


def shutdown_render_pool():
    """关闭合成进程池（改设置或退出程序时调用）"""
    global _RENDER_POOL, _RENDER_POOL_WORKERS
    if _RENDER_POOL is not None:
        try:
            _RENDER_POOL.terminate()
            _RENDER_POOL.join()
        except Exception:
            pass
        _RENDER_POOL = None
    _RENDER_POOL_WORKERS = 0


# 程序被直接关掉时也把子进程收走，免得任务管理器里留一堆 python.exe
import atexit as _atexit
_atexit.register(shutdown_render_pool)


def resolve_workers(workers, n_events):
    """把设置里的进程数换算成实际能用几个，音符太少时多进程反而不划算。"""
    try:
        workers = int(workers or 1)
    except (TypeError, ValueError):
        workers = 1
    if workers <= 1 or n_events < 64:
        return 1
    return max(1, min(workers, os.cpu_count() or 1, n_events // 32))


def _mix_segment(timeline, start_sample, channels, data, pan, sample_rate=44100):
    """把一段已渲染好的 int16 数据叠进时间轴，必要时先把时间轴加长"""
    if start_sample < 0:
        # 起点落在 0 之前（负时间）时，丢掉超出开头的那一截，别让负索引绕到末尾
        skip = -start_sample
        data = data[skip * 2:] if channels == 2 else data[skip:]
        start_sample = 0

    n_frames = len(data) // 2 if channels == 2 else len(data)
    end_sample = start_sample + n_frames
    if end_sample > timeline.shape[0]:
        extra = end_sample - timeline.shape[0]
        if timeline.ndim == 2:
            timeline = np.vstack([timeline, np.zeros((extra, 2), dtype=np.float32)])
        else:
            timeline = np.append(timeline, np.zeros(extra, dtype=np.float32))

    seg = data.astype(np.float32) / 32767.0
    if channels == 2:
        timeline[start_sample:end_sample] += seg.reshape(-1, 2)
    elif timeline.ndim == 2:
        pan = max(-1.0, min(1.0, pan))
        if pan > 0:
            left_gain, right_gain = 1.0 - pan, 1.0
        elif pan < 0:
            left_gain, right_gain = 1.0, 1.0 + pan
        else:
            left_gain = right_gain = 1.0
        timeline[start_sample:end_sample, 0] += seg * left_gain
        timeline[start_sample:end_sample, 1] += seg * right_gain
    else:
        timeline[start_sample:end_sample] += seg
    return timeline


def _timeline_to_segment(timeline, sample_rate=44100):
    """浮点时间轴 -> int16 的 SimpleAudioSegment（归一化 + 立体声交错）"""
    max_val = float(np.max(np.abs(timeline))) if timeline.size else 0.0
    if max_val > 1.0:
        timeline = timeline / max_val
    if timeline.ndim == 2:
        interleaved = (timeline * 32767).astype(np.int16).ravel()
        return SimpleAudioSegment(array('h', interleaved), sample_rate,
                                  sample_width=2, channels=2)
    return SimpleAudioSegment(array('h', (timeline * 32767).astype(np.int16)), sample_rate)


def _generate_parallel(normal_events, time_unit, max_end, has_stereo,
                       tail_fadeout, fadeout_duration, workers, progress=None):
    """多进程合成：子进程各渲染一块，主进程只负责叠加混音。"""
    sample_rate = 44100
    total = len(normal_events)
    base_samples = int(max_end * sample_rate)
    # 混响尾巴最长 4.8 s，先留 5 s 余量，最后再按实际用到的长度裁掉
    if has_stereo:
        timeline = np.zeros((base_samples + sample_rate * 5, 2), dtype=np.float32)
    else:
        timeline = np.zeros(base_samples + sample_rate * 5, dtype=np.float32)

    pool = _get_render_pool(workers)
    chunk = max(4, min(48, total // (workers * 4) or 4))
    jobs = [(normal_events[i:i + chunk], time_unit, tail_fadeout, fadeout_duration)
            for i in range(0, total, chunk)]

    needed = base_samples
    done = 0
    for results in pool.imap(_render_chunk, jobs):
        for start_sample, channels, data, pan in results:
            timeline = _mix_segment(timeline, start_sample, channels, data, pan, sample_rate)
            n_frames = len(data) // 2 if channels == 2 else len(data)
            needed = max(needed, start_sample + n_frames)
        done += len(results)
        if progress is not None:
            try:
                progress(min(done, total), total)
            except Exception:
                pass

    timeline = timeline[:needed] if needed > 0 else timeline[:base_samples]
    return timeline


def generate_audio_from_events(events, time_unit, tail_fadeout=False, fadeout_duration=0.5,
                               progress=None, workers=None):
    """生成普通音色的音乐音频（排除音频样本事件），支持立体声

    progress: 可选回调 progress(done, total)，每合成完一个音符调用一次。
              用来给 GUI 的进度条喂数据；不传就和以前完全一样。
    workers:  合成进程数。>1 时用多进程并行渲染（音符很少或出错会自动退回单进程）。
    """
    sample_rate = 44100
    
    # 过滤掉音频样本事件
    normal_events = [event for event in events if event.get('wavetype') != 'audio_sample']
    
    if not normal_events:
        empty_data = np.zeros(sample_rate, dtype=np.int16)
        return SimpleAudioSegment(array('h', empty_data), sample_rate)
    
    # 判断是否需要立体声：custom_harmonics 本身是立体声，或者有事件带了声像
    has_stereo = any(event.get('wavetype') == 'custom_harmonics' for event in normal_events) \
        or any((event.get('pan') or 0.0) != 0.0 for event in normal_events)
    
    # 计算最大结束时间
    max_end = 0
    for event in normal_events:
        event_duration = 2.5 if event['wavetype'] == 'harmonics' else event['duration'] * time_unit
        event_end = event['start_time'] * time_unit + event_duration
        max_end = max(max_end, event_end)

    # ---- 多进程分支 ----
    workers = resolve_workers(workers, len(normal_events))
    if workers > 1:
        try:
            timeline = _generate_parallel(normal_events, time_unit, max_end, has_stereo,
                                          tail_fadeout, fadeout_duration, workers, progress)
            return _timeline_to_segment(timeline, sample_rate)
        except Exception as exc:
            # 子进程环境有问题（例如从交互式解释器/打包环境启动）就退回单进程
            print(f"多进程合成失败，改用单进程：{exc}")

    # ---- 单进程分支 ----
    # 预分配时多留 5 s 余量，最后按实际用到的长度裁掉。
    # 老写法是每遇到超长段就 np.append 一次整个时间轴，几千音符时会退化成平方复杂度。
    base_samples = int(max_end * sample_rate)
    if has_stereo:
        timeline = np.zeros((base_samples + sample_rate * 5, 2), dtype=np.float32)
    else:
        timeline = np.zeros(base_samples + sample_rate * 5, dtype=np.float32)

    needed = base_samples
    for index, event in enumerate(normal_events, 1):
        start_sample, channels, data, pan = _render_event(
            (event, time_unit, tail_fadeout, fadeout_duration))
        timeline = _mix_segment(timeline, start_sample, channels, data, pan, sample_rate)
        n_frames = len(data) // 2 if channels == 2 else len(data)
        needed = max(needed, max(0, start_sample) + n_frames)

        if progress is not None:
            try:
                progress(index, len(normal_events))
            except Exception:
                pass        # 进度条出问题不该让整首曲子合成失败

    return _timeline_to_segment(timeline[:needed], sample_rate)

def load_sequence_from_txt(txt_filename):
    events = []
    try:
        with open(txt_filename, 'r', encoding='utf-8') as txtfile:
            for line in txtfile:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('TimeSignature'): 
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 7:  # 修改：现在需要至少7个参数
                    continue
                
                try:
                    # 关键修改：读取振幅变化率而不是结束振幅
                    wavetype, freq, start_time, duration, start_amp, amp_change_rate = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    kwargs_str = ",".join(parts[6:]).strip() if len(parts) > 6 else ""
                    # 3.0.0：JSON 之后可能跟着音轨号，先切出来
                    track_index = 0
                    if '}' in kwargs_str:
                        cut = kwargs_str.rfind('}')
                        tail = kwargs_str[cut + 1:].strip().lstrip(',').strip()
                        kwargs_str = kwargs_str[:cut + 1]
                        if tail:
                            try:
                                track_index = int(float(tail))
                            except ValueError:
                                track_index = 0
                    kwargs = {}
                    
                    if kwargs_str:
                        try:
                            if kwargs_str.startswith('{') and kwargs_str.endswith('}'):
                                kwargs = json.loads(kwargs_str)
                            elif ':' in kwargs_str:
                                key, value = kwargs_str.split(':', 1)
                                key = key.strip().strip('"\'')
                                value = value.strip().strip('"\'')
                                try:
                                    value = float(value) if '.' in value else int(value)
                                except ValueError:
                                    pass
                                kwargs[key] = value
                        except:
                            pass
                    
                    events.append({
                        'wavetype': wavetype,
                        'freq': freq,
                        'start_time': start_time,
                        'duration': duration,
                        'start_amp': start_amp,
                        'amp_change_rate': amp_change_rate,  # 关键修改
                        'kwargs': kwargs,
                        'track': track_index,   # 3.0.0
                    })
                except:
                    continue
    except:
        pass
    
    return events

# 注意：play_audio函数已被删除，因为现在使用pygame播放音频


# 旧名保留，兼容既有调用（本函数加载的是音序，不是波形）
load_wave_sequence_from_txt = load_sequence_from_txt
