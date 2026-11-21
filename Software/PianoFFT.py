import librosa
import numpy as np
import matplotlib.pyplot as plt

def analyze_harmonics(file_path, n_harmonics=5, f0=None):
    """
    分析MP3文件的整数次谐波幅值
    
    参数:
        file_path: MP3文件路径
        n_harmonics: 要分析的谐波数量(包括基波)
        f0: 基频(Hz)，如果为None则自动估计
        
    返回:
        harmonics: 谐波频率列表
        amplitudes: 相对幅值列表(基波为1)
    """
    # 加载音频文件
    y, sr = librosa.load(file_path, sr=None, mono=True)
    
    # 应用汉宁窗减少频谱泄漏
    window = np.hanning(len(y))
    y_windowed = y * window
    
    # 执行FFT
    fft = np.fft.fft(y_windowed)
    magnitudes = np.abs(fft)  # 获取幅度
    frequencies = np.fft.fftfreq(len(fft), 1/sr)  # 获取频率数组
    
    # 只取正频率部分
    positive_freq_mask = frequencies >= 0
    frequencies = frequencies[positive_freq_mask]
    magnitudes = magnitudes[positive_freq_mask]
    
    # 如果未提供基频，则自动估计
    if f0 is None:
        # 使用librosa的基频估计
        f0 = librosa.pyin(y, fmin=80, fmax=400, sr=sr)[0]
        f0 = np.nanmedian(f0)  # 取中位数避免NaN值
        
    # 寻找最接近估计基频的实际峰值频率
    idx = np.argmin(np.abs(frequencies - f0))
    actual_f0 = frequencies[idx]
    fundamental_amp = magnitudes[idx]
    
    # 计算谐波
    harmonics = []
    harmonic_amplitudes = []
    
    for n in range(1, n_harmonics+1):
        harmonic_freq = actual_f0 * n
        
        # 寻找最接近理论谐波频率的实际峰值
        idx = np.argmin(np.abs(frequencies - harmonic_freq))
        actual_harmonic_freq = frequencies[idx]
        harmonic_amp = magnitudes[idx]
        
        harmonics.append(actual_harmonic_freq)
        harmonic_amplitudes.append(harmonic_amp / fundamental_amp)
    
    return harmonics, harmonic_amplitudes

def plot_harmonics(harmonics, harmonic_amplitudes, file_name):
    """可视化谐波分析结果"""
    plt.figure(figsize=(12, 6))
    
    # 绘制谐波幅值图
    plt.subplot(1, 2, 1)
    positions = np.arange(1, len(harmonic_amplitudes) + 1)
    plt.bar(positions, harmonic_amplitudes, alpha=0.7, color='skyblue')
    plt.xlabel('谐波次数')
    plt.ylabel('相对幅值 (基波=1)')
    plt.title(f'{file_name} - 谐波幅值')
    plt.xticks(positions, [f'H{i}' for i in range(1, len(harmonic_amplitudes) + 1)])
    plt.grid(True, alpha=0.3)
    
    # 绘制极坐标图
    plt.subplot(1, 2, 2, polar=True)
    angles = np.linspace(0, 2 * np.pi, len(harmonic_amplitudes), endpoint=False)
    plt.polar(angles, harmonic_amplitudes, 'o-', linewidth=2)
    plt.thetagrids(np.arange(0, 360, 360/len(harmonic_amplitudes)), 
                   [f'H{i}' for i in range(1, len(harmonic_amplitudes) + 1)])
    plt.title('谐波幅值极坐标表示')
    plt.tight_layout()
    plt.rcParams['font.family'] = ['sans-serif'] # 使用无衬线字体族
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体为黑体（根据上表替换为你系统的字体）
    plt.rcParams['axes.unicode_minus'] = False   # 解决负号'-'显示为方块的问题[2,4](@ref)
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 替换为你的MP3文件路径
    mp3_file = "Samples\小提琴.mp3"    
    # 分析谐波（可以指定基频或设为None自动检测）
    harmonics, amplitudes = analyze_harmonics(mp3_file, n_harmonics=10, f0=740)
    
    # 打印结果
    print("谐波分析结果:")
    print(f"[1.0,")
    for i in range(1, len(harmonics)):
        print(f"{amplitudes[i]:.7f},")
    
    # 可视化结果
    plot_harmonics(harmonics, amplitudes, "示例音频")