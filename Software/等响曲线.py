import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def loudness_contour_model(f, a, b, c, d):
    """改进的等响曲线模型"""
    return a * np.log10(f) + b / f + c * f + d

# 使用合理的等响曲线数据
freq_data = np.array([20, 100, 500, 1000, 2000, 4000, 8000, 10000])
spl_data_40_phon = np.array([78, 51, 39, 37, 36, 35, 42, 48])  # 40方等响曲线近似值

try:
    # 初始猜测值调整
    initial_guess = (10, 1000, 0.001, 40)
    params, covariance = curve_fit(loudness_contour_model, freq_data, spl_data_40_phon, p0=initial_guess)
    
    # 计算拟合质量
    spl_pred = loudness_contour_model(freq_data, *params)
    residuals = spl_data_40_phon - spl_pred
    r_squared = 1 - np.var(residuals) / np.var(spl_data_40_phon)
    
    print(f"拟合参数: a={params[0]:.4f}, b={params[1]:.4f}, c={params[2]:.6f}, d={params[3]:.4f}")
    print(f"R² = {r_squared:.4f}")
    
    # 绘图
    freq_plot = np.logspace(np.log10(20), np.log10(20000), 500)
    spl_fit_plot = loudness_contour_model(freq_plot, *params)
    
    plt.figure(figsize=(12, 8))
    plt.semilogx(freq_data, spl_data_40_phon, 'ro', markersize=8, label='ISO 226 数据点 (40方)')
    plt.semilogx(freq_plot, spl_fit_plot, 'b-', label='拟合曲线')
    plt.xlabel('频率 (Hz)')
    plt.ylabel('声压级 (dB SPL)')
    plt.title('40方等响曲线拟合')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.xlim(20, 20000)
    plt.show()
    
except Exception as e:
    print(f"拟合失败: {e}")