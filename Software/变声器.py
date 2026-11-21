import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import librosa
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import threading
import os
import sounddevice as sd
import time
# ------------------------------------------------------------
# 音频变调处理模块（基于 FFT / IFFT，不变时长）
# ------------------------------------------------------------
def _pitch_shift_core(y, sr, semitones):
    """
    核心变调函数：使用 librosa 的相位声码器实现高质量变调，不变时长。
    y : numpy.ndarray, shape (n_samples,), dtype=float32, 范围 [-1,1]
    sr : int, 采样率
    semitones : float, 半音数（正数为升调，负数为降调）
    """
    if semitones == 0:
        return y

    # 确保输入为 float32
    y_float = y.astype(np.float32)
    # 调用 librosa 变调
    y_shifted = librosa.effects.pitch_shift(y_float, sr=sr, n_steps=semitones)

    # 对齐长度（librosa 可能输出略长或略短）
    if len(y_shifted) > len(y):
        y_shifted = y_shifted[:len(y)]
    elif len(y_shifted) < len(y):
        y_shifted = np.pad(y_shifted, (0, len(y) - len(y_shifted)), 'constant')

    return y_shifted.astype(np.float32)


def pitch_shift(y, sr, semitones, band=None):
    """
    变调函数，支持只对指定频率范围进行变调。
    y : numpy.ndarray, 音频数据
    sr : int, 采样率
    semitones : float, 半音数
    band : (low_hz, high_hz) 或 None，指定要变调的频带
    """
    if band is not None:
        low, high = band
        # 设计巴特沃斯带通滤波器（6阶）
        sos = signal.butter(6, [low, high], btype='band', fs=sr, output='sos')
        bandpass = signal.sosfilt(sos, y)          # 提取频带内成分
        remainder = y - bandpass                   # 频带外成分
        bandpass_shifted = _pitch_shift_core(bandpass, sr, semitones)
        return bandpass_shifted + remainder
    else:
        return _pitch_shift_core(y, sr, semitones)


# ------------------------------------------------------------
# 音轨管理
# ------------------------------------------------------------
class Track:
    """单个音轨，包含若干不重叠的音频片段"""
    def __init__(self, sr=44100):
        self.segments = []    # 每个元素为 dict: { 'data', 'start', 'end', 'semitones', 'band' }
        self.sr = sr

    def add_segment(self, data, start, end, semitones=0, band=None):
        """
        添加一个片段，自动检查与现有片段的时间重叠。
        data : numpy.ndarray, 音频数据
        start, end : float, 片段在时间轴上的起止时间（秒）
        semitones : float, 变调半音数
        band : (low, high) 或 None
        """
        # 检查重叠
        for seg in self.segments:
            if not (end <= seg['start'] or start >= seg['end']):
                raise ValueError("新片段与已有片段时间重叠")
        self.segments.append({
            'data': data,
            'start': start,
            'end': end,
            'semitones': semitones,
            'band': band
        })
        # 按起始时间排序
        self.segments.sort(key=lambda x: x['start'])

    def remove_segment(self, index):
        """删除指定索引的片段"""
        if 0 <= index < len(self.segments):
            del self.segments[index]

    def get_total_duration(self):
        """返回该音轨上所有片段的最晚结束时间"""
        if not self.segments:
            return 0
        return max(seg['end'] for seg in self.segments)

    def copy_segment(self, index, offset=0.1):
        """
        复制一个片段，新片段起始时间在原片段起始时间+offset，若重叠则自动后移。
        返回新片段的索引，若无法放置返回None。
        """
        if index < 0 or index >= len(self.segments):
            return None
        seg = self.segments[index]
        new_start = seg['start'] + offset
        new_end = new_start + (seg['end'] - seg['start'])
        # 尝试放置，如果重叠则不断后移
        while True:
            overlap = False
            for other in self.segments:
                if not (new_end <= other['start'] or new_start >= other['end']):
                    overlap = True
                    break
            if not overlap:
                break
            new_start += offset
            new_end += offset
        # 深拷贝音频数据
        new_data = seg['data'].copy()
        self.add_segment(new_data, new_start, new_end, seg['semitones'], seg['band'])
        # 返回新片段索引（排序后）
        for i, s in enumerate(self.segments):
            if s is self.segments[-1]:  # 新添加的在末尾，但排序后可能变化
                break
        # 简单返回新片段索引，实际上需要重新查找
        # 这里直接返回 None，调用者可通过 get_segment_index 查找
        return None

    def get_segment_index(self, seg):
        """返回片段在列表中的索引"""
        for i, s in enumerate(self.segments):
            if s is seg:
                return i
        return -1


# ------------------------------------------------------------
# 混合所有音轨
# ------------------------------------------------------------
def mix_tracks(tracks, sr):
    """将所有音轨的音频混合为一个 numpy 数组（归一化到 [-1,1]）"""
    if not tracks:
        return np.array([], dtype=np.float32)

    total_duration = max(t.get_total_duration() for t in tracks)
    if total_duration == 0:
        return np.array([], dtype=np.float32)

    total_samples = int(total_duration * sr)
    mix = np.zeros(total_samples, dtype=np.float32)

    for track in tracks:
        for seg in track.segments:
            data = seg['data']
            # 应用变调
            if seg['semitones'] != 0 or seg['band'] is not None:
                data = pitch_shift(data, sr, seg['semitones'], seg['band'])
            start_sample = int(seg['start'] * sr)
            end_sample = start_sample + len(data)
            if end_sample > total_samples:
                data = data[:total_samples - start_sample]
                end_sample = total_samples
            mix[start_sample:end_sample] += data

    # 防止削波，归一化到 [-1, 1]
    max_val = np.max(np.abs(mix))
    if max_val > 1.0:
        mix = mix / max_val
    return mix


# ------------------------------------------------------------
# 工具函数：读写 WAV 文件，归一化等
# ------------------------------------------------------------
def load_wav(filepath, target_sr=44100):
    """读取 WAV 文件，重采样到目标采样率，返回 (data, sr)"""
    sr, data = wavfile.read(filepath)
    # 转换为 float32，范围 [-1,1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)

    # 立体声转单声道（如果需要）
    if data.ndim == 2:
        data = np.mean(data, axis=1)

    # 重采样到目标采样率
    if sr != target_sr:
        number_of_samples = int(len(data) * target_sr / sr)
        data = signal.resample(data, number_of_samples)
        sr = target_sr

    return data, sr


def save_wav(filepath, data, sr):
    """保存为 WAV 文件，归一化后转为 int16"""
    # 确保数据在 [-1,1] 范围内
    max_val = np.max(np.abs(data))
    if max_val > 1.0:
        data = data / max_val
    # 转换为 int16
    data_int16 = (data * 32767).astype(np.int16)
    wavfile.write(filepath, sr, data_int16)


# ------------------------------------------------------------
# 图形用户界面
# ------------------------------------------------------------
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("变声器 - 最多3个音轨")
        self.root.geometry("1000x700")

        # 全局设置
        self.sr = 44100                     # 统一采样率
        self.tracks = [Track(self.sr) for _ in range(3)]   # 3个音轨
        self.current_track = 0              # 当前选中的音轨（0,1,2）

        # 临时变量：用于分割操作
        self.temp_audio_data = None         # 最近导入的音频数据
        self.temp_audio_path = None

        # 时间轴视图相关
        self.timeline_canvas = None
        self.rect_to_seg = {}               # 映射 canvas item id -> 片段对象引用
        self.selected_rect = None           # 当前选中的矩形 id
        self.drag_mode = None               # 'move', 'resize_left', 'resize_right'
        self.drag_start_x = 0               # 鼠标按下时的画布x坐标
        self.drag_start_time = 0.0          # 片段原始起始时间
        self.drag_orig_start = 0.0          # 片段原始起始时间（用于resize）
        self.drag_orig_end = 0.0            # 片段原始结束时间
        self.timeline_max_time = 30.0       # 时间轴显示的最大时间（秒），可动态调整
        self.timeline_scale_x = 1.0         # 像素/秒

        # 播放相关
        self.is_playing = False          # 是否正在播放
        self.play_thread = None          # 播放线程
        self.stop_playback = False       # 停止标志

        # 创建界面
        self.create_widgets()
        self.update_segment_list()
        self.draw_timeline()

    def create_widgets(self):
        # 顶部控制区
        top_frame = ttk.Frame(self.root)
        top_frame.pack(pady=5, fill=tk.X)

        # 音轨选择
        ttk.Label(top_frame, text="当前音轨：").pack(side=tk.LEFT, padx=5)
        # 播放控制
        ttk.Button(top_frame, text="播放", command=self.play_audio).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="停止", command=self.stop_audio).pack(side=tk.LEFT, padx=5)
        self.track_var = tk.IntVar(value=1)
        track_select = ttk.Combobox(top_frame, textvariable=self.track_var,
                                    values=[1, 2, 3], state="readonly", width=5)
        track_select.pack(side=tk.LEFT, padx=5)
        track_select.bind("<<ComboboxSelected>>", self.on_track_change)

        # 导入按钮
        ttk.Button(top_frame, text="导入音频文件", command=self.import_audio).pack(side=tk.LEFT, padx=10)
        ttk.Button(top_frame, text="分割导入的音频", command=self.split_audio_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="导出合成音频", command=self.export_audio).pack(side=tk.LEFT, padx=10)

        # 时间轴视图（可视化滑块）
        timeline_frame = ttk.LabelFrame(self.root, text="时间轴编辑器（拖动矩形调整位置/长度，右键菜单）")
        timeline_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 画布和滚动条
        canvas_frame = ttk.Frame(timeline_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.timeline_canvas = tk.Canvas(canvas_frame, bg='lightgray', height=150)
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.timeline_canvas.xview)
        self.timeline_canvas.configure(xscrollcommand=h_scroll.set)
        self.timeline_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # 绑定事件
        self.timeline_canvas.bind("<Button-1>", self.on_canvas_click)
        self.timeline_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.timeline_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.timeline_canvas.bind("<Button-3>", self.on_canvas_right_click)

        # 片段列表区域
        list_frame = ttk.LabelFrame(self.root, text="当前音轨的片段列表")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.segment_listbox = tk.Listbox(list_frame, height=10)
        self.segment_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.segment_listbox.bind('<<ListboxSelect>>', self.on_segment_select)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.segment_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.segment_listbox.config(yscrollcommand=scrollbar.set)

        # 片段编辑区
        edit_frame = ttk.LabelFrame(self.root, text="编辑选中片段")
        edit_frame.pack(fill=tk.X, padx=10, pady=5)

        # 起始时间
        ttk.Label(edit_frame, text="起始时间 (秒):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.start_entry = ttk.Entry(edit_frame, width=10)
        self.start_entry.grid(row=0, column=1, padx=5, pady=2)

        # 结束时间（只读显示）
        # 在 edit_frame 中，原结束时间部分替换为：
        ttk.Label(edit_frame, text="结束时间 (秒):").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.end_entry = ttk.Entry(edit_frame, width=10)
        self.end_entry.grid(row=0, column=3, padx=5, pady=2)

        # 变调半音数
        ttk.Label(edit_frame, text="变调 (半音):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.semitones_entry = ttk.Entry(edit_frame, width=10)
        self.semitones_entry.grid(row=1, column=1, padx=5, pady=2)

        # 频率范围
        ttk.Label(edit_frame, text="变调频率范围 (Hz, 可选):").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.band_low_entry = ttk.Entry(edit_frame, width=8)
        self.band_low_entry.grid(row=1, column=3, padx=2, pady=2)
        ttk.Label(edit_frame, text="-").grid(row=1, column=4)
        self.band_high_entry = ttk.Entry(edit_frame, width=8)
        self.band_high_entry.grid(row=1, column=5, padx=2, pady=2)

        # 按钮
        btn_frame = ttk.Frame(edit_frame)
        btn_frame.grid(row=2, column=0, columnspan=6, pady=5)
        ttk.Button(btn_frame, text="更新片段", command=self.update_segment).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="删除片段", command=self.delete_segment).pack(side=tk.LEFT, padx=5)

        # 状态栏
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_track_change(self, event=None):
        self.current_track = self.track_var.get() - 1
        self.update_segment_list()
        self.draw_timeline()

    def update_segment_list(self):
        self.segment_listbox.delete(0, tk.END)
        track = self.tracks[self.current_track]
        for i, seg in enumerate(track.segments):
            info = f"{i}: {seg['start']:.2f}s -> {seg['end']:.2f}s  | 变调: {seg['semitones']} 半音"
            if seg['band']:
                info += f"  | 频带: {seg['band'][0]}-{seg['band'][1]} Hz"
            self.segment_listbox.insert(tk.END, info)
        # 清空编辑区
        self.start_entry.delete(0, tk.END)
        self.end_entry.delete(0, tk.END)
        self.semitones_entry.delete(0, tk.END)
        self.band_low_entry.delete(0, tk.END)
        self.band_high_entry.delete(0, tk.END)

    def draw_timeline(self):
        """绘制当前音轨的时间轴视图"""
        if not self.timeline_canvas:
            return
        self.timeline_canvas.delete("all")
        track = self.tracks[self.current_track]
        if not track.segments:
            self.timeline_canvas.config(scrollregion=(0, 0, 10, 150))
            return

        # 计算时间轴范围（最大结束时间 + 1秒余量）
        max_end = max(seg['end'] for seg in track.segments)
        self.timeline_max_time = max(max_end + 1, 10.0)
        canvas_width = self.timeline_canvas.winfo_width()
        if canvas_width <= 1:
            canvas_width = 800  # 初始宽度
        self.timeline_scale_x = canvas_width / self.timeline_max_time

        # 绘制时间刻度（简单）
        for t in range(0, int(self.timeline_max_time) + 1, 1):
            x = t * self.timeline_scale_x
            self.timeline_canvas.create_line(x, 0, x, 150, fill='gray')
            self.timeline_canvas.create_text(x, 145, text=str(t), anchor=tk.N, font=('Arial', 8))

        # 绘制每个片段矩形
        self.rect_to_seg.clear()
        for seg in track.segments:
            x1 = seg['start'] * self.timeline_scale_x
            x2 = seg['end'] * self.timeline_scale_x
            # 颜色根据变调程度变化
            color = '#%02x%02x%02x' % (100 + int(abs(seg['semitones'])*5), 200, 150)
            rect_id = self.timeline_canvas.create_rectangle(x1, 20, x2, 130, fill=color, outline='black')
            self.timeline_canvas.create_text((x1 + x2)/2, 75, text=f"{seg['semitones']:+}半音", font=('Arial', 9))
            self.rect_to_seg[rect_id] = seg

        # 设置滚动区域
        self.timeline_canvas.config(scrollregion=(0, 0, self.timeline_max_time * self.timeline_scale_x, 150))
    
    def play_audio(self):
        """播放当前所有音轨的混合音频（在后台线程中）"""
        if self.is_playing:
            messagebox.showinfo("提示", "正在播放中，请先停止")
            return
        # 获取混合音频
        try:
            self.status_var.set("正在合成音频，请稍候...")
            self.root.update()
            mixed = mix_tracks(self.tracks, self.sr)
            if len(mixed) == 0:
                messagebox.showwarning("警告", "没有音频片段可播放")
                return
            self.status_var.set("准备播放...")
            self.root.update()
            
            # 在后台线程播放
            self.is_playing = True
            self.stop_playback = False
            self.play_thread = threading.Thread(target=self._play_audio_thread, args=(mixed,))
            self.play_thread.daemon = True
            self.play_thread.start()
        except Exception as e:
            self.status_var.set("播放失败")
            messagebox.showerror("错误", f"播放失败：{str(e)}")
            self.is_playing = False

    def _play_audio_thread(self, audio_data):
        """播放线程，支持停止"""
        try:
            # 将音频数据转换为 float32 并归一化（sounddevice 需要 float32 范围 -1~1）
            audio_data = audio_data.astype(np.float32)
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data = audio_data / max_val
            # 播放，阻塞直到播放结束或用户停止
            sd.play(audio_data, self.sr)
            # 轮询检查停止标志
            while sd.get_stream().active and not self.stop_playback:
                time.sleep(0.1)
            if self.stop_playback:
                sd.stop()
            self.status_var.set("播放完成")
        except Exception as e:
            self.status_var.set(f"播放出错: {str(e)}")
        finally:
            self.is_playing = False
            self.stop_playback = False

    def stop_audio(self):
        """停止播放"""
        if self.is_playing:
            self.stop_playback = True
            self.status_var.set("停止播放")

    def get_segment_from_coords(self, x, y):
        """根据画布坐标获取片段对象"""
        items = self.timeline_canvas.find_overlapping(x-2, y-2, x+2, y+2)
        for item in items:
            if item in self.rect_to_seg:
                return item, self.rect_to_seg[item]
        return None, None

    def on_segment_select(self, event=None):
        selection = self.segment_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        track = self.tracks[self.current_track]
        seg = track.segments[idx]
        self.start_entry.delete(0, tk.END)
        self.start_entry.insert(0, f"{seg['start']:.3f}")
        self.end_entry.delete(0, tk.END)
        self.end_entry.insert(0, f"{seg['end']:.3f}")
        self.semitones_entry.delete(0, tk.END)
        self.semitones_entry.insert(0, f"{seg['semitones']:.1f}")
        self.band_low_entry.delete(0, tk.END)
        self.band_high_entry.delete(0, tk.END)
        if seg['band']:
            self.band_low_entry.insert(0, f"{seg['band'][0]:.0f}")
            self.band_high_entry.insert(0, f"{seg['band'][1]:.0f}")

    def on_canvas_click(self, event):
        """鼠标左键按下：选择片段，并判断拖动模式"""
        item, seg = self.get_segment_from_coords(event.x, event.y)
        if item is None:
            self.selected_rect = None
            return
        self.selected_rect = item
        # 获取矩形坐标
        coords = self.timeline_canvas.coords(item)
        x1, y1, x2, y2 = coords
        # 判断鼠标在矩形中的位置
        if abs(event.x - x1) < 5:
            self.drag_mode = 'resize_left'
        elif abs(event.x - x2) < 5:
            self.drag_mode = 'resize_right'
        else:
            self.drag_mode = 'move'
        self.drag_start_x = event.x
        self.drag_start_time = seg['start']
        self.drag_orig_start = seg['start']
        self.drag_orig_end = seg['end']

    def on_canvas_drag(self, event):
        """鼠标拖拽：实时更新矩形位置/大小"""
        if self.selected_rect is None or self.drag_mode is None:
            return
        seg = self.rect_to_seg.get(self.selected_rect)
        if seg is None:
            return
        dx = (event.x - self.drag_start_x) / self.timeline_scale_x
        if self.drag_mode == 'move':
            new_start = self.drag_start_time + dx
            new_end = new_start + (seg['end'] - seg['start'])
            # 限制边界（不能小于0）
            if new_start < 0:
                new_start = 0
                new_end = seg['end'] - seg['start']
            # 临时更新画布上的矩形
            x1 = new_start * self.timeline_scale_x
            x2 = new_end * self.timeline_scale_x
            self.timeline_canvas.coords(self.selected_rect, x1, 20, x2, 130)
        elif self.drag_mode == 'resize_left':
            new_start = self.drag_start_time + dx
            if new_start >= seg['end'] - 0.01:
                new_start = seg['end'] - 0.01
            if new_start < 0:
                new_start = 0
            x1 = new_start * self.timeline_scale_x
            x2 = seg['end'] * self.timeline_scale_x
            self.timeline_canvas.coords(self.selected_rect, x1, 20, x2, 130)
        elif self.drag_mode == 'resize_right':
            new_end = self.drag_orig_end + dx
            if new_end <= seg['start'] + 0.01:
                new_end = seg['start'] + 0.01
            x1 = seg['start'] * self.timeline_scale_x
            x2 = new_end * self.timeline_scale_x
            self.timeline_canvas.coords(self.selected_rect, x1, 20, x2, 130)

    def on_canvas_release(self, event):
        """鼠标释放：应用更改（检查重叠并更新数据结构）"""
        if self.selected_rect is None or self.drag_mode is None:
            self.drag_mode = None
            return
        seg = self.rect_to_seg.get(self.selected_rect)
        if seg is None:
            self.drag_mode = None
            return
        # 获取最终矩形坐标
        coords = self.timeline_canvas.coords(self.selected_rect)
        x1, y1, x2, y2 = coords
        new_start = x1 / self.timeline_scale_x
        new_end = x2 / self.timeline_scale_x
        # 如果起始或结束发生变化，则尝试更新
        if new_start != seg['start'] or new_end != seg['end']:
            track = self.tracks[self.current_track]
            # 删除原片段
            idx = track.get_segment_index(seg)
            if idx == -1:
                self.drag_mode = None
                return
            track.remove_segment(idx)
            # 尝试添加新片段
            try:
                track.add_segment(seg['data'], new_start, new_end, seg['semitones'], seg['band'])
                # 更新成功
                self.update_segment_list()
                self.draw_timeline()
                self.status_var.set("片段位置/长度已更新")
            except ValueError as e:
                # 重叠，恢复原片段
                track.add_segment(seg['data'], seg['start'], seg['end'], seg['semitones'], seg['band'])
                messagebox.showwarning("重叠", "无法移动/调整，与其它片段时间重叠")
                self.draw_timeline()  # 恢复显示
        self.drag_mode = None

    def on_canvas_right_click(self, event):
        """右键菜单：复制/删除/分割片段"""
        item, seg = self.get_segment_from_coords(event.x, event.y)
        if item is None:
            return
        # 获取鼠标在画布上的实际坐标（考虑滚动）
        canvas_x = self.timeline_canvas.canvasx(event.x)
        split_time = canvas_x / self.timeline_scale_x
        # 弹出菜单
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label="复制", command=lambda: self.copy_segment_from_rect(item))
        menu.add_command(label="删除", command=lambda: self.delete_segment_from_rect(item))
        # 判断是否可分割：时间点在片段内部且不靠近边界（至少0.01秒）
        if seg['start'] < split_time < seg['end'] and (split_time - seg['start']) > 0.01 and (seg['end'] - split_time) > 0.01:
            menu.add_command(label="分割", command=lambda: self.split_segment_at_rect(item, split_time))
        else:
            # 不可分割时显示灰色选项
            menu.add_command(label="分割 (无效位置)", state=tk.DISABLED)
        menu.post(event.x_root, event.y_root)

    def copy_segment_from_rect(self, rect_id):
        seg = self.rect_to_seg.get(rect_id)
        if seg is None:
            return
        track = self.tracks[self.current_track]
        idx = track.get_segment_index(seg)
        if idx == -1:
            return
        try:
            track.copy_segment(idx, offset=0.1)
            self.update_segment_list()
            self.draw_timeline()
            self.status_var.set("片段已复制")
        except Exception as e:
            messagebox.showerror("错误", f"复制失败：{str(e)}")

    def delete_segment_from_rect(self, rect_id):
        seg = self.rect_to_seg.get(rect_id)
        if seg is None:
            return
        track = self.tracks[self.current_track]
        idx = track.get_segment_index(seg)
        if idx == -1:
            return
        track.remove_segment(idx)
        self.update_segment_list()
        self.draw_timeline()
        self.status_var.set("片段已删除")

    def split_segment_at_rect(self, rect_id, split_time):
        """分割指定矩形对应的片段"""
        seg = self.rect_to_seg.get(rect_id)
        if seg is None:
            return
        track = self.tracks[self.current_track]
        idx = track.get_segment_index(seg)
        if idx == -1:
            return

        sr = self.sr
        split_sample = int((split_time - seg['start']) * sr)
        if split_sample <= 0 or split_sample >= len(seg['data']):
            return  # 边界情况，安全起见

        # 分割数据（深拷贝）
        data_left = seg['data'][:split_sample].copy()
        data_right = seg['data'][split_sample:].copy()

        # 保存原片段信息以便恢复
        old_seg = seg.copy()
        # 删除原片段
        track.remove_segment(idx)

        success = True
        try:
            # 添加左片段（起始时间不变，结束时间为分割点）
            track.add_segment(data_left, seg['start'], split_time, seg['semitones'], seg['band'])
            # 添加右片段（起始时间为分割点，结束时间为原结束时间）
            track.add_segment(data_right, split_time, seg['end'], seg['semitones'], seg['band'])
        except ValueError as e:
            success = False
            # 恢复原片段
            track.add_segment(old_seg['data'], old_seg['start'], old_seg['end'], old_seg['semitones'], old_seg['band'])

        if success:
            self.update_segment_list()
            self.draw_timeline()
            self.status_var.set("片段已分割")
        else:
            messagebox.showerror("错误", "分割失败：与其它片段时间重叠")

    def import_audio(self):
        """导入音频文件（作为一个整体片段）"""
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not filepath:
            return
        try:
            data, sr = load_wav(filepath, self.sr)
            duration = len(data) / self.sr
            # 询问起始时间
            start_str = simpledialog.askstring("起始时间", f"输入该片段在音轨上的起始时间（秒）\n音频时长 {duration:.2f} 秒：", initialvalue="0")
            if start_str is None:
                return
            try:
                start = float(start_str)
            except ValueError:
                messagebox.showerror("输入错误", "起始时间必须是有效的数字")
                return
            end = start + duration
            # 尝试添加到当前音轨
            track = self.tracks[self.current_track]
            track.add_segment(data, start, end, semitones=0, band=None)
            self.update_segment_list()
            self.draw_timeline()
            self.status_var.set(f"已导入：{os.path.basename(filepath)}，时长 {duration:.2f} 秒，起始时间 {start:.2f}s")
        except Exception as e:
            messagebox.showerror("错误", f"导入失败：{str(e)}")

    def split_audio_dialog(self):
        """分割音频：先导入一个音频文件，然后按分割点生成多个片段"""
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if not filepath:
            return
        try:
            data, sr = load_wav(filepath, self.sr)
            duration = len(data) / self.sr
            # 输入分割点
            split_str = simpledialog.askstring("分割点", f"请输入分割时间点（秒），用逗号分隔。例如：10,20\n音频总时长 {duration:.2f} 秒")
            if split_str is None:
                return
            splits = [0] + [float(x.strip()) for x in split_str.split(",") if x.strip()] + [duration]
            splits.sort()
            # 生成片段
            track = self.tracks[self.current_track]
            start_time = 0  # 默认每个片段在音轨上的起始时间从当前时间轴末尾开始连续放置？简单起见，让用户逐个输入起始时间
            for i in range(len(splits) - 1):
                seg_start = splits[i]
                seg_end = splits[i+1]
                if seg_end - seg_start <= 0:
                    continue
                seg_data = data[int(seg_start * self.sr):int(seg_end * self.sr)]
                # 询问该片段在音轨上的起始时间
                start_str = simpledialog.askstring(f"片段 {i+1} 起始时间",
                                                      f"片段 {i+1} 时间范围 {seg_start:.2f}-{seg_end:.2f} 秒\n请输入它在音轨上的起始时间（秒）：",
                                                      initialvalue=str(start_time))
                if start_str is None:
                    return
                start_pos = float(start_str)
                end_pos = start_pos + (seg_end - seg_start)
                track.add_segment(seg_data, start_pos, end_pos, semitones=0, band=None)
                start_time = end_pos  # 下一个片段默认接在后面
            self.update_segment_list()
            self.draw_timeline()
            self.status_var.set(f"已分割导入：{os.path.basename(filepath)}，共 {len(splits)-1} 个片段")
        except Exception as e:
            messagebox.showerror("错误", f"分割失败：{str(e)}")

    def delete_segment(self):
        """删除当前选中的片段（从列表）"""
        selection = self.segment_listbox.curselection()
        if not selection:
            messagebox.showinfo("提示", "请先选择一个片段")
            return
        idx = selection[0]
        track = self.tracks[self.current_track]
        track.remove_segment(idx)
        self.update_segment_list()
        self.draw_timeline()
        self.status_var.set("已删除片段")

    def update_segment(self):
        """更新当前选中的片段参数（支持修改起始时间、结束时间、变调、频带）"""
        selection = self.segment_listbox.curselection()
        if not selection:
            messagebox.showinfo("提示", "请先选择一个片段")
            return
        idx = selection[0]
        track = self.tracks[self.current_track]
        seg = track.segments[idx]

        try:
            new_start = float(self.start_entry.get())
            new_end = float(self.end_entry.get())
            if new_end <= new_start:
                raise ValueError("结束时间必须大于起始时间")
            semitones = float(self.semitones_entry.get()) if self.semitones_entry.get() else 0.0
            low_str = self.band_low_entry.get().strip()
            high_str = self.band_high_entry.get().strip()
            band = None
            if low_str and high_str:
                band = (float(low_str), float(high_str))
        except ValueError as e:
            messagebox.showerror("输入错误", f"无效的数字: {e}")
            return

        # 计算新时长和原始时长，必要时调整音频数据
        orig_duration = len(seg['data']) / self.sr
        new_duration = new_end - new_start
        new_data = seg['data'].copy()
        if abs(new_duration - orig_duration) > 1e-6:
            new_samples = int(new_duration * self.sr)
            if new_samples > len(new_data):
                new_data = np.pad(new_data, (0, new_samples - len(new_data)), 'constant')
            else:
                new_data = new_data[:new_samples]

        # 尝试替换片段
        old_seg = seg.copy()
        track.remove_segment(idx)
        try:
            track.add_segment(new_data, new_start, new_end, semitones, band)
            self.update_segment_list()      # 刷新列表
            self.draw_timeline()            # 刷新时间轴

            # 重新选中刚更新的片段（根据新的起止时间定位）
            for i, s in enumerate(track.segments):
                if abs(s['start'] - new_start) < 1e-6 and abs(s['end'] - new_end) < 1e-6:
                    self.segment_listbox.selection_clear(0, tk.END)
                    self.segment_listbox.selection_set(i)
                    self.segment_listbox.activate(i)
                    self.segment_listbox.see(i)
                    # 主动调用 on_segment_select 填充编辑区（可选，但已有绑定会自动触发）
                    break
            self.status_var.set("片段已更新")
        except Exception as e:
            # 恢复原片段
            track.add_segment(old_seg['data'], old_seg['start'], old_seg['end'],
                            old_seg['semitones'], old_seg['band'])
            messagebox.showerror("错误", f"更新失败: {e}")

    def export_audio(self):
        """导出所有音轨的混合音频"""
        output_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if not output_path:
            return
        try:
            # 在后台线程中执行，避免界面卡死
            self.status_var.set("正在合成音频，请稍候...")
            self.root.update()
            mixed = mix_tracks(self.tracks, self.sr)
            if len(mixed) == 0:
                messagebox.showwarning("警告", "没有音频片段可导出")
                return
            save_wav(output_path, mixed, self.sr)
            self.status_var.set(f"导出成功：{output_path}")
            messagebox.showinfo("完成", f"音频已导出到：{output_path}")
        except Exception as e:
            self.status_var.set("导出失败")
            messagebox.showerror("错误", f"导出失败：{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()