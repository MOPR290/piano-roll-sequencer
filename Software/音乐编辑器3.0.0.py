import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import math
import copy
# 合成走 multiprocessing.Pool，Windows 下用 spawn 拉子进程，每个子进程都会重跑本文件顶层，
# 于是每个子进程都要再 import 一次 pygame，控制台就被 "Hello from the pygame community" 刷屏。
# 这个变量必须在 import pygame 之前设；子进程继承父进程的环境，设一次就全部静音。
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
# 顺带把 pygame 内部 pkgdata 触发的 pkg_resources 弃用警告也压掉（同样是每个子进程一条）
import warnings
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')
import pygame
import pygame.mixer as mixer
from array import array
import threading
import time
import queue
import numpy as np
from sound_generator import (
    SimpleAudioSegment,
    generate_audio_from_events,
    shutdown_render_pool,
    resolve_workers,
)
# 在文件开头添加必要的导入
# 替换原来的simpleaudio导入

# Alt 在鼠标事件里的 state 位：Windows 是 0x20000，X11 是 Mod1=0x0008。
# 不能只在 Windows 上判 0x0008——那个位会被 NumLock 点亮，开着 NumLock 时每次点击都带它，
# 于是“点音符”全变成“拉框选”（2026-11-21 实测：NumLock 开着时裸点击 state=0x8）。
ALT_MASK = 0x20000 if os.name == 'nt' else 0x0008


class ProgressDialog(tk.Toplevel):
    """合成 / 导出时的模态进度条。

    只有主线程可以碰 Tk，所以工作线程只往队列里投递 (阶段, 百分比)，
    由主线程的 after 轮询取出来刷新这里的控件。
    """

    def __init__(self, master, title="正在处理…", subtitle=""):
        super().__init__(master)
        self.title(title)
        self.resizable(False, False)
        self.transient(master)
        self.protocol("WM_DELETE_WINDOW", lambda: None)   # 处理期间不许关
        self._job = None
        self._t0 = time.time()

        body = ttk.Frame(self, padding=(18, 16, 18, 12))
        body.pack(fill=tk.BOTH, expand=True)

        self.stage_var = tk.StringVar(value=subtitle or "准备中…")
        ttk.Label(body, textvariable=self.stage_var, width=42,
                  anchor=tk.W).pack(fill=tk.X)

        self.pb = ttk.Progressbar(body, mode="determinate", maximum=100.0,
                                  length=340, value=0)
        self.pb.pack(fill=tk.X, pady=(10, 6))

        foot = ttk.Frame(body)
        foot.pack(fill=tk.X)
        self.pct_var = tk.StringVar(value="0%")
        self.time_var = tk.StringVar(value="0.0 s")
        ttk.Label(foot, textvariable=self.pct_var, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Label(foot, textvariable=self.time_var, anchor=tk.E,
                  foreground="#666666").pack(side=tk.RIGHT)

        self._center(master)
        try:
            self.grab_set()          # 模态：处理期间点不到主窗口
        except tk.TclError:
            pass                     # 主窗口还没显示出来时会失败，忽略即可
        self._tick()

    def _center(self, master):
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        try:
            x = master.winfo_rootx() + (master.winfo_width() - w) // 2
            y = master.winfo_rooty() + (master.winfo_height() - h) // 3
        except tk.TclError:
            x = y = 200
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _tick(self):
        """每 0.1 s 刷新一次已用时间（长曲子也能看出没卡死）"""
        try:
            self.time_var.set(f"{time.time() - self._t0:.1f} s")
            self._job = self.after(100, self._tick)
        except tk.TclError:
            self._job = None

    def set(self, stage, percent):
        try:
            if stage:
                self.stage_var.set(stage)
            percent = max(0.0, min(100.0, float(percent)))
            self.pb.configure(value=percent)
            self.pct_var.set(f"{percent:.0f}%")
        except tk.TclError:
            pass

    def close(self):
        if self._job is not None:
            try:
                self.after_cancel(self._job)
            except tk.TclError:
                pass
            self._job = None
        try:
            self.grab_release()
            self.destroy()
        except tk.TclError:
            pass


class PianoRollEditor:
    _pygame_initialized = False
    def __init__(self, root):
        self.root = root
        self._busy = False       # 正在后台合成/导出时为 True
        # 音频合成进程数（设置里可改；1 = 只用主进程串行合成）
        self.max_synth_workers = max(1, min(16, os.cpu_count() or 1))
        self.synth_workers = 4
        self.load_settings()
        self.root.title("钢琴卷帘音序器 3.0")
        self.root.geometry("1200x800")
        # 改为：
        self.toolbar = None  # 将在 create_main_interface 中初始化
        # 设置时间单位（十六分音符）
        self.time_unit = tk.DoubleVar(value=1.0)  # 1.0 表示十六分音符

        # 横向缩放
        self.pixels_per_unit = 40.0  # 每单位像素数（初始缩放）
        # 添加纵向缩放
        self.pixels_per_pitch = 20.0  # 每个音高的像素高度
        
        # 添加节拍设置（默认4/4拍）
        self.time_signature = [4, 4]  # [分子, 分母]
        self.units_per_measure = 16  # 每小节单位数（十六分音符）

        # ============ 3.0.0 新增：音轨 ============
        # 每条音轨：名称、默认音色、音量、声像、静音、独奏、颜色
        # 音符用 'track' 字段索引到这里；文件里缺少 track 列的旧音符一律归 0 轨
        self.tracks = [self._make_track("轨道 1", "triangle", "#99FF99")]
        self.current_track = 0          # 新音符落在哪条轨

        # ============ 3.0.0 新增：网格吸附 ============
        # 值是该档位对应的时值（十六分音符为单位），0 表示不吸附
        self.snap_options = {
            "1/4 拍": 4.0,     # 四分音符
            "1/8 拍": 2.0,     # 八分音符
            "1/16 拍": 1.0,    # 十六分音符
            "1/32 拍": 0.5,
            "三连音": 4.0 / 3.0,
            "关闭吸附": 0.0,
        }
        self.snap_var = tk.StringVar(value="1/16 拍")

        # ============ 3.0.0 新增：循环播放 / 剪贴板 / 光标位置 ============
        self.loop_enabled = tk.BooleanVar(value=False)
        self.clipboard_notes = []       # 复制出来的音符（已归一到 0 起点）
        self.cursor_unit = 0.0          # 最近一次点击的时间位置，粘贴落点
        self.metronome_enabled = tk.BooleanVar(value=False)
        # 结束小节是否由用户手动指定；False 时自动跟随乐曲实际长度
        self.measure_range_manual = False


        # 添加播放相关变量
        self.is_playing = False
        self.play_head = None
        self.play_start_time = 0
        self.current_play_time = 0
        self.play_update_id = None
        # 采样解码缓存：(文件, 振幅) -> 波形数组，密集采样段落避免重复解码
        self._sample_cache = {}
        # 播放线程只置这个标志，收尾由主线程定时器做（Tk 不是线程安全的）
        self._playback_ended = False
        self.music_channel = None
        self.bpm = 90  # 默认BPM
        self.play_obj = None  # 添加播放对象跟踪

        self.create_menu()
        self.create_main_interface()  # 先创建主界面（包含工具栏）
        self.create_playback_controls()  # 然后在现有工具栏上添加播放控件
        
        # 存储音序数据
        self.sequence = []  # 存储所有音符对象
        self.selected_note = None  # 当前选中的音符
        
        # 拖拽状态
        self.dragging = False
        self.drag_type = None  # 'start', 'duration', 'pitch', 'multi'
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_start_value = 0
        # 拖动真正改动音符时才写撤销栈（点一下只选中不该产生快照）
        self._pending_snapshot = False
        # 音符配色缓存：(音色, 力度档, 静音) -> (填充色, 音名文字色)
        self._note_color_cache = {}
        # 画布 <Configure> 防抖：连续缩放窗口时只重绘一次
        self._configure_job = None
        
        # 多选状态：框选由“按住 Alt 拖动”触发，不设常驻开关，因此没有可残留的模式状态
        self.multi_selected_notes = []  # 存储多选的音符
        self.selection_rect_id = None  # 选择框ID
        # 边框高亮的上次状态：{rect_id: 2=多选/1=单选}。None 表示需全量重刷
        self._hl_state = None
        # 框选拖动的节流：鼠标事件远多于屏幕刷新，攒到下一帧再动一次选框
        self._rect_job = None
        self._rect_pending = None
        
        # 设置最大小节数 (64小节)
        self.max_measures = 256
        
        # 在画布初始化中设置更大的滚动区域
        self.canvas.config(scrollregion=(0, 0, self.max_measures * 16 * self.pixels_per_unit, 96 * 20))


        # 绑定事件 - 添加右键事件绑定
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_right_click)  # 添加右键事件绑定
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        self.root.bind("<Control-c>", self.copy_to_clipboard)   # 3.0.0：改为复制到剪贴板
        self.root.bind("<Control-x>", self.cut_to_clipboard)
        self.root.bind("<Control-v>", self.paste_from_clipboard)
        self.root.bind("<Control-d>", self.duplicate_selection)  # 原来的“就地复制”挪到这里
        # 3.0.0：框选不再用 Alt 按键事件做状态开关（松开事件丢失就会退不出来），
        # 改为按下鼠标时直接读 event.state 的 Alt 位，见 on_canvas_click
        # 3.0.0 新增快捷键：空格播放/停止、Delete 删除
        self.root.bind("<space>", self.toggle_playback)
        self.root.bind("<Delete>", self.delete_selected_notes)
        self.root.bind("<BackSpace>", self.delete_selected_notes)
        
        # 撤销/重做栈
        self.undo_stack = []
        self.redo_stack = []
        
        # 当前文件
        self.current_file = None
        self.annotations = []  # 存储批注数据
        self.annotations_visible = True  # 批注是否可见

        # ============ 3.1 新增：效果器块 ============
        # 一个块 = 画布上一块半透明色块；落在块内的音符自动套用它启用的效果。
        # {'id', 'start', 'end', 'pitch_low', 'pitch_high', 'color',
        #  'timbres': None 表示块内所有音色，否则是允许列表,
        #  'fx': {'lowpass_1st': {'on': bool},
        #         'sustain':      {'on': bool, 'value': 0.2},
        #         'reverb':       {'on': bool, 'ir': 路径, 'mix': 0.3}}}
        self.effect_blocks = []
        self.selected_block = None
        self.effect_blocks_visible = True
        self._fx_next_id = 1
        self._block_press = None    # 按下色块时的拖拽现场

        
        # 绘制初始网格
        # 在菜单中添加新功能
        self.create_menu()
        self.draw_grid()
        # 添加音频样本管理
        self.audio_samples = []  # 存储音频样本信息
        self.audio_samples_dir = "audio_samples"  # 样本存储目录
        # 项目根目录下已有 audio_samples 时优先用它，避免从 Software/ 启动时另建空目录
        _project_samples = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio_samples")
        if os.path.isdir(_project_samples):
            self.audio_samples_dir = _project_samples
        self.create_audio_samples_dir()  # 创建样本目录
        # 3.1：混响预置 IR 目录（环绕声/浴室/大堂），与 audio_samples 平级
        self.ir_preset_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "impulse_responses")
        # 初始化pygame mixer
        if not PianoRollEditor._pygame_initialized:
            pygame.init()
            mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            # 3.0.0：pygame 默认只有 8 个声道，采样事件一多就会抢占并切断整曲；
            # 这里放宽到 32 个，并预留 0 号声道给整曲混音，采样只用 1 号及以后
            mixer.set_num_channels(32)
            mixer.set_reserved(1)
            PianoRollEditor._pygame_initialized = True
    
    # ------------------------------------------------------------ 音轨
    TRACK_COLORS = ["#99FF99", "#99CCFF", "#FFCC99", "#FF99CC",
                    "#F9F49F", "#00FFE0", "#CBA000", "#CB7942"]

    # 3.0.0：每种音色一个专属色，音符块填充色按音色取（音轨色退为边框）
    WAVETYPE_COLORS = {
        "custom_harmonics":      "#C0CA33",  # 黄绿
        "harmonics":             "#FFA726",  # 橙
        "harmonica":             "#66BB6A",  # 绿
        "violin":                "#EF5350",  # 红
        "string":                "#8D6E63",  # 棕
        "ocarina":               "#26C6DA",  # 青
        "organ":                 "#B39DDB",  # 淡紫
        "steel-tongue-drum":     "#7E57C2",  # 紫
        "sine":                  "#4C9AFF",  # 蓝
        "square":                "#FF7043",  # 橙红
        "triangle":              "#FFC400",  # 黄
        "sawtooth":              "#AB47BC",  # 品红紫
        "noise":                 "#9E9E9E",  # 灰
        "filtered_sq":           "#26A69A",  # 墨绿
        "filtered_noise":        "#78909C",  # 蓝灰
        "double_filtered_noise": "#546E7A",  # 深蓝灰
        "double_filtered_sq":    "#00897B",  # 深青绿
        "beat":                  "#EC407A",  # 玫红
        "empty":                 "#E0E0E0",  # 空白
        "audio_sample":          "#5C6BC0",  # 靛蓝
    }

    def wavetype_color(self, wavetype):
        """取音色对应的色块颜色，未知音色给中性灰"""
        return self.WAVETYPE_COLORS.get(wavetype, "#B0BEC5")

    def _make_track(self, name, wavetype="triangle", color=None):
        """构造一条音轨的默认属性"""
        return {
            'name': name,
            'wavetype': wavetype,
            'volume': 0.8,
            'pan': 0.0,
            'mute': False,
            'solo': False,
            'color': color or self.TRACK_COLORS[0],
        }

    def get_track(self, index):
        """取音轨，索引越界时归到 0 轨"""
        if 0 <= index < len(self.tracks):
            return self.tracks[index]
        return self.tracks[0]

    def track_color(self, index):
        return self.get_track(index).get('color', "#99FF99")

    def add_track(self):
        """新增一条音轨"""
        idx = len(self.tracks)
        self.tracks.append(self._make_track(
            f"轨道 {idx + 1}", "triangle", self.TRACK_COLORS[idx % len(self.TRACK_COLORS)]))
        self.current_track = idx
        self.refresh_track_panel()
        self.draw_notes()
        self.status_var.set(f"已添加 轨道 {idx + 1}")

    def remove_track(self, index=None):
        """删除音轨，该轨音符并入 0 轨（0 轨不可删）"""
        index = self.current_track if index is None else index
        if len(self.tracks) <= 1:
            self.status_var.set("至少要保留一条音轨")
            return
        if index == 0:
            self.status_var.set("轨道 1 不能删除")
            return
        self.save_state()
        for note in self.sequence:
            if note.get('track', 0) == index:
                note['track'] = 0
            elif note.get('track', 0) > index:
                note['track'] -= 1
        del self.tracks[index]
        self.current_track = min(self.current_track, len(self.tracks) - 1)
        self.refresh_track_panel()
        self.draw_notes()
        self.status_var.set("已删除音轨，其音符已并入轨道 1")

    def rename_track(self):
        """重命名当前音轨"""
        from tkinter import simpledialog
        track = self.get_track(self.current_track)
        name = simpledialog.askstring("重命名音轨", "轨道名称:",
                                      initialvalue=track['name'], parent=self.root)
        if name:
            track['name'] = name.replace(";", "；").replace("\n", " ")
            self.refresh_track_panel()

    def rename_track_at(self, index):
        """双击轨道名时重命名指定轨道"""
        self.set_current_track(index)
        self.rename_track()

    def assign_selection_to_track(self, index):
        """把选中的音符移到指定音轨"""
        targets = self.multi_selected_notes or ([self.selected_note] if self.selected_note else [])
        if not targets:
            self.status_var.set("没有选中任何音符")
            return
        self.save_state()
        for note in targets:
            note['track'] = index
        self.draw_notes()
        self.status_var.set(f"{len(targets)} 个音符已移到 {self.get_track(index)['name']}")

    def apply_track_wavetype(self, index):
        """把轨道音色应用到该轨全部音符"""
        notes = [n for n in self.sequence if n.get('track', 0) == index]
        if not notes:
            self.status_var.set("该轨没有音符")
            return
        self.save_state()
        wavetype = self.get_track(index)['wavetype']
        for note in notes:
            note['wavetype'] = wavetype
        self.draw_notes()
        self.status_var.set(f"{len(notes)} 个音符已改为 {wavetype}")

    def create_audio_samples_dir(self):
        """创建音频样本目录"""
        if not os.path.exists(self.audio_samples_dir):
            os.makedirs(self.audio_samples_dir)
    
    def import_audio_sample(self, file_path):
        """导入音频样本文件"""
        try:
            # 复制文件到样本目录
            filename = os.path.basename(file_path)
            sample_name = os.path.splitext(filename)[0]
            dest_path = os.path.join(self.audio_samples_dir, filename)
            
            # 如果文件已存在，添加编号
            counter = 1
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(filename)
                dest_path = os.path.join(self.audio_samples_dir, f"{name}_{counter}{ext}")
                counter += 1
            
            import shutil
            shutil.copy2(file_path, dest_path)
            
            # 添加到样本列表
            sample_id = len(self.audio_samples) + 1
            sample_info = {
                'id': sample_id,
                'name': sample_name,
                'file_path': dest_path,
                'filename': os.path.basename(dest_path)
            }
            self.audio_samples.append(sample_info)
            
            return sample_info
        except Exception as e:
            messagebox.showerror("错误", f"导入音频样本失败: {str(e)}")
            return None
    
    def get_audio_sample_by_id(self, sample_id):
        """根据ID获取音频样本"""
        for sample in self.audio_samples:
            if sample['id'] == sample_id:
                return sample
        return None
    
    def load_audio_samples(self):
        """加载音频样本目录中的文件"""
        self.audio_samples = []
        if os.path.exists(self.audio_samples_dir):
            for filename in os.listdir(self.audio_samples_dir):
                if filename.lower().endswith(('.wav', '.mp3', '.ogg')):
                    sample_id = len(self.audio_samples) + 1
                    sample_name = os.path.splitext(filename)[0]
                    file_path = os.path.join(self.audio_samples_dir, filename)
                    
                    self.audio_samples.append({
                        'id': sample_id,
                        'name': sample_name,
                        'file_path': file_path,
                        'filename': filename
                    })

    # ------------------------------------------------------------ 3.0.0 编辑辅助
    def _text_has_focus(self):
        """焦点在输入框里时，快捷键让位给文本编辑"""
        try:
            w = self.root.focus_get()
        except Exception:
            return False
        return isinstance(w, (tk.Entry, ttk.Entry, tk.Text, ttk.Combobox))

    def snap_value(self):
        """当前吸附步长（十六分音符单位），0 表示不吸附"""
        return self.snap_options.get(self.snap_var.get(), 0.0)

    def snap_units(self, value, minimum=0.0):
        """把时间值吸附到当前网格"""
        step = self.snap_value()
        if step <= 0:
            return max(minimum, value)
        return max(minimum, round(value / step) * step)

    def toggle_playback(self, event=None):
        """空格：播放 / 停止 来回切换"""
        if self._text_has_focus():
            return
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback()
        return "break"

    def _selection_notes(self):
        """当前要操作的音符集合（多选优先）"""
        if self.multi_selected_notes:
            return list(self.multi_selected_notes)
        if self.selected_note:
            return [self.selected_note]
        return []

    def copy_to_clipboard(self, event=None):
        """Ctrl+C：复制选中音符到内部剪贴板（起点归一到 0）"""
        if self._text_has_focus():
            return
        notes = self._selection_notes()
        if not notes:
            self.status_var.set("没有选中任何音符")
            return "break"
        base = min(n['start'] for n in notes)
        self.clipboard_notes = []
        for note in notes:
            item = copy.deepcopy(note)
            item.pop('rect_id', None)
            item.pop('text_id', None)
            item['start'] = item['start'] - base
            self.clipboard_notes.append(item)
        self.status_var.set(f"已复制 {len(notes)} 个音符，Ctrl+V 粘贴到光标处")
        return "break"

    def cut_to_clipboard(self, event=None):
        """Ctrl+X：剪切"""
        if self._text_has_focus():
            return
        self.copy_to_clipboard()
        self.delete_selected_notes()
        return "break"

    def paste_from_clipboard(self, event=None):
        """Ctrl+V：把剪贴板内容粘贴到光标位置"""
        if self._text_has_focus():
            return
        if not self.clipboard_notes:
            self.status_var.set("剪贴板是空的")
            return "break"
        self.save_state()
        anchor = self.cursor_unit
        pasted = []
        for item in self.clipboard_notes:
            new_note = copy.deepcopy(item)
            new_note['start'] = anchor + item['start']
            new_note['track'] = self.current_track
            self.sequence.append(new_note)
            pasted.append(new_note)
        self.multi_selected_notes = pasted
        self.selected_note = pasted[-1]
        # 光标后移，连续粘贴能首尾相接
        block_len = max(n['start'] + n['duration'] for n in pasted) - anchor
        self.cursor_unit = anchor + block_len
        self.draw_notes()
        self.status_var.set(f"已粘贴 {len(pasted)} 个音符到 {anchor:.2f}")
        return "break"

    def paste_at(self, canvas_x):
        """粘贴到指定横坐标（右键菜单用）"""
        self.cursor_unit = self.snap_units(canvas_x / self.pixels_per_unit)
        self.paste_from_clipboard()

    def duplicate_selection(self, event=None):
        """Ctrl+D：原地复制（旧版 Ctrl+C 的行为）"""
        if self._text_has_focus():
            return
        notes = self._selection_notes()
        if not notes:
            return
        self.save_state()
        new_notes = []
        for note in notes:
            item = copy.deepcopy(note)
            item.pop('rect_id', None)
            item.pop('text_id', None)
            item['start'] = item['start'] + 4.0
            self.sequence.append(item)
            new_notes.append(item)
        self.multi_selected_notes = new_notes
        self.selected_note = new_notes[-1]
        self.draw_notes()
        self.status_var.set(f"已原地复制 {len(new_notes)} 个音符")

    def copy_note(self, note=None, offset=1.0):
        """复制音符并就地向右偏移 offset
        不传 note 时复制当前选区；传了 note 但它属于当前框选时，按整组复制
        （右键菜单的「复制音符」与 Ctrl+C 保持同一套选区语义）
        """
        if note is None:
            notes = self._selection_notes()
        elif self.multi_selected_notes and id(note) in {id(n) for n in self.multi_selected_notes}:
            notes = list(self.multi_selected_notes)
        else:
            notes = [note]
        if not notes:
            return None

        # 保存当前状态到撤销栈
        self.save_state()

        new_notes = []
        for src in notes:
            # 深拷贝选中的音符
            new_note = copy.deepcopy(src)
            new_note.pop('rect_id', None)
            new_note.pop('text_id', None)
            new_note['start'] += offset
            # 添加到音序
            self.sequence.append(new_note)
            new_notes.append(new_note)

        # 复制出来的这组成为当前选区（多个时保持多选，方便继续操作）
        self.multi_selected_notes = new_notes if len(new_notes) > 1 else []
        self.selected_note = new_notes[-1]

        # 重新绘制
        self.draw_notes()
        self.status_var.set(f"已复制 {len(new_notes)} 个音符")
        return new_notes[-1]

    def copy_selected_note(self, event=None):
        """复制选中的音符（兼容旧调用）"""
        return self.copy_note()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        # 文件菜单
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="新建", command=self.new_file)
        file_menu.add_command(label="打开", command=self.open_file)
        file_menu.add_command(label="保存", command=self.save_file)
        file_menu.add_command(label="另存为", command=self.save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="导出为WAV", command=self.export_to_wav)  # 新增导出功能
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=file_menu)
        
        # 节拍菜单
        meter_menu = tk.Menu(menubar, tearoff=0)
        meter_menu.add_command(label="4/4拍", command=lambda: self.set_time_signature(4, 4))
        meter_menu.add_command(label="3/4拍", command=lambda: self.set_time_signature(3, 4))
        meter_menu.add_command(label="6/8拍", command=lambda: self.set_time_signature(6, 8))
        menubar.add_cascade(label="节拍设置", menu=meter_menu)

        # 编辑菜单 - 修改批注相关选项
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="添加音符", command=self.add_note)
        edit_menu.add_command(label="删除音符", command=self.delete_selected_notes)
        edit_menu.add_separator()
        edit_menu.add_command(label="一键变调", command=self.transpose_all)
        edit_menu.add_command(label="多选变调", command=self.transpose_selected)
        edit_menu.add_separator()
        # 3.0.0：不再保留“多选模式”开关（开启后无法退出）；框选统一用按住 Alt 拖动
        edit_menu.add_command(label="清除选择", command=self.clear_multi_selection)
        edit_menu.add_command(label="框选多个音符：按住 Alt 拖动", state=tk.DISABLED)
        edit_menu.add_separator()
        edit_menu.add_command(label="复制选中音符", command=self.copy_selected_note)
        edit_menu.add_command(label="移动选中音符", command=self.move_selected_notes_with_input)
        edit_menu.add_command(label="按时间段选中 / 复制粘贴…", command=self.segment_copy_paste)
        edit_menu.add_separator()
        # 3.0.0：剪贴板
        edit_menu.add_command(label="复制", command=self.copy_to_clipboard, accelerator="Ctrl+C")
        edit_menu.add_command(label="剪切", command=self.cut_to_clipboard, accelerator="Ctrl+X")
        edit_menu.add_command(label="粘贴到光标", command=self.paste_from_clipboard, accelerator="Ctrl+V")
        edit_menu.add_command(label="原地复制", command=self.duplicate_selection, accelerator="Ctrl+D")
        edit_menu.add_separator()
        edit_menu.add_command(label="批注", command=self.show_annotation_dialog)  # 修改为单一按钮
        edit_menu.add_separator()
        edit_menu.add_command(label="撤销", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="重做", command=self.redo, accelerator="Ctrl+Y")

        
        menubar.add_cascade(label="编辑", menu=edit_menu)
        # 视图菜单
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="放大", command=lambda: self.zoom(1.2))
        view_menu.add_command(label="缩小", command=lambda: self.zoom(0.8))
        view_menu.add_command(label="重置缩放", command=lambda: self.zoom(0))
        menubar.add_cascade(label="视图", menu=view_menu)

        # 3.0.0 新增：轨道菜单
        track_menu = tk.Menu(menubar, tearoff=0)
        track_menu.add_command(label="添加音轨", command=self.add_track)
        track_menu.add_command(label="删除当前音轨", command=self.remove_track)
        track_menu.add_command(label="重命名当前音轨", command=self.rename_track)
        track_menu.add_separator()
        track_menu.add_command(label="把选中音符移到当前音轨",
                               command=lambda: self.assign_selection_to_track(self.current_track))
        track_menu.add_command(label="把轨道音色应用到本轨全部音符",
                               command=lambda: self.apply_track_wavetype(self.current_track))
        menubar.add_cascade(label="轨道", menu=track_menu)

        # 宏操作：一次改动一整批音符的批量操作
        macro_menu = tk.Menu(menubar, tearoff=0)
        macro_menu.add_command(label="设定同一音色幅值…", command=self.set_timbre_amplitude)
        macro_menu.add_command(label="统一设定振幅（仅选中音符）…",
                               command=self.unify_amplitude_for_selected)
        macro_menu.add_separator()

        # 同步移动：整批音符平移同样距离，使最早的一个从指定 16 分音符开始
        move_menu = tk.Menu(macro_menu, tearoff=0)
        move_menu.add_command(label="移动所有音符…",
                              command=lambda: self.sync_move_notes('all'))
        move_menu.add_command(label="仅移动选中音符…",
                              command=lambda: self.sync_move_notes('sel'))
        macro_menu.add_cascade(label="同步移动音符", menu=move_menu)
        menubar.add_cascade(label="宏操作", menu=macro_menu)

        # 3.1 新增：效果器块
        fx_menu = tk.Menu(menubar, tearoff=0)
        fx_menu.add_command(label="新建效果器块（框住选中音符）", command=self.add_effect_block)
        fx_menu.add_command(label="编辑选中的效果器块", command=self.edit_selected_block)
        fx_menu.add_command(label="删除选中的效果器块", command=self.delete_selected_block)
        fx_menu.add_separator()
        fx_menu.add_command(label="显示/隐藏效果器色块", command=self.toggle_effect_blocks)
        menubar.add_cascade(label="效果器", menu=fx_menu)
        
        # 设置菜单
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="音频合成进程数…", command=self.open_settings)
        settings_menu.add_separator()
        settings_menu.add_command(label="恢复默认设置", command=self.reset_settings)
        menubar.add_cascade(label="设置", menu=settings_menu)

        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self.show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        
        self.root.config(menu=menubar)
    # 添加新的批注对话框方法
    def show_annotation_dialog(self):
        """显示批注对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("批注")
        dialog.geometry("500x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 主框架
        main_frame = ttk.Frame(dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 批注列表框架
        list_frame = ttk.LabelFrame(main_frame, text="现有批注", padding=5)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建批注列表
        self.annotation_listbox = tk.Listbox(list_frame, height=8)
        annotation_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.annotation_listbox.yview)
        self.annotation_listbox.config(yscrollcommand=annotation_scrollbar.set)
        
        self.annotation_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        annotation_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 更新批注列表
        self.update_annotation_listbox()
        
        # 批注操作框架
        operation_frame = ttk.Frame(main_frame)
        operation_frame.pack(fill=tk.X, pady=5)
        
        # 小节号输入
        ttk.Label(operation_frame, text="小节号:").pack(side=tk.LEFT, padx=5)
        self.annotation_measure_var = tk.StringVar(value="1")
        measure_entry = ttk.Entry(operation_frame, textvariable=self.annotation_measure_var, width=5)
        measure_entry.pack(side=tk.LEFT, padx=5)
        
        # 按钮框架
        button_frame = ttk.Frame(operation_frame)
        button_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(button_frame, text="添加批注", command=lambda: self.add_annotation_from_dialog(dialog)).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="删除选中", command=self.delete_selected_annotation).pack(side=tk.LEFT, padx=2)
        
        # 批注内容框架
        content_frame = ttk.LabelFrame(main_frame, text="批注内容", padding=5)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.annotation_content_text = tk.Text(content_frame, height=6, width=50)
        content_scrollbar = ttk.Scrollbar(content_frame, orient=tk.VERTICAL, command=self.annotation_content_text.yview)
        self.annotation_content_text.config(yscrollcommand=content_scrollbar.set)
        
        self.annotation_content_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        content_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定列表选择事件
        self.annotation_listbox.bind('<<ListboxSelect>>', self.on_annotation_selected)
        
        # 底部按钮框架
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(bottom_frame, text="关闭", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def update_annotation_listbox(self):
        """更新批注列表显示"""
        self.annotation_listbox.delete(0, tk.END)
        for annotation in self.annotations:
            measure = annotation['measure']
            content_preview = annotation['content'][:30] + "..." if len(annotation['content']) > 30 else annotation['content']
            self.annotation_listbox.insert(tk.END, f"第{measure}小节: {content_preview}")

    def on_annotation_selected(self, event):
        """当批注被选中时显示内容"""
        selection = self.annotation_listbox.curselection()
        if selection:
            index = selection[0]
            annotation = self.annotations[index]
            self.annotation_content_text.delete(1.0, tk.END)
            self.annotation_content_text.insert(1.0, annotation['content'])
            self.annotation_measure_var.set(str(annotation['measure']))

    def add_annotation_from_dialog(self, dialog):
        """从对话框添加批注"""
        try:
            measure = int(self.annotation_measure_var.get())
            content = self.annotation_content_text.get(1.0, tk.END).strip()
            
            if not content:
                messagebox.showwarning("警告", "批注内容不能为空")
                return
            
            # 检查是否已存在该小节的批注
            for i, annotation in enumerate(self.annotations):
                if annotation['measure'] == measure:
                    # 更新现有批注
                    self.annotations[i]['content'] = content
                    self.update_annotation_listbox()
                    messagebox.showinfo("成功", f"已更新第{measure}小节的批注")
                    return
            
            # 添加新批注
            self.annotations.append({
                'measure': measure,
                'content': content
            })
            
            self.update_annotation_listbox()
            self.annotation_content_text.delete(1.0, tk.END)
            messagebox.showinfo("成功", f"已添加第{measure}小节的批注")
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的小节号")

    def delete_selected_annotation(self):
        """删除选中的批注"""
        selection = self.annotation_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "请先选择要删除的批注")
            return
        
        index = selection[0]
        annotation = self.annotations[index]
        
        if messagebox.askyesno("确认删除", f"确定要删除第{annotation['measure']}小节的批注吗？"):
            self.annotations.pop(index)
            self.update_annotation_listbox()
            self.annotation_content_text.delete(1.0, tk.END)
            messagebox.showinfo("成功", "批注已删除")

    def create_main_interface(self):
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 工具栏
        self.toolbar = ttk.Frame(main_frame)
        self.toolbar.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(self.toolbar, text="添加音符", command=self.add_note).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar, text="删除音符", command=self.delete_selected_notes).pack(side=tk.LEFT, padx=2)  # 修改为多选删除
        ttk.Button(self.toolbar, text="导出WAV", command=self.export_to_wav).pack(side=tk.LEFT, padx=2)  # 新增导出按钮
        ttk.Button(self.toolbar, text="放大", command=lambda: self.zoom(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar, text="缩小", command=lambda: self.zoom(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.toolbar, text="重置缩放", command=lambda: self.zoom(0)).pack(side=tk.LEFT, padx=2)

        # 3.0.0：播放控制单独占一行，避免按钮被挤出窗口
        self.playback_bar = ttk.Frame(main_frame)
        self.playback_bar.pack(fill=tk.X, padx=5, pady=(0, 5))
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 创建画布框架 - 移除了钢琴键盘区域
        self.canvas_frame = ttk.Frame(main_frame)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 3.0.0：左侧音轨面板
        self.track_panel_frame = tk.Frame(self.canvas_frame, width=248, bg="#EFEFEF")
        self.track_panel_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        self.track_panel_frame.pack_propagate(False)
        self.build_track_panel()
        
        # 添加水平滚动条
        self.h_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 添加垂直滚动条
        self.v_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        

        # 创建画布
        self.canvas = tk.Canvas(
            self.canvas_frame, 
            bg="white", 
            bd=1, 
            relief=tk.SUNKEN,
            xscrollcommand=self.h_scroll.set,
            yscrollcommand=self.v_scroll.set
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 配置滚动条
        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.config(command=self.canvas.yview)
        
        # 绑定画布配置事件
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # 设置垂直滚动范围（0-95音高）
        self.canvas.config(scrollregion=(0, 0, 10000, 96 * self.pixels_per_pitch))
    
    # ------------------------------------------------------------ 3.0.0 音轨面板
    WAVE_TYPES = ["triangle", "sine", "square", "sawtooth", "filtered_sq",
                  "harmonica", "violin", "string", "ocarina", "organ",
                  "steel-tongue-drum",
                  "harmonics", "custom_harmonics", "beat", "noise",
                  "filtered_noise", "double_filtered_noise", "double_filtered_sq",
                  "audio_sample", "empty"]

    def build_track_panel(self):
        """重建左侧音轨面板"""
        for child in self.track_panel_frame.winfo_children():
            child.destroy()

        header = tk.Frame(self.track_panel_frame, bg="#EFEFEF")
        header.pack(fill=tk.X, pady=(2, 4))
        tk.Label(header, text="音轨", bg="#EFEFEF",
                 font=("Microsoft YaHei", 9, "bold")).pack(side=tk.LEFT, padx=4)
        ttk.Button(header, text="+", width=2, command=self.add_track).pack(side=tk.RIGHT, padx=1)
        ttk.Button(header, text="−", width=2,
                   command=lambda: self.remove_track()).pack(side=tk.RIGHT, padx=1)

        if not hasattr(self, 'current_track_var'):
            self.current_track_var = tk.IntVar(value=0)
        self.current_track_var.set(self.current_track)

        for idx, track in enumerate(self.tracks):
            self._build_track_row(idx, track)

    def refresh_track_panel(self):
        """刷新音轨面板（增删轨道后调用）"""
        self.build_track_panel()

    def _build_track_row(self, idx, track):
        """单条音轨的控制行：选中 / 名称 / 静音 / 独奏 / 音色 / 音量 / 声像"""
        bg = "#FAFAFA"
        row = tk.Frame(self.track_panel_frame, bd=1, relief=tk.GROOVE, bg=bg)
        row.pack(fill=tk.X, pady=1, padx=2)

        # 第一行：选中点 + 颜色块名称 + M/S
        top = tk.Frame(row, bg=bg)
        top.pack(fill=tk.X)
        tk.Radiobutton(top, variable=self.current_track_var, value=idx, bg=bg,
                       command=lambda i=idx: self.set_current_track(i)).pack(side=tk.LEFT)

        # 3.0.0：M/S 先从右边固定住，否则名称标签会把它们挤出面板
        mute_var = tk.BooleanVar(value=track['mute'])
        solo_var = tk.BooleanVar(value=track['solo'])
        track['_mute_var'] = mute_var
        track['_solo_var'] = solo_var
        tk.Checkbutton(top, text="S", variable=solo_var, bg=bg, font=("", 7), padx=1,
                       command=lambda i=idx: self.on_track_solo(i)).pack(side=tk.RIGHT)
        tk.Checkbutton(top, text="M", variable=mute_var, bg=bg, font=("", 7), padx=1,
                       command=lambda i=idx: self.on_track_mute(i)).pack(side=tk.RIGHT)

        name_label = tk.Label(top, text=track['name'], bg=track['color'], anchor=tk.W, width=8,
                              font=("Microsoft YaHei", 8))
        name_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        name_label.bind("<Double-Button-1>", lambda e, i=idx: self.rename_track_at(i))

        # 第二行：音色（左边挂一个该音色的色块，和钢琴卷帘里的音符同色）
        wave_var = tk.StringVar(value=track['wavetype'])
        track['_wave_var'] = wave_var
        wave_row = tk.Frame(row, bg=bg)
        wave_row.pack(fill=tk.X, padx=2, pady=1)
        swatch = tk.Label(wave_row, text=" ", bg=self.wavetype_color(track['wavetype']),
                          width=2, height=1, bd=1, relief=tk.SOLID)
        swatch.pack(side=tk.LEFT, padx=(0, 2))
        track['_wave_swatch'] = swatch
        combo = ttk.Combobox(wave_row, textvariable=wave_var, values=self.WAVE_TYPES,
                             state="readonly", width=12, font=("", 7))
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        combo.bind("<<ComboboxSelected>>", lambda e, i=idx: self.on_track_wavetype(i))

        # 第三行：音量
        vol_var = tk.DoubleVar(value=track['volume'])
        track['_vol_var'] = vol_var
        vol_row = tk.Frame(row, bg=bg)
        vol_row.pack(fill=tk.X, padx=2)
        tk.Label(vol_row, text="音量", bg=bg, font=("", 7), width=4, anchor=tk.W).pack(side=tk.LEFT)
        tk.Scale(vol_row, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                 showvalue=0, bg=bg, highlightthickness=0, variable=vol_var,
                 command=lambda v, i=idx: self.on_track_volume(i, v)).pack(
                     side=tk.LEFT, fill=tk.X, expand=True)

        # 第四行：声像
        pan_var = tk.DoubleVar(value=track['pan'])
        track['_pan_var'] = pan_var
        pan_row = tk.Frame(row, bg=bg)
        pan_row.pack(fill=tk.X, padx=2, pady=(0, 2))
        tk.Label(pan_row, text="声像", bg=bg, font=("", 7), width=4, anchor=tk.W).pack(side=tk.LEFT)
        tk.Scale(pan_row, from_=-1.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 showvalue=0, bg=bg, highlightthickness=0, variable=pan_var,
                 command=lambda v, i=idx: self.on_track_pan(i, v)).pack(
                     side=tk.LEFT, fill=tk.X, expand=True)

    def set_current_track(self, index):
        """切换当前音轨（其它音轨的音符会在画布上淡下去）

        以前这里只改状态栏文字，画布纹丝不动，看起来像“点了没反应”；
        现在换了轨就重画一次，非当前轨的音符压暗，当前轨保持原色。
        """
        index = max(0, min(int(index), len(self.tracks) - 1))
        changed = (index != self.current_track)
        self.current_track = index
        self.current_track_var.set(index)
        self.status_var.set(f"当前音轨: {self.get_track(index)['name']}")
        if changed:
            self.refresh_track_highlight()

    def on_track_volume(self, index, value):
        self.get_track(index)['volume'] = float(value)

    def on_track_pan(self, index, value):
        self.get_track(index)['pan'] = float(value)

    def on_track_mute(self, index):
        track = self.get_track(index)
        track['mute'] = bool(track['_mute_var'].get())

    def on_track_solo(self, index):
        track = self.get_track(index)
        track['solo'] = bool(track['_solo_var'].get())

    def on_track_wavetype(self, index):
        track = self.get_track(index)
        track['wavetype'] = track['_wave_var'].get()
        # 色块跟着音色走
        swatch = track.get('_wave_swatch')
        if swatch is not None:
            try:
                swatch.config(bg=self.wavetype_color(track['wavetype']))
            except tk.TclError:
                pass
        self.status_var.set(
            f"{track['name']} 默认音色 -> {track['wavetype']}（只影响新音符，"
            f"要改已有音符请用 轨道菜单 → 应用轨道音色）")

    def create_playback_controls(self):
        """创建播放控制（3.0.0：独立一行，不再挤在文件工具栏里）"""
        playback_frame = self.playback_bar

        # 3.0.0：播放按钮放在最左，永远不会被挤出窗口
        self.play_btn = ttk.Button(playback_frame, text="▶ 播放", command=self.start_playback)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        self.stop_btn = ttk.Button(playback_frame, text="■ 停止", command=self.stop_playback,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=2)

        ttk.Separator(playback_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        # BPM设置
        ttk.Label(playback_frame, text="BPM:").pack(side=tk.LEFT)
        self.bpm_var = tk.StringVar(value="90")
        bpm_entry = ttk.Entry(playback_frame, textvariable=self.bpm_var, width=5)
        bpm_entry.pack(side=tk.LEFT, padx=2)
        
        # 添加开始小节设置
        ttk.Label(playback_frame, text="开始小节:").pack(side=tk.LEFT, padx=(10,0))
        self.start_measure_var = tk.StringVar(value="1")
        start_measure_entry = ttk.Entry(playback_frame, textvariable=self.start_measure_var, width=4)
        start_measure_entry.pack(side=tk.LEFT, padx=2)
        
        # 3.0.0：结束小节默认跟随乐曲实际长度；用户一旦手动输入，就不再自动改
        ttk.Label(playback_frame, text="结束小节:").pack(side=tk.LEFT, padx=(10,0))
        self.end_measure_var = tk.StringVar(value="64")
        end_measure_entry = ttk.Entry(playback_frame, textvariable=self.end_measure_var, width=4)
        end_measure_entry.pack(side=tk.LEFT, padx=2)
        start_measure_entry.bind("<KeyRelease>", self._mark_measure_range_manual)
        end_measure_entry.bind("<KeyRelease>", self._mark_measure_range_manual)
        
        # 3.0.0：网格吸附
        ttk.Label(playback_frame, text="吸附:").pack(side=tk.LEFT, padx=(10, 0))
        self.snap_combo = ttk.Combobox(playback_frame, textvariable=self.snap_var,
                                       values=list(self.snap_options.keys()),
                                       state="readonly", width=8)
        self.snap_combo.pack(side=tk.LEFT, padx=2)

        # 3.0.0：循环 / 节拍器
        ttk.Checkbutton(playback_frame, text="循环", variable=self.loop_enabled).pack(
            side=tk.LEFT, padx=(12, 2))
        ttk.Checkbutton(playback_frame, text="节拍器", variable=self.metronome_enabled).pack(
            side=tk.LEFT, padx=2)

        # 快捷键提示
        ttk.Label(playback_frame, text="（空格 播放/停止）", foreground="#888888").pack(
            side=tk.LEFT, padx=(10, 0))
        
        # 绑定输入验证
        bpm_entry.bind("<FocusOut>", self.validate_bpm)
        start_measure_entry.bind("<FocusOut>", self.validate_measure_range)
        end_measure_entry.bind("<FocusOut>", self.validate_measure_range)
    
    def validate_bpm(self, event=None):
        """验证并设置BPM值"""
        try:
            bpm = float(self.bpm_var.get())
            if 20 <= bpm <= 300:
                self.bpm = bpm
            else:
                messagebox.showwarning("警告", "BPM值应在20-300之间")
                self.bpm_var.set(str(self.bpm))
        except ValueError:
            messagebox.showwarning("警告", "请输入有效的数字")
            self.bpm_var.set(str(self.bpm))
    
    def _mark_measure_range_manual(self, event=None):
        """用户手动改过开始/结束小节后，就不再自动跟随乐曲长度"""
        self.measure_range_manual = True

    def music_last_measure(self):
        """乐曲实际占用的小节数（按最后一个音符的结束位置向上取整，至少 1）"""
        if not self.sequence:
            return 1
        last_unit = max(n['start'] + n['duration'] for n in self.sequence)
        return max(1, int(math.ceil(last_unit / float(self.units_per_measure) - 1e-9)))

    def sync_measure_range(self, force=False):
        """把播放范围对齐到乐曲长度。

        3.0.0 修复：以前结束小节固定默认 64，超过 64 小节的文件后半段根本不发声
        （Keil-7 这类 101 小节的曲子会丢掉 1500 多个音符）。
        force=True 用于打开/新建文件，无条件重置；否则只在用户没手动指定时跟随。
        """
        total = self.music_last_measure()
        if total > self.max_measures:
            # 网格/滚动区也得跟着长，否则超出部分画不出来也播不到
            self.max_measures = total + 8
            self.draw_grid()
            self.canvas.config(scrollregion=(
                0, 0, self.max_measures * 16 * self.pixels_per_unit,
                96 * self.pixels_per_pitch))
        if force or not self.measure_range_manual:
            self.start_measure_var.set("1")
            self.end_measure_var.set(str(total))
        return total

    def validate_measure_range(self, event=None):
        """验证并设置小节范围"""
        try:
            start_measure = int(self.start_measure_var.get())
            end_measure = int(self.end_measure_var.get())
            
            # 确保小节范围有效
            start_measure = max(1, min(self.max_measures, start_measure))
            end_measure = max(start_measure, min(self.max_measures, end_measure))
            
            self.start_measure_var.set(str(start_measure))
            self.end_measure_var.set(str(end_measure))
            
        except ValueError:
            messagebox.showwarning("警告", "请输入有效的小节数字")
            self.start_measure_var.set("1")
            self.end_measure_var.set(str(self.max_measures))

    def _run_in_background(self, title, work, on_done, subtitle="", delay_ms=150):
        """后台线程跑 work(report)，主线程用模态进度条显示阶段和百分比。

        work(report) 在工作线程里执行，只能做纯计算 —— 不要碰 Tk 控件或 Tk 变量，
        否则轻则界面错乱，重则直接崩。report(阶段文字, 百分比) 可以在工作线程里随便调。
        on_done(result, error) 回到主线程执行，负责收尾和更新界面。
        delay_ms：跑这么久还没结束才把进度条弹出来，几十毫秒的小任务不会闪一下白框。
        """
        box = queue.Queue()
        state = {'done': False, 'dlg': None}
        self._busy = True      # 合成/导出期间挡住重入（快捷键、菜单、按钮都别再触发一次）

        def report(stage=None, percent=None):
            box.put(('p', stage, 0.0 if percent is None else percent))

        def runner():
            try:
                box.put(('r', work(report), None))
            except BaseException as exc:      # 线程里出任何错都要回传，否则弹窗关不掉
                box.put(('e', None, exc))

        def show():
            if state['done']:
                return
            try:
                state['dlg'] = ProgressDialog(self.root, title, subtitle)
            except tk.TclError:
                state['dlg'] = None
            self.root.after(50, poll)

        def poll():
            if state['done']:
                return
            dlg = state['dlg']
            latest = None
            try:
                while True:
                    kind, value, extra = box.get_nowait()
                    if kind == 'p':
                        latest = (value, extra)
                    else:
                        state['done'] = True
                        self._busy = False
                        if dlg is not None:
                            dlg.close()
                        on_done(value, extra)
                        return
            except queue.Empty:
                pass
            if latest and dlg is not None:
                dlg.set(latest[0], latest[1])
            self.root.after(50, poll)

        threading.Thread(target=runner, daemon=True).start()
        self.root.after(delay_ms, show)

    # ------------------------------------------------------------ 设置
    def settings_path(self):
        """设置文件放在程序目录下，跟着程序走"""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')

    def default_settings(self):
        return {'synth_workers': 4}

    def load_settings(self):
        """读取设置；文件不存在或内容不合法就用默认值"""
        defaults = self.default_settings()
        data = {}
        try:
            with open(self.settings_path(), 'r', encoding='utf-8') as f:
                data = json.load(f) or {}
        except Exception:
            data = {}
        try:
            workers = int(data.get('synth_workers', defaults['synth_workers']))
        except (TypeError, ValueError):
            workers = defaults['synth_workers']
        self.synth_workers = max(1, min(self.max_synth_workers, workers))

    def save_settings(self):
        try:
            with open(self.settings_path(), 'w', encoding='utf-8') as f:
                json.dump({'synth_workers': int(self.synth_workers)}, f,
                          ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存设置失败: {e}")

    def apply_synth_workers(self, workers):
        """改进程数：旧的合成进程池是按旧数量建的，直接扔掉，下次合成重建"""
        workers = max(1, min(self.max_synth_workers, int(workers)))
        if workers != self.synth_workers:
            self.synth_workers = workers
            try:
                shutdown_render_pool()
            except Exception as e:
                print(f"关闭合成进程池失败: {e}")
        self.save_settings()
        if hasattr(self, 'status_var'):
            self.status_var.set(f"音频合成进程数：{workers}"
                                + ("（串行）" if workers == 1 else ""))
        return self.synth_workers

    def reset_settings(self):
        self.apply_synth_workers(self.default_settings()['synth_workers'])
        messagebox.showinfo("设置", f"已恢复默认设置：音频合成进程数 "
                                    f"{self.synth_workers}")

    def open_settings(self):
        SettingsDialog(self, self.root)

    def start_playback(self):
        """开始播放音频（使用pygame）"""
        if getattr(self, '_busy', False):
            return          # 上一次合成还没结束
        if not self.sequence:
            messagebox.showinfo("提示", "没有可播放的音符数据")
            return
        
        # 3.0.0：范围未被手动指定时，先跟随乐曲实际长度（新加的音符也能播到）
        self.sync_measure_range()

        # 验证BPM和小节范围
        self.validate_bpm()
        self.validate_measure_range()
        
        # 获取开始和结束小节
        start_measure = int(self.start_measure_var.get()) - 1  # 转换为0-based
        end_measure = int(self.end_measure_var.get())
        
        # 计算时间范围（十六分音符单位）
        start_time = start_measure * self.units_per_measure
        end_time = end_measure * self.units_per_measure
        
        # 关键修改：过滤在时间范围内的音符（只包含起始位置在播放范围内的音符）
        filtered_notes = []
        for note in self.sequence:
            note_start = note['start']
            # 只选择起始时间在播放开始时间之后，且在播放结束时间之前的音符
            if note_start >= start_time and note_start <= end_time:
                filtered_notes.append(note)
        
        if not filtered_notes:
            messagebox.showinfo("提示", f"在{start_measure+1}-{end_measure}小节范围内没有可播放的音符数据")
            return
        
        # 如果已经在播放，先停止
        if self.is_playing:
            self.stop_playback()
            time.sleep(0.1)
        
        try:
            # 分离普通事件和音频样本事件（使用过滤后的音符）
            normal_events = []
            sample_events = []

            # 3.0.0：静音 / 独奏 全局判定
            any_solo = any(track.get('solo') for track in self.tracks)

            for note in filtered_notes:
                track = self.get_track(note.get('track', 0))
                if track.get('mute'):
                    continue
                if any_solo and not track.get('solo'):
                    continue

                # 调整音符的起始时间为相对时间（相对于播放开始时间）
                relative_start = note['start'] - start_time  # 直接计算相对时间
                gain = track.get('volume', 1.0)

                # 修改这部分代码：
                if note['wavetype'] == 'audio_sample':
                    sample_events.append({
                        # 与导出一致：采样提前 0.25 个十六分音符，抵消触发/解码延迟
                        'start_time': max(0.0, relative_start - 0.25),
                        'sample_info': note['kwargs'],
                        # 关键修改：添加振幅信息
                        'start_amp': note['start_amp'] * gain
                    })
                else:
                    midi_pitch = note['pitch'] + 12
                    freq = 440 * (2 ** ((midi_pitch - 69) / 12))

                    event = {
                        'wavetype': note['wavetype'],
                        'freq': freq,
                        'start_time': relative_start,  # 使用相对时间
                        'duration': note['duration'],  # 使用原始时长
                        'start_amp': note['start_amp'] * gain,
                        'amp_change_rate': note.get('amp_change_rate', 0.0),
                        # 3.1：音符自身参数 + 效果器块折算出来的参数（播放与导出共用）
                        'kwargs': self.merged_note_kwargs(note),
                        'pan': track.get('pan', 0.0),   # 3.0.0：声像
                    }
                    normal_events.append(event)

            # 3.0.0：节拍器（只在播放时叠加，不写进文件）
            if self.metronome_enabled.get():
                beat_units = 4 if self.time_signature[1] == 4 else 2
                beats_per_measure = max(1, self.time_signature[0])
                offset = 0.0
                beat_index = 0
                span = end_time - start_time
                while offset < span:
                    freq = 1568.0 if beat_index % beats_per_measure == 0 else 1046.5
                    normal_events.append({
                        'wavetype': 'sine',
                        'freq': freq,
                        'start_time': offset,
                        'duration': 0.6,
                        'start_amp': 0.22,
                        'amp_change_rate': 0.0,
                        'kwargs': {},
                        'pan': 0.0,
                    })
                    offset += beat_units
                    beat_index += 1

            # 计算时间单位
            time_unit = 60.0 / (self.bpm * 4)
            self.loop_units = end_time - start_time     # 3.0.0：循环长度（单位）
            self.play_time_unit = time_unit

            # Tk 变量只能在主线程读，先取好再交给后台线程
            self._play_looping = self.loop_enabled.get()

            n_normal = len(normal_events)
            n_sample = len(sample_events)
            # 主线程取好，别让工作线程碰设置；音符太少时实际会退回串行
            workers = resolve_workers(self.synth_workers, n_normal)

            def work(report):
                """后台线程：合成 + 混采样。纯计算，不碰任何 Tk 对象。"""
                report("准备合成…", 2)
                if n_normal:
                    def on_note(done, total):
                        report(f"合成音符 {done}/{total}",
                               2 + 76.0 * done / max(1, total))
                    if workers > 1:
                        report(f"正在启动 {workers} 个合成进程…", 1)
                    audio = generate_audio_from_events(
                        normal_events, time_unit, progress=on_note, workers=workers)
                else:
                    empty_data = np.zeros(44100, dtype=np.int16)
                    audio = SimpleAudioSegment(array('h', empty_data), 44100)

                # 3.0.0：把采样离线混进整曲（以前靠实时声道触发，密集段落会随机丢采样）
                if sample_events:
                    report(f"混合 {n_sample} 个音频采样…", 84)
                    audio = self.mix_audio_with_samples(audio, sample_events, time_unit)

                report("准备播放缓冲…", 97)
                return audio

            def done(audio, error):
                self._start_playback_buffer(audio, error,
                                            start_measure, end_measure, start_time)

            self.status_var.set(f"正在合成 {n_normal} 个音符…")
            title = (f"正在合成音频（{workers} 进程并行）" if workers > 1
                     else "正在合成音频（串行）")
            self._run_in_background(
                title, work, done,
                subtitle=f"{n_normal} 个音符 / {n_sample} 个采样")

        except Exception as e:
            messagebox.showerror("错误", f"音频生成或播放失败: {str(e)}")
            import traceback
            traceback.print_exc()

    def _start_playback_buffer(self, music_audio, error, start_measure, end_measure, start_time):
        """合成结束（主线程）：把 buffer 交给 pygame 播放，并更新界面状态"""
        if error is not None:
            self.status_var.set("合成失败")
            messagebox.showerror("错误", f"音频生成失败: {str(error)}")
            return

        self.music_audio = music_audio

        # 采样已混入 buffer，播放线程不再需要实时触发
        self.sample_events = []
        self.current_play_time = 0
        self.played_samples = set()

        # 存储播放范围信息
        self.play_start_measure = start_measure
        self.play_end_measure = end_measure
        self.play_start_time_units = start_time  # 存储起始时间（单位）

        self.is_playing = True
        self.play_start_time = time.time()
        self.play_thread_running = True
        self._playback_ended = False

        # 开始播放（在后台线程）
        self.play_thread = threading.Thread(target=self.play_audio_thread)
        self.play_thread.daemon = True
        self.play_thread.start()

        # 开始更新播放位置（从起始小节开始）
        self.update_playback_position()

        # 更新按钮状态
        self.play_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_var.set(f"正在播放 {start_measure+1}-{end_measure}小节 - BPM: {self.bpm}")
    
    def play_audio_sample(self, sample_event):
        """独立播放音频样本 - 添加振幅控制"""
        try:
            sample_info = sample_event['sample_info']
            sample_file = sample_info.get('sample_file')
            sample_id = sample_info.get('sample_id')
            
            # 关键修改：获取振幅信息
            start_amp = sample_event.get('start_amp', 1.0)  # 默认1.0
            # 确保振幅在合理范围内
            volume = max(0.0, min(1.0, start_amp))
            
            if sample_file and os.path.exists(sample_file):
                # 使用pygame直接播放音频文件，应用振幅控制
                sample_sound = pygame.mixer.Sound(sample_file)
                
                # 关键修改：设置音量
                sample_sound.set_volume(volume)
                
                # 获取一个空闲的声道播放样本（0 号留给整曲，采样只用 1 号及以后）
                first_ch = 1 if pygame.mixer.get_num_channels() > 1 else 0
                for i in range(first_ch, pygame.mixer.get_num_channels()):
                    channel = pygame.mixer.Channel(i)
                    if not channel.get_busy():
                        channel.play(sample_sound)
                        break
                else:
                    # 没有空闲声道时抢占第一个采样声道，绝不动 0 号（整曲）
                    pygame.mixer.Channel(first_ch).play(sample_sound)
                
                # 可选：打印调试信息
                # print(f"播放音频样本: {os.path.basename(sample_file)}, 音量: {volume:.2f}")
                
        except Exception as e:
            print(f"播放音频样本失败: {str(e)}")

    def create_audio_events(self):
        """将音序转换为音频事件列表"""
        events = []
        for note in self.sequence:
            # 将音高转换为频率
            midi_pitch = note['pitch'] + 12  # C0在MIDI系统中是12
            freq = 440 * (2 ** ((midi_pitch - 69) / 12))
            
            # 关键修改：修复键名错误
            event = {
                'wavetype': note['wavetype'],
                'freq': freq,
                'start_time': note['start'],
                'duration': note['duration'],
                'start_amp': note['start_amp'],
                'amp_change_rate': note.get('amp_change_rate', 0.0),  # 修复这里
                'kwargs': note['kwargs']
            }
            events.append(event)
        return events
    
    def play_audio_thread(self):
        """使用pygame播放音频 - 支持音频样本独立播放"""
        try:
            # 初始化pygame mixer
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            
            # 播放音乐音频（普通音色）
            # 关键修复：pygame mixer 初始化为立体声(channels=2)，若音频为单声道
            # 需转换为立体声，否则 pygame 会将每对样本误解释为一个立体声帧，
            # 导致播放速度×2、音高+1八度
            if self.music_audio.channels == 1:
                mono = np.array(self.music_audio.data, dtype=np.int16)
                stereo = np.empty(len(mono) * 2, dtype=np.int16)
                stereo[0::2] = mono  # 左声道
                stereo[1::2] = mono  # 右声道（复制）
                music_data = stereo.tobytes()
            else:
                music_data = self.music_audio.data.tobytes()
            music_sound = pygame.mixer.Sound(buffer=music_data)
            # 3.0.0：整曲固定占 0 号声道；循环播放时让 mixer 自己无限循环，省掉每次重置的开销
            looping = getattr(self, '_play_looping', False)
            music_channel = pygame.mixer.Channel(0)
            self.music_channel = music_channel
            music_channel.play(music_sound, loops=-1 if looping else 0)
            
            # 记录播放开始时间
            play_start_time = time.time()
            
            # 3.0.0：采样已离线混入同一 buffer，这里只需等待整曲播完。
            # （旧实现靠实时声道逐个触发采样，声道不足就会随机吞掉采样）
            while self.play_thread_running:
                if not looping and not music_channel.get_busy():
                    break
                time.sleep(0.02)
                
        except Exception as e:
            print(f"播放错误: {str(e)}")
        finally:
            # 不能在线程里碰 Tk（after/messagebox 都不是线程安全的），
            # 只置标志，由主线程的定时器收尾
            self._playback_ended = True
    
    def export_to_wav(self):
        """导出当前工程为WAV音频文件（可选择8bit或16bit）"""
        if getattr(self, '_busy', False):
            return          # 上一次合成还没结束
        if not self.sequence:
            messagebox.showinfo("提示", "没有可导出的音符数据")
            return
        
        # 创建位深度选择对话框
        bit_depth_dialog = tk.Toplevel(self.root)
        bit_depth_dialog.title("选择音频位深度")
        bit_depth_dialog.geometry("300x200")
        bit_depth_dialog.transient(self.root)
        bit_depth_dialog.grab_set()
        
        ttk.Label(bit_depth_dialog, text="请选择音频位深度:").pack(pady=15)
        
        self.bit_depth_var = tk.StringVar(value="16")  # 默认16bit
        
        bit_frame = ttk.Frame(bit_depth_dialog)
        bit_frame.pack(pady=10)
        
        ttk.Radiobutton(bit_frame, text="8-bit (较小文件，音质较低)", 
                    variable=self.bit_depth_var, value="8").pack(anchor=tk.W)
        ttk.Radiobutton(bit_frame, text="16-bit (标准CD音质)", 
                    variable=self.bit_depth_var, value="16").pack(anchor=tk.W)
        
        def on_bit_depth_selected():
            bit_depth_dialog.destroy()
            self.continue_export_to_wav()
        
        def cancel_export():
            bit_depth_dialog.destroy()
            return
         
        btn_frame = ttk.Frame(bit_depth_dialog)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="确定", command=on_bit_depth_selected).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=cancel_export).pack(side=tk.LEFT, padx=10)

    def continue_export_to_wav(self):
        """继续导出WAV文件（根据选择的位深度）"""
        if getattr(self, '_busy', False):
            return          # 上一次合成还没结束
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV音频文件", "*.wav"), ("所有文件", "*.*")],
            title="导出为WAV文件"
        )
        
        if not file_path:
            return  # 用户取消了保存
        
        try:
            # 显示导出进度
            self.status_var.set("正在生成音频...")
            self.root.update()
            
            # 获取当前播放设置
            self.validate_bpm()
            self.validate_measure_range()
            
            start_measure = int(self.start_measure_var.get()) - 1
            end_measure = int(self.end_measure_var.get())
            
            # 计算时间范围（十六分音符单位）
            start_time = start_measure * self.units_per_measure
            end_time = end_measure * self.units_per_measure
            
            # 过滤在时间范围内的音符
            filtered_notes = []
            for note in self.sequence:
                note_start = note['start']
                if note_start >= start_time and note_start <= end_time:
                    filtered_notes.append(note)
            
            if not filtered_notes:
                messagebox.showinfo("提示", f"在{start_measure+1}-{end_measure}小节范围内没有可导出的音符数据")
                return
            
            # 分离普通事件和音频样本事件
            normal_events = []
            sample_events = []

            # 3.0.0：导出同样遵守 静音 / 独奏 / 音量 / 声像
            any_solo = any(track.get('solo') for track in self.tracks)

            for note in filtered_notes:
                track = self.get_track(note.get('track', 0))
                if track.get('mute'):
                    continue
                if any_solo and not track.get('solo'):
                    continue

                # 调整音符的起始时间为相对时间
                relative_start = note['start'] - start_time
                gain = track.get('volume', 1.0)
                
                if note['wavetype'] == 'audio_sample':
                    # 关键修改：音频样本播放时间提前0.25个16分音符
                    adjusted_start = max(0, relative_start - 0.25)  # 确保不会变成负数
                    sample_events.append({
                        'start_time': adjusted_start,  # 使用调整后的时间
                        'sample_info': note['kwargs'],
                        'start_amp': note['start_amp'] * gain  # 添加振幅信息
                    })
                else:
                    midi_pitch = note['pitch'] + 12
                    freq = 440 * (2 ** ((midi_pitch - 69) / 12))
                    
                    event = {
                        'wavetype': note['wavetype'],
                        'freq': freq,
                        'start_time': relative_start,  # 使用原始相对时间
                        'duration': note['duration'],
                        'start_amp': note['start_amp'] * gain,
                        'amp_change_rate': note.get('amp_change_rate', 0.0),
                        # 3.1：音符自身参数 + 效果器块折算出来的参数（播放与导出共用）
                        'kwargs': self.merged_note_kwargs(note),
                        'pan': track.get('pan', 0.0),   # 3.0.0：声像
                    }
                    normal_events.append(event)
            
            # 计算时间单位
            time_unit = 60.0 / (self.bpm * 4)

            n_normal = len(normal_events)
            n_sample = len(sample_events)
            bit8 = self.bit_depth_var.get() == "8"
            out_path = file_path
            workers = resolve_workers(self.synth_workers, n_normal)

            def work(report):
                """后台线程：合成 + 混采样 + 转位深 + 写文件。纯计算，不碰 Tk。"""
                report("准备合成…", 2)
                if n_normal:
                    def on_note(done, total):
                        report(f"合成音符 {done}/{total}",
                               2 + 74.0 * done / max(1, total))
                    if workers > 1:
                        report(f"正在启动 {workers} 个合成进程…", 1)
                    music_audio = generate_audio_from_events(
                        normal_events, time_unit, progress=on_note, workers=workers)
                else:
                    # 如果没有普通事件，创建空音频
                    empty_data = np.zeros(44100, dtype=np.int16)
                    music_audio = SimpleAudioSegment(array('h', empty_data), 44100)

                # 3.0.0：采样离线混入整曲；播放与导出共用同一份逻辑，两边结果一致
                if sample_events:
                    report(f"混合 {n_sample} 个音频采样…", 82)
                    music_audio = self.mix_audio_with_samples(
                        music_audio, sample_events, time_unit)

                # 根据选择的位深度进行转换
                report("转换位深度…", 90)
                if bit8:
                    data8 = (np.array(music_audio.data, dtype=np.float32) / 32767.0 * 127.5 + 127.5).astype(np.uint8)
                    final_audio = SimpleAudioSegment(array('B', data8), music_audio.frame_rate,
                                                     sample_width=1, channels=music_audio.channels)
                else:
                    final_audio = music_audio

                report("写入 WAV 文件…", 95)
                final_audio.export(out_path)
                return os.path.getsize(out_path)

            def done(size, error):
                if error is not None:
                    self.status_var.set("导出失败")
                    messagebox.showerror("导出错误", f"导出音频时出错: {str(error)}")
                    return
                bit_depth_text = "8-bit" if bit8 else "16-bit"
                self.status_var.set(f"{bit_depth_text}音频已导出: {os.path.basename(out_path)}")
                messagebox.showinfo(
                    "导出成功",
                    f"{bit_depth_text}音频文件已成功导出到:\n{out_path}"
                    f"\n（{size / 1048576.0:.1f} MB）")

            self.status_var.set(f"正在合成 {n_normal} 个音符…")
            title = (f"正在导出音频（{workers} 进程并行）" if workers > 1
                     else "正在导出音频（串行）")
            self._run_in_background(
                title, work, done,
                subtitle=f"{n_normal} 个音符 / {n_sample} 个采样 / "
                         f"{'8-bit' if bit8 else '16-bit'}")

        except Exception as e:
            messagebox.showerror("导出错误", f"导出音频时出错: {str(e)}")
            self.status_var.set("导出失败")
            import traceback
            traceback.print_exc()

    def _resolve_sample_path(self, path):
        """把采样路径解析成真实文件。

        音序文件里存的是相对路径（如 audio_samples\\beat_17.mp3），而程序可能从
        任意工作目录启动；找不到文件时采样会被静默丢弃，听起来就是"丢音"。
        这里依次在 当前目录 / 程序目录 / 项目根 / 音序文件所在目录 里找。
        """
        if not path or os.path.isabs(path):
            return path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [path,
                      os.path.join(base_dir, path),
                      os.path.join(os.path.dirname(base_dir), path)]
        current_file = getattr(self, 'current_file', None)
        if current_file:
            song_dir = os.path.dirname(os.path.abspath(current_file))
            candidates += [os.path.join(song_dir, path),
                           os.path.join(os.path.dirname(song_dir), path)]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return path

    def _decode_sample_base(self, file_path):
        """把采样解码成 44.1kHz 单声道浮点（振幅处理之前），按文件缓存。

        密集曲子里同一个采样文件会被上百个音符复用，只解码一次即可，
        否则每次都要重新解码+重采样，播放前会卡十几秒。
        """
        file_path = self._resolve_sample_path(file_path)
        cached = self._sample_cache.get(file_path)
        if cached is not None:
            return cached
        try:
            sound = pygame.mixer.Sound(file_path)
            raw = pygame.sndarray.array(sound)
            if len(raw.shape) > 1:
                raw = raw.mean(axis=1)
            duration = sound.get_length()
            original_rate = int(len(raw) / duration) if duration else 44100
            if original_rate != 44100:
                from scipy import signal
                raw = signal.resample(raw, int(len(raw) * 44100 / original_rate))
            data = raw.astype(np.float32) / 32767.0
        except Exception as e:
            print(f"加载完整音频样本失败: {str(e)}")
            return None
        if len(self._sample_cache) < 1024:
            self._sample_cache[file_path] = data
        return data

    def _decode_sample(self, file_path, amplitude):
        """在基础波形上应用振幅与淡入淡出，得到可直接混入整曲的采样。"""
        base = self._decode_sample_base(file_path)
        if base is None or not len(base):
            return None
        audio = base * float(amplitude)          # 复制一份，别改到缓存
        fade_in_length = min(300, len(audio) // 10)
        if fade_in_length > 0:
            audio[:fade_in_length] *= np.linspace(0, 1, fade_in_length)
        fade_out_length = min(int(0.5 * 44100), len(audio))
        if fade_out_length > 0:
            audio[-fade_out_length:] *= np.linspace(1, 0, fade_out_length)
        peak = float(np.max(np.abs(audio)))
        if peak > 1.0:
            audio /= peak
        return np.round(audio * 31 - 1) / 32

    def mix_audio_with_samples(self, music_audio, sample_events, time_unit):
        """把 audio_sample 事件离线混进整曲，返回新的 SimpleAudioSegment。

        3.0.0 修复：播放原先是用 pygame 实时声道逐个触发采样的——声道只有 8 个，
        而且每次触发都要重新解码采样文件；音符密集时就会有采样被抢占或迟到，
        听感上就是“随机丢失部分音符”。导出走的是离线混合所以没有问题。
        这里让播放也走离线混合，两边结果才一致。
        """
        sample_rate = getattr(music_audio, 'frame_rate', 44100) or 44100
        channels = getattr(music_audio, 'channels', 1) or 1

        raw = np.array(music_audio.data, dtype=np.float32) / 32767.0
        base = raw.reshape(-1, 2) if channels == 2 else raw
        total_frames = base.shape[0]

        # 先把采样全部解码，并把时间线扩到能装下最后一个采样
        decoded = []
        for event in sample_events:
            info = event.get('sample_info') or {}
            path = self._resolve_sample_path(info.get('sample_file'))
            if not path or not os.path.exists(path):
                continue
            data = self._decode_sample(path, event.get('start_amp', 1.0))
            if data is None or not len(data):
                continue
            start_frame = int(max(0.0, event['start_time'] * time_unit) * sample_rate)
            decoded.append((start_frame, data))
            total_frames = max(total_frames, start_frame + len(data))

        if not decoded:
            return music_audio

        if channels == 2:
            timeline = np.zeros((total_frames, 2), dtype=np.float32)
            timeline[:base.shape[0], 0] = base[:, 0]
            timeline[:base.shape[0], 1] = base[:, 1]
        else:
            timeline = np.zeros(total_frames, dtype=np.float32)
            timeline[:base.shape[0]] = base

        for start_frame, data in decoded:
            end_frame = start_frame + len(data)
            if channels == 2:
                timeline[start_frame:end_frame, 0] += data
                timeline[start_frame:end_frame, 1] += data
            else:
                timeline[start_frame:end_frame] += data

        peak = float(np.max(np.abs(timeline))) if timeline.size else 0.0
        if peak > 1.0:
            timeline /= peak

        pcm = (timeline.reshape(-1) * 32767).astype(np.int16)
        pcm_array = array('h')
        pcm_array.frombytes(pcm.tobytes())
        return SimpleAudioSegment(pcm_array, sample_rate, sample_width=2, channels=channels)

    def load_complete_audio_sample(self, file_path, target_sample_rate=44100, amplitude=1.0):
        """加载完整的音频样本（不截断，只播放一次）"""
        try:
            # 使用pygame加载音频
            sound = pygame.mixer.Sound(file_path)
            
            # 获取音频数据
            raw_array = pygame.sndarray.array(sound)
            
            # 转换为单声道
            if len(raw_array.shape) > 1:
                raw_array = raw_array.mean(axis=1)
            
            # 获取原始音频长度（秒）
            original_duration = sound.get_length()
            
            # 计算原始采样率
            original_sample_rate = int(len(raw_array) / original_duration)
            
            # 如果需要重采样到目标采样率
            if original_sample_rate != target_sample_rate:
                from scipy import signal
                resample_num = int(len(raw_array) * target_sample_rate / original_sample_rate)
                audio_data = signal.resample(raw_array, resample_num)
            else:
                audio_data = raw_array
            
            # 转换为浮点数并进行归一化
            audio_data = audio_data.astype(np.float32) / 32767.0
            
            # 应用振幅
            audio_data = audio_data * amplitude
            
            # 应用淡入淡出（保持原有的淡入淡出效果）
            fade_in_length = min(300, len(audio_data) // 10)
            if fade_in_length > 0:
                fade_in = np.linspace(0, 1, fade_in_length)
                audio_data[:fade_in_length] *= fade_in
            
            # 淡出
            fade_out_length = min(int(0.5 * target_sample_rate), len(audio_data))
            if fade_out_length > 0:
                fade_out = np.linspace(1, 0, fade_out_length)
                audio_data[-fade_out_length:] *= fade_out
            #-16~15
            # 确保值在有效范围内
            max_val = np.max(np.abs(audio_data))
            if max_val > 1.0:
                audio_data /= max_val
            audio_data=np.round(audio_data*31-1)/32
            return audio_data
            
        except Exception as e:
            print(f"加载完整音频样本失败: {str(e)}")
            return None
        
    def cleanup(self):
        """安全清理资源"""
        self.stop_playback()
        
        # 等待线程结束
        if hasattr(self, 'play_thread') and self.play_thread.is_alive():
            self.play_thread.join(timeout=0.1)

        # 回收合成子进程，否则关窗口后它们会留在后台
        try:
            shutdown_render_pool()
        except Exception:
            pass

        # 退出pygame
        pygame.mixer.quit()
        pygame.quit()
        
    def update_playback_position(self):
        """更新播放位置并滚动视图"""
        # 3.0.0：播放线程播完只置标志，这里在主线程收尾
        if self._playback_ended:
            self._playback_ended = False
            self.on_playback_finished()
            return

        if not self.is_playing:
            return
        
        # 计算当前播放时间
        self.current_play_time = time.time() - self.play_start_time
        
        # 计算当前播放位置（十六分音符），考虑起始小节
        time_per_unit = 60.0 / (self.bpm * 4)
        current_unit = self.current_play_time / time_per_unit

        # 3.0.0：循环播放时播放头在区间内回绕，不触发停止
        loop_units = getattr(self, 'loop_units', 0)
        if self.loop_enabled.get() and loop_units > 0:
            absolute_unit = self.play_start_time_units + (current_unit % loop_units)
            x_pos = absolute_unit * self.pixels_per_unit
            self._follow_playhead(x_pos)
            self.play_update_id = self.root.after(20, self.update_playback_position)
            return

        # 计算绝对位置（考虑起始小节偏移）
        absolute_unit = self.play_start_time_units + current_unit
        
        # 计算播放位置X坐标
        x_pos = absolute_unit * self.pixels_per_unit
        
        # 关键修改：检查是否到达结束小节（基于绝对位置）
        end_time_units = self.play_end_measure * self.units_per_measure
        if absolute_unit >= end_time_units:
            self.stop_playback()
            return
        
        # 滚动视图，使播放位置位于中央
        self._follow_playhead(x_pos)
        
        # 每20毫秒更新一次
        self.play_update_id = self.root.after(20, self.update_playback_position)
    
    def _follow_playhead(self, x_pos):
        """播放时跟随播放头滚动（节流到约 10Hz）。

        xview_moveto 会触发整块画布重绘；音符密度高时画布有上万个图元，
        每 20ms 滚一次会把主线程拖满，音频回调来不及就被迫吐静音，
        听感正是"随机丢音"。这里限制到 100ms 一次，肉眼几乎看不出差别。
        """
        now = time.time()
        if now - getattr(self, '_last_scroll_time', 0.0) < 0.1:
            return
        self._last_scroll_time = now
        self.scroll_to_position(x_pos)

    def scroll_to_position(self, x_pos):
        """滚动到指定位置"""
        # 获取当前视图范围
        xview = self.canvas.xview()
        canvas_width = self.canvas.winfo_width()
        
        # 计算视图中心位置
        view_center = (xview[0] + xview[1]) / 2 * self.canvas.bbox("all")[2]
        
        # 计算需要滚动到的位置（使播放位置位于视图中央）
        scroll_pos = (x_pos - canvas_width / 2) / self.canvas.bbox("all")[2]
        self.canvas.xview_moveto(max(0, min(1, scroll_pos)))
    
    def stop_playback(self):
        """停止播放"""
        if not self.is_playing:
            return
        
        # 设置停止标志
        self.play_thread_running = False
        self.is_playing = False
        
        # 停止pygame mixer
        pygame.mixer.stop()
        
        # 更新状态栏
        self.status_var.set("停止中...")
        
        # 等待播放线程结束（增加超时时间）
        if hasattr(self, 'play_thread') and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)  # 增加超时时间
        
        # 更新按钮状态
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        # 取消更新定时器
        if self.play_update_id:
            self.root.after_cancel(self.play_update_id)
            self.play_update_id = None
        
        # 更新状态栏
        self.status_var.set("播放已停止")

    def on_playback_finished(self):
        """播放完成时的处理"""
        # 添加安全检查
        if not self.is_playing:
            return
        
        # 停止播放对象
        if self.play_obj and self.play_obj.is_playing():
            self.play_obj.stop()
        
        # 更新播放状态
        self.is_playing = False
        
        # 更新按钮状态
        self.play_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        
        # 取消更新定时器
        if self.play_update_id:
            self.root.after_cancel(self.play_update_id)
            self.play_update_id = None
        
        # 更新状态栏
        self.status_var.set("播放完成")
        
        # 清理播放对象
        self.play_obj = None

    def set_time_signature(self, numerator, denominator):
        """设置节拍"""
        self.time_signature = [numerator, denominator]
        
        # 计算每小节的单位数（十六分音符）
        # 4/4拍: 1小节 = 4拍 × 4（十六分音符）= 16单位
        # 3/4拍: 1小节 = 3拍 × 4（十六分音符）= 12单位
        # 6/8拍: 1小节 = 6拍 × 2（十六分音符）= 12单位
        if denominator == 4:
            self.units_per_measure = numerator * 4
        elif denominator == 8:
            self.units_per_measure = numerator * 2
        
        # 更新网格
        self.draw_grid()
        self.status_var.set(f"节拍已设置为: {numerator}/{denominator}")

    def add_annotation(self):
        """添加批注对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("添加批注")
        dialog.geometry("400x300")
        
        ttk.Label(dialog, text="小节号:").pack(pady=5)
        measure_var = tk.StringVar(value="1")
        measure_entry = ttk.Entry(dialog, textvariable=measure_var, width=10)
        measure_entry.pack(pady=5)
        
        ttk.Label(dialog, text="批注内容:").pack(pady=5)
        content_text = tk.Text(dialog, height=5, width=40)
        content_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        def save_annotation():
            try:
                measure = int(measure_var.get())
                content = content_text.get("1.0", tk.END).strip()
                if content:
                    self.annotations.append({
                        'measure': measure,
                        'content': content
                    })
                    self.draw_annotations()
                    dialog.destroy()
            except ValueError:
                messagebox.showerror("错误", "请输入有效的小节号")
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="确定", command=save_annotation).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=10)

    def draw_grid(self):
        """绘制钢琴卷帘网格（支持最多self.max_measures小节）"""
        """绘制钢琴卷帘网格（支持纵向缩放）"""
        self.canvas.delete("grid")
        
        # 计算总宽度和总高度（使用缩放因子）
        total_width = self.max_measures * 16 * self.pixels_per_unit
        num_pitches = 96  # C0到B7（96个半音）
        total_height = num_pitches * self.pixels_per_pitch
        
        # 更新画布滚动区域
        self.canvas.config(scrollregion=(0, 0, total_width, total_height))
        
        # 绘制垂直网格线（时间线）
        # 计算每拍的单位数
        if self.time_signature[1] == 4:  # 4分音符为一拍
            units_per_beat = 4
        elif self.time_signature[1] == 8:  # 8分音符为一拍
            units_per_beat = 2
        
        # 绘制所有self.max_measures小节的网格线
        for measure_idx in range(self.max_measures + 1):  # 包括0到64小节
            x_measure = measure_idx * self.units_per_measure * self.pixels_per_unit
            
            # 小节线（实线）
            self.canvas.create_line(
                x_measure, 0, x_measure, total_height,
                fill="#c0c0c0", 
                tags="grid"
            )
            
            # 添加小节编号
            if measure_idx % 4 == 0 and measure_idx < self.max_measures:
                self.canvas.create_text(
                    x_measure + 5, 5,
                    text=f"小节 {measure_idx+1}",
                    anchor=tk.NW,
                    fill="#808080",
                    tags="grid"
                )
            
            # 绘制拍线（虚线）
            if measure_idx < self.max_measures:
                for beat_idx in range(1, self.time_signature[0]):  # 每小节内的拍子
                    x_beat = x_measure + beat_idx * units_per_beat * self.pixels_per_unit
                    self.canvas.create_line(
                        x_beat, 0, x_beat, total_height,
                        fill="#e0e0e0", 
                        dash=(4, 2),
                        tags="grid"
                    )
        
        # 音符名称列表
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        for pitch_idx in range(num_pitches):
            # 使用纵向缩放因子计算位置
            y = (num_pitches - 1 - pitch_idx) * self.pixels_per_pitch
            
            # 计算实际MIDI音高
            midi_pitch = pitch_idx + 12
            octave = midi_pitch // 12
            note_idx = midi_pitch % 12
            note_name = notes[note_idx]
            
            # 绘制网格线
            self.canvas.create_line(
                0, y, total_width, y,
                fill="#e0e0e0" if pitch_idx % 12 != 0 else "#c0c0c0",
                dash=(2, 2) if pitch_idx % 12 != 0 else (),
                tags="grid"
            )
            
            # 在左侧添加音高编号和音符名称
            self.canvas.create_text(
                5, y + self.pixels_per_pitch / 2,
                text=f"{note_name}{octave-1}",
                anchor=tk.W,
                fill="#000000",
                tags="grid"
            )
        
        # 3.1：先画效果器色块（压在音符下面），再画音符
        self.draw_effect_blocks()
        # 绘制音符
        self.draw_notes()
    
    def shade_color(self, hex_color, factor):
        """把颜色按亮度系数缩放到 (0~1)，用于力度可视化和静音变灰"""
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return "#CCCCCC"
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = max(0, min(255, int(r * factor)))
        g = max(0, min(255, int(g * factor)))
        b = max(0, min(255, int(b * factor)))
        return f"#{r:02X}{g:02X}{b:02X}"

    def _note_colors(self, note, track, active=True):
        """音符填充色与音名文字色（按 音色×力度×静音×是否当前轨 缓存）

        3.0.0：填充色 = 音色颜色 × 力度（每种音色一个色块）；静音轨统一灰掉
        力度用平方根映射，避免低幅度音符（和弦 0.15）被压成近黑色
        active=False 表示这条音符不在当前音轨上，压暗一档，好让当前轨跳出来
        """
        amp = max(0.0, min(1.0, note.get('start_amp', 0.5)))
        brightness = round(0.62 + 0.38 * math.sqrt(amp), 2)
        key = (note.get('wavetype'), brightness, bool(track.get('mute')), bool(active))
        cached = self._note_color_cache.get(key)
        if cached is not None:
            return cached

        fill_color = self.shade_color(self.wavetype_color(key[0]), brightness)
        if key[2]:
            fill_color = self.shade_color(fill_color, 0.55)
        if not key[3]:
            fill_color = self.shade_color(fill_color, 0.60)
        # 深色块配白字、浅色块配黑字，保证音名看得清
        try:
            _c = fill_color.lstrip('#')
            _lum = (0.299 * int(_c[0:2], 16) + 0.587 * int(_c[2:4], 16)
                    + 0.114 * int(_c[4:6], 16)) / 255.0
        except (ValueError, IndexError):
            _lum = 1.0
        result = (fill_color, "#000000" if _lum > 0.55 else "#FFFFFF")
        self._note_color_cache[key] = result
        return result

    def _note_rect(self, note):
        """音符在画布上的 (x1, y1, x2, y2)——横向用缩放因子，纵向用缩放因子"""
        num_pitches = 96  # 总音高数量
        x1 = note['start'] * self.pixels_per_unit
        y1 = (num_pitches - 1 - note['pitch']) * self.pixels_per_pitch
        x2 = x1 + note['duration'] * self.pixels_per_unit
        y2 = y1 + self.pixels_per_pitch  # 使用纵向缩放的高度
        return x1, y1, x2, y2

    def _update_note_geometry(self, note, update_label=False):
        """只把单个音符挪到新位置（拖动时用，避免每帧重建全部音符图元）

        update_label=True 时同时刷新音名文字：改音高后块会动、字不会自己动，
        不刷就会一直顶着旧音名。
        """
        rect_id = note.get('rect_id')
        if not rect_id:
            return
        x1, y1, x2, y2 = self._note_rect(note)
        self.canvas.coords(rect_id, x1, y1, x2, y2)
        text_id = note.get('text_id')
        if text_id:
            self.canvas.coords(text_id, x1 + 5, y1 + 10)
            if update_label:
                self.canvas.itemconfig(
                    text_id, text=self.get_note_from_pitch(note['pitch']))

    def draw_notes(self):
        """绘制所有音符（填充 = 音色色 × 力度，边框 = 音轨色，静音轨变灰）"""
        self.canvas.delete("note")

        # 选中判定走 id 集合：音符是 dict，用 in/== 会做逐键深比较，多选时是 O(n×m)
        selected_ids = {id(n) for n in self.multi_selected_notes}
        single_id = id(self.selected_note) if self.selected_note else None
        outline_cache = {}

        for note in self.sequence:
            x1, y1, x2, y2 = self._note_rect(note)
            track_idx = note.get('track', 0)
            track = self.get_track(track_idx)
            fill_color, label_color = self._note_colors(
                note, track, active=(track_idx == self.current_track))

            # 根据选择状态设置边框颜色
            note_id = id(note)
            if note_id in selected_ids:
                outline_color = "purple"
                width = 2
            elif note_id == single_id:
                outline_color = "green"
                width = 2
            else:
                # 未选中时边框用音轨色，音色看填充、音轨看边框
                track_color = track.get('color', "#99FF99")
                outline_color = outline_cache.get(track_color)
                if outline_color is None:
                    outline_color = self.shade_color(track_color, 0.5)
                    outline_cache[track_color] = outline_color
                width = 1

            rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=fill_color,
                outline=outline_color,
                width=width,
                tags=("note", f"note_{note_id}", "notebox")
            )

            # 显示音名（使用新的音高范围）
            note_name = self.get_note_from_pitch(note['pitch'])  # 获取音名
            text = f"{note_name}"  # 格式：音色@音名

            text_id = self.canvas.create_text(
                x1 + 5, y1 + 10,
                text=text,
                anchor=tk.W,
                fill=label_color,
                tags=("note", f"note_{note_id}", "label")
            )

            # 保存图元 ID 到音符对象（拖动时据此做增量更新）
            note['rect_id'] = rect_id
            note['text_id'] = text_id

        # 全量重建后图元 ID 全换了，高亮差异表作废，下次重新刷一遍
        self._hl_state = None

    def refresh_track_highlight(self):
        """只按当前音轨刷新音符填充色，不重建图元

        换轨变的是“哪条轨亮”，位置、边框、选中态都没变，不必全量重绘：
        670 个音符全量重绘约 50 ms，逐个 itemconfigure 改色只要几毫秒。
        边框由 _hl_state 那套单独管，这里只动 fill，不会打架。
        """
        for note in self.sequence:
            rect_id = note.get('rect_id')
            if not rect_id:
                continue
            track_idx = note.get('track', 0)
            fill, label = self._note_colors(
                note, self.get_track(track_idx),
                active=(track_idx == self.current_track))
            try:
                self.canvas.itemconfigure(rect_id, fill=fill)
                text_id = note.get('text_id')
                if text_id:
                    self.canvas.itemconfigure(text_id, fill=label)
            except tk.TclError:
                return

    # ================================================== 效果器块（3.1）
    FX_COLORS = ["#7E57C2", "#26A69A", "#EF6C00", "#0288D1", "#C2185B", "#5D4037"]
    FX_LABELS = (("lowpass_1st", "一阶低通"), ("sustain", "延音"), ("reverb", "卷积混响"))

    def _new_effect_block(self, start, end, pitch_low, pitch_high):
        """造一个空的效果器块（三个效果默认都关，等用户在弹窗里勾）"""
        blk = {
            'id': self._fx_next_id,
            'start': float(start),
            'end': float(end),
            'pitch_low': int(pitch_low),
            'pitch_high': int(pitch_high),
            'color': self.FX_COLORS[(self._fx_next_id - 1) % len(self.FX_COLORS)],
            'timbres': None,        # None = 块内所有音色
            'fx': {
                'lowpass_1st': {'on': False},
                'sustain': {'on': False, 'value': 0.2},
                'reverb': {'on': False, 'ir': '', 'mix': 0.3},
            },
        }
        self._fx_next_id += 1
        return blk

    def _block_rect(self, blk):
        """效果器块在画布上的 (x1, y1, x2, y2)"""
        num_pitches = 96
        x1 = blk['start'] * self.pixels_per_unit
        x2 = blk['end'] * self.pixels_per_unit
        y1 = (num_pitches - 1 - blk.get('pitch_high', 95)) * self.pixels_per_pitch
        y2 = (num_pitches - blk.get('pitch_low', 0)) * self.pixels_per_pitch
        return x1, y1, x2, y2

    def _block_label(self, blk):
        """色块里那行小字：启用了哪些效果 + 音色过滤情况"""
        fx = blk.get('fx', {})
        names = [label for key, label in self.FX_LABELS if fx.get(key, {}).get('on')]
        text = f"FX{blk['id']}：" + ("/".join(names) if names else "空（点击设置）")
        if blk.get('timbres') is not None:
            text += f" | {len(blk['timbres'])} 种音色"
        return text

    def _block_covers(self, blk, note):
        """音符是否落在块内：起点在时间范围内、音高在行范围内"""
        return (blk['start'] - 1e-6 <= note['start'] < blk['end'] - 1e-6
                and blk.get('pitch_low', 0) <= note['pitch'] <= blk.get('pitch_high', 95))

    def effect_kwargs_for_note(self, note):
        """把覆盖该音符的所有效果器块折算成一组渲染参数（不改动音符本身）

        只返回“该音符该额外带上的东西”，播放和导出都调这一个函数，
        两边结果因此天然一致。
        """
        extra = {}
        if note.get('wavetype') == 'audio_sample':
            return extra          # 采样只受延音影响，低通/混响不参与
        wavetype = note.get('wavetype')
        for blk in self.effect_blocks:
            if not self._block_covers(blk, note):
                continue
            allowed = blk.get('timbres')
            if allowed is not None and wavetype not in allowed:
                continue
            fx = blk.get('fx', {})
            if fx.get('lowpass_1st', {}).get('on'):
                extra['lowpass_1st'] = True
            if fx.get('sustain', {}).get('on'):
                try:
                    extra['sustain_ratio'] = max(0.0, min(1.0, float(fx['sustain'].get('value', 0.0))))
                except (TypeError, ValueError):
                    pass
            reverb = fx.get('reverb', {})
            if reverb.get('on') and reverb.get('ir'):
                extra['reverb_ir'] = reverb['ir']
                try:
                    extra['reverb_mix'] = max(0.0, min(1.0, float(reverb.get('mix', 0.3))))
                except (TypeError, ValueError):
                    extra['reverb_mix'] = 0.3
        return extra

    def merged_note_kwargs(self, note):
        """音符自身的 kwargs + 效果器块折算出来的参数（块覆盖单个音符的设置）"""
        kwargs = dict(note.get('kwargs') or {})
        kwargs.update(self.effect_kwargs_for_note(note))
        return kwargs

    def draw_effect_blocks(self):
        """画效果器色块（半透明网点填充 + 虚线边框 + 左上角小字），压在音符下面"""
        self.canvas.delete("fxblock")
        if not self.effect_blocks_visible:
            return
        for blk in self.effect_blocks:
            x1, y1, x2, y2 = self._block_rect(blk)
            color = blk.get('color') or self.FX_COLORS[0]
            selected = blk is self.selected_block
            rect_id = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=color, stipple="gray12",
                outline=color, width=3 if selected else 2,
                dash=(8, 4),
                tags=("fxblock", f"fxblock_{blk['id']}")
            )
            text_id = self.canvas.create_text(
                x1 + 6, y2 - 9,
                text=self._block_label(blk),
                anchor=tk.W,
                fill=self.shade_color(color, 0.55),
                font=("", 7, "bold"),
                tags=("fxblock", "fxlabel")
            )
            blk['rect_id'] = rect_id
            blk['label_id'] = text_id
        # 色块永远在音符之下，否则会盖住音名
        try:
            self.canvas.tag_lower("fxblock", "note")
        except tk.TclError:
            pass

    def _update_block_geometry(self, blk):
        """只挪一个色块（拖动/改长度时用，不重建全部图元）"""
        rect_id = blk.get('rect_id')
        if not rect_id:
            self.draw_effect_blocks()
            return
        x1, y1, x2, y2 = self._block_rect(blk)
        self.canvas.coords(rect_id, x1, y1, x2, y2)
        label_id = blk.get('label_id')
        if label_id:
            self.canvas.coords(label_id, x1 + 6, y2 - 9)
            self.canvas.itemconfig(label_id, text=self._block_label(blk))

    def _block_at(self, x, y):
        """命中测试：返回光标下的效果器块（后建的在上层，倒序找）"""
        if not self.effect_blocks_visible:
            return None
        for blk in reversed(self.effect_blocks):
            x1, y1, x2, y2 = self._block_rect(blk)
            if x1 <= x <= x2 and y1 <= y <= y2:
                return blk
        return None

    def block_pitch_span(self, timbres):
        """一组音色在整首曲子里占据的最低/最高音高（含端点）

        timbres 为 None 表示“不限制音色”（所有非采样音符）。
        效果器色块的上下界就靠它来贴：弹窗里勾了哪几种音色，
        色块就贴到这些音色的最高音和最低音，与当前选中了哪些音符无关。
        """
        pitches = []
        for n in self.sequence:
            wt = n.get('wavetype')
            if wt == 'audio_sample':      # 采样不参与低通/混响，不参与取范围
                continue
            if timbres is not None and wt not in timbres:
                continue
            pitches.append(n['pitch'])
        if not pitches:
            return None, None
        low, high = min(pitches), max(pitches)
        # 单一音高时色块只有一行（约 8 px），看不清也够不着右边界热区，补到 3 行
        need = 3 - (high - low + 1)
        if need > 0:
            high = min(95, high + need // 2)
            low = max(0, low - (need - need // 2))
            while high - low + 1 < 3:
                if high < 95:
                    high += 1
                elif low > 0:
                    low -= 1
                else:
                    break
        return int(low), int(high)

    def fit_block_pitch(self, blk, timbres):
        """把色块上下界贴到 timbres 这组音色的音域

        只改 pitch_low/pitch_high（也就是色块大小），不动 fx、时间范围、
        音色过滤这些作用逻辑；因为取的是该音色自身的 min/max，
        该音色的音符一个都不会被排除，效果覆盖范围保持原样。
        """
        low, high = self.block_pitch_span(timbres)
        if low is None:
            return False
        blk['pitch_low'], blk['pitch_high'] = low, high
        return True

    def add_effect_block(self):
        """新建效果器块：有选中音符就框住它们，否则盖住光标处 16 个十六分音符

        上下界先按“不限制音色”贴到全曲音域；弹窗里勾选音色后
        （见 EffectBlockDialog.on_timbre_toggle）会自动收窄到所选音色的音域。
        """
        notes = list(self.multi_selected_notes)
        if not notes and self.selected_note:
            notes = [self.selected_note]
        if notes:
            start = min(n['start'] for n in notes) - 1.0
            end = max(n['start'] + n['duration'] for n in notes) + 1.0
        else:
            start = self.snap_units(self.cursor_unit)
            end = start + 16.0
        if end <= start:
            end = start + 1.0
        low, high = self.block_pitch_span(None)
        if low is None:
            low, high = 0, 95
        self.save_state()
        blk = self._new_effect_block(max(0.0, start), end, low, high)
        self.effect_blocks.append(blk)
        self.selected_block = blk
        self.effect_blocks_visible = True
        self.draw_effect_blocks()
        self.status_var.set(
            f"已新建效果器块 FX{blk['id']}（{start:.2f}–{end:.2f}，"
            f"{self.get_note_from_pitch(low)}–{self.get_note_from_pitch(high)}）")
        self.open_block_dialog(blk)

    def edit_selected_block(self):
        if not self.selected_block:
            messagebox.showinfo("提示", "先点击画布上的效果器色块选中它")
            return
        self.open_block_dialog(self.selected_block)

    def delete_selected_block(self):
        if not self.selected_block:
            messagebox.showinfo("提示", "先点击画布上的效果器色块选中它")
            return
        blk = self.selected_block
        self.save_state()
        self.effect_blocks = [b for b in self.effect_blocks if b is not blk]
        self.selected_block = None
        self.draw_effect_blocks()
        self.status_var.set(f"已删除效果器块 FX{blk['id']}")

    def toggle_effect_blocks(self):
        self.effect_blocks_visible = not self.effect_blocks_visible
        self.draw_effect_blocks()
        self.status_var.set("效果器色块：" + ("显示" if self.effect_blocks_visible else "隐藏"))

    def _start_block_press(self, blk, x, y):
        """按下色块：记住现场；松手时没移动就当作“点开设置”"""
        self._cancel_rect_job()
        if self.selected_block is not blk:
            self.selected_block = blk
            self.draw_effect_blocks()
        x1, y1, x2, y2 = self._block_rect(blk)
        self._block_press = {
            'block': blk, 'x': x, 'y': y, 'moved': False,
            'start': blk['start'], 'end': blk['end'],
        }
        self.dragging = True
        # 靠右边界 8 px 内拖 = 改长度，其余位置拖 = 整体平移
        self.drag_type = 'block_end' if (x2 - x) <= 8 else 'block_move'
        self._pending_snapshot = True
        self.status_var.set(f"效果器块 FX{blk['id']}：拖动可移动/改长度，单击打开设置")

    def open_block_dialog(self, blk):
        EffectBlockDialog(self, blk)

    def get_note_from_pitch(self, pitch):
        """将音高转换为音符名称（使用新的音高范围）"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # 计算实际MIDI音高（C0=0, C#0=1, ..., B7=95）
        midi_pitch = pitch + 12  # C0在MIDI系统中是12
        
        # 计算八度和音符名称
        octave = (midi_pitch // 12) - 1  # 减去1修正八度显示
        note_idx = midi_pitch % 12
        note = notes[note_idx]
        
        return f"{note}{octave}"

    def get_pitch_from_note(self, note):
        """将音符名称转换为音高（使用新的音高范围）"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        try:
            note_name = note[:-1]
            octave = int(note[-1])
            
            # 计算MIDI音高
            midi_pitch = (octave * 12) + notes.index(note_name)
            
            # 转换为内部音高表示（0-95）
            return midi_pitch - 12  # C0在MIDI系统中是12
        except:
            return 48 - 12  # 默认中央C（C4在MIDI中是60，转换为内部表示48）
      
    def zoom(self, factor):
        """缩放视图 - 同时缩放横向和纵向"""
        if factor == 0:  # 重置缩放
            self.pixels_per_unit = 40.0
            self.pixels_per_pitch = 20.0
        else:
            self.pixels_per_unit *= factor
            self.pixels_per_pitch *= factor
            
        # 限制缩放范围
        self.pixels_per_unit = max(10.0, min(100.0, self.pixels_per_unit))
        self.pixels_per_pitch = max(10.0, min(50.0, self.pixels_per_pitch))
        
        # 重新绘制
        self.draw_grid()
        self.status_var.set(f"缩放: 横向{self.pixels_per_unit:.1f}px/单位, 纵向{self.pixels_per_pitch:.1f}px/音高")
    
    def get_canvas_coords(self, event):
        """获取考虑滚动偏移的绝对画布坐标"""
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        return x, y

    def clear_multi_selection(self):
        """清除多选选择"""
        self.multi_selected_notes = []
        self.draw_notes()  # 重新绘制以清除多选高亮
        self.status_var.set("已清除选择")

    def select_notes_in_rect(self, x1, y1, x2, y2):
        """选择指定矩形区域内的所有音符"""
        # 清除之前的选择
        self.multi_selected_notes = []

        # 确保坐标顺序正确
        rect_x1 = min(x1, x2)
        rect_y1 = min(y1, y2)
        rect_x2 = max(x1, x2)
        rect_y2 = max(y1, y2)

        # 查找在矩形区域内的音符（直接用音符自己的坐标算，不再逐个问画布要 bbox）
        hits = []
        for note in self.sequence:
            if not note.get('rect_id'):
                continue
            bx1, by1, bx2, by2 = self._note_rect(note)
            if (rect_x1 <= bx1 <= rect_x2 or rect_x1 <= bx2 <= rect_x2 or
                    (bx1 <= rect_x1 and bx2 >= rect_x2)):
                if (rect_y1 <= by1 <= rect_y2 or rect_y1 <= by2 <= rect_y2 or
                        (by1 <= rect_y1 and by2 >= rect_y2)):
                    hits.append(note)
        self.multi_selected_notes = hits

        self._apply_highlight()

        # 更新状态栏
        self.status_var.set(f"已选择 {len(self.multi_selected_notes)} 个音符")

    def _apply_highlight(self):
        """按当前选区给音符上边框高亮。

        旧写法对每个音符都发一次 itemconfig（1000 音符 = 1000 次 Tcl 调用，
        实测 11 ms，2000 音符涨到 28 ms）。这里只改动“状态真的变了”的图元，
        并记住上次的映射；只有全量重绘过（_hl 失效）才走一遍全量。
        """
        want = {}
        for note in self.multi_selected_notes:
            rid = note.get('rect_id')
            if rid:
                want[rid] = 2                      # 2 = 多选（紫）
        if self.selected_note:
            rid = self.selected_note.get('rect_id')
            if rid and rid not in want:
                want[rid] = 1                      # 1 = 单选（绿）

        prev = self._hl_state
        if prev is None:                           # 无记录：全量刷一遍
            for note in self.sequence:
                rid = note.get('rect_id')
                if not rid:
                    continue
                style = want.get(rid, 0)
                if style == 2:
                    self.canvas.itemconfig(rid, outline="purple", width=2)
                elif style == 1:
                    self.canvas.itemconfig(rid, outline="green", width=2)
                else:
                    self.canvas.itemconfig(rid, outline="black", width=1)
        else:
            for rid in want:
                if prev.get(rid) != want[rid]:
                    if want[rid] == 2:
                        self.canvas.itemconfig(rid, outline="purple", width=2)
                    else:
                        self.canvas.itemconfig(rid, outline="green", width=2)
            for rid in prev:
                if want.get(rid, 0) != prev[rid]:
                    self.canvas.itemconfig(rid, outline="black", width=1)
        self._hl_state = want

    def move_selected_notes(self, offset):
        """移动选中的音符"""
        if not self.multi_selected_notes and not self.selected_note:
            return
        
        # 保存当前状态到撤销栈
        self.save_state()
        
        # 确定要移动的音符集合
        notes_to_move = self.multi_selected_notes
        if not notes_to_move and self.selected_note:
            notes_to_move = [self.selected_note]
        
        # 移动所有选中的音符
        for note in notes_to_move:
            note['start'] = max(0, note['start'] + offset)
        
        # 重新绘制
        self.draw_notes()
        self.status_var.set(f"已移动 {len(notes_to_move)} 个音符")
    
    def copy_selected_note(self, event=None, offset=1.0):
        """复制选中的音符（支持多选）"""
        # 确定要复制的音符集合
        notes_to_copy = []
        if self.multi_selected_notes:
            notes_to_copy = self.multi_selected_notes
        elif self.selected_note:
            notes_to_copy = [self.selected_note]
        else:
            return
            
        # 保存当前状态到撤销栈
        self.save_state()
        
        new_notes = []
        for note in notes_to_copy:
            # 深拷贝选中的音符
            new_note = copy.deepcopy(note)
            
            # 偏移新音符的起始位置
            new_note['start'] += offset
            
            # 添加到音序
            self.sequence.append(new_note)
            new_notes.append(new_note)
        
        # 更新选择状态（选中新复制的音符）
        self.multi_selected_notes = new_notes
        self.selected_note = None  # 清除单点选择
        
        # 重新绘制
        self.draw_notes()
        self.status_var.set(f"已复制 {len(new_notes)} 个音符")
    
    def move_selected_notes_with_input(self):
        """通过对话框移动选中的音符"""
        # 确定要移动的音符集合
        notes_to_move = []
        if self.multi_selected_notes:
            notes_to_move = self.multi_selected_notes
        elif self.selected_note:
            notes_to_move = [self.selected_note]
        else:
            messagebox.showinfo("提示", "请先选择要移动的音符")
            return
            
        # 创建对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("移动音符")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="输入移动量（十六分音符）:").pack(pady=10)
        
        offset_var = tk.StringVar(value="0.0")
        entry = ttk.Entry(dialog, textvariable=offset_var, width=10)
        entry.pack(pady=5)
        entry.select_range(0, tk.END)
        entry.focus_set()
        
        status_var = tk.StringVar(value="")
        ttk.Label(dialog, textvariable=status_var).pack(pady=5)
        
        def apply_move():
            try:
                offset = float(offset_var.get())
                self.save_state()
                
                # 移动所有选中的音符
                for note in notes_to_move:
                    note['start'] = max(0, note['start'] + offset)
                
                # 重新绘制
                self.draw_notes()
                self.status_var.set(f"已移动 {len(notes_to_move)} 个音符")
                dialog.destroy()
                
            except ValueError:
                status_var.set("请输入有效的数字")
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="确定", command=apply_move).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
        
        # 绑定回车键
        dialog.bind("<Return>", lambda e: apply_move())

    def move_selected_notes_left(self, event=None):
        """向左移动选中的音符（0.5单位）"""
        self.move_selected_notes(-0.5)
    
    def move_selected_notes_right(self, event=None):
        """向右移动选中的音符（0.5单位）"""
        self.move_selected_notes(0.5)

    def transpose_selected(self):
        """对选中的多个音符进行同时变调"""
        # 确定要变调的音符集合
        notes_to_transpose = []
        if self.multi_selected_notes:
            notes_to_transpose = self.multi_selected_notes
        elif self.selected_note:
            notes_to_transpose = [self.selected_note]
        else:
            messagebox.showinfo("提示", "请先选择要变调的音符")
            return
            
        # 创建对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("多选变调")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="输入要升高的半音数（正数升高，负数降低）:").pack(pady=10)
        
        semitones_var = tk.StringVar(value="0")
        entry = ttk.Entry(dialog, textvariable=semitones_var, width=10)
        entry.pack(pady=5)
        entry.select_range(0, tk.END)
        entry.focus_set()
        
        status_var = tk.StringVar(value="")
        ttk.Label(dialog, textvariable=status_var).pack(pady=5)
        
        def apply_transpose():
            try:
                semitones = int(semitones_var.get())
                if not notes_to_transpose:
                    status_var.set("没有音符可调整")
                    return
                    
                # 保存当前状态到撤销栈
                self.save_state()
                
                # 执行变调操作
                for note in notes_to_transpose:
                    new_pitch = note['pitch'] + semitones
                    # 确保音高在0-95范围内
                    note['pitch'] = max(0, min(95, new_pitch))
                
                # 重新绘制音符
                self.draw_notes()
                self.status_var.set(f"已对 {len(notes_to_transpose)} 个音符变调 {semitones} 个半音")
                dialog.destroy()
                
            except ValueError:
                status_var.set("请输入有效的整数")
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="应用", command=apply_transpose).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
        
        # 绑定回车键
        dialog.bind("<Return>", lambda e: apply_transpose())

    def _note_at(self, x, y):
        """取画布坐标下的音符：用 find_overlapping 直接问画布，不遍历全部图元
        旧写法是遍历所有 note 图元逐个 bbox，1000 音符时一次点击要 40 ms。
        重叠时取最上层的那个（后画的在上面，与所见一致）
        """
        hit = None
        for item in self.canvas.find_overlapping(x - 1, y - 1, x + 1, y + 1):
            if self.canvas.type(item) == "rectangle":
                hit = item
        if hit is None:
            return None
        for note in self.sequence:
            if note.get('rect_id') == hit:
                return note
        return None

    def _context_targets(self, note):
        """右键某一项操作的作用对象
        点中的音符若在框选里就按整组处理，否则只作用于它自己
        """
        if self.multi_selected_notes and id(note) in {
                id(n) for n in self.multi_selected_notes}:
            return list(self.multi_selected_notes)
        return [note]

    def _focus_single_note(self, note):
        """把某个音符设为唯一选中
        旧的框选若不含这个音符就作废——否则 Ctrl+C / 复制 / 删除 / 换轨
        会作用在上一次框选的音符上，而不是刚点的这个
        """
        if self.multi_selected_notes and id(note) not in {
                id(n) for n in self.multi_selected_notes}:
            self.multi_selected_notes = []
            self.selected_note = note
            self.draw_notes()
        self.selected_note = note

    def on_canvas_configure(self, event):
        """画布大小改变时重新绘制网格（防抖：拖动窗口边缘时不要每个像素都重绘）"""
        if self._configure_job:
            self.root.after_cancel(self._configure_job)
        self._configure_job = self.root.after(150, self._redraw_after_configure)

    def _redraw_after_configure(self):
        """<Configure> 防抖到期后的实际重绘"""
        self._configure_job = None
        self.draw_grid()

    def on_canvas_click(self, event):
        """画布点击事件（修正纵坐标计算）"""
        x, y = self.get_canvas_coords(event)

        # 3.0.0：记录光标位置（按吸附格），作为粘贴落点
        self.cursor_unit = self.snap_units(x / self.pixels_per_unit)

        num_pitches = 96  # 总音高数量
        
        # 修改：计算音高编号使用纵向缩放因子
        pitch = num_pitches - 1 - int(y / self.pixels_per_pitch)
        pitch = max(0, min(95, pitch))
        
        # 查找点击的音符
        clicked_note = self._note_at(x, y)
        # 3.0.0：框选统一由“按住 Alt 拖动”触发，没有常驻的多选模式开关。
        # 直接读鼠标事件的 Alt 位（Windows 为 0x20000，X11 为 Mod1=0x8），
        # 不依赖 Alt 的按下/松开事件，也就不会出现“进入后退出不来”的状态残留。
        # 注意按平台取位，见文件顶部 ALT_MASK 的注释（0x8 在 Windows 上会被 NumLock 点亮）。
        alt_held = bool(event.state & ALT_MASK)
        if alt_held:
            # 框选 - 开始绘制选择框
            self._cancel_rect_job()
            self.dragging = True
            self.drag_type = 'multi'
            self.drag_start_x, self.drag_start_y = self.get_canvas_coords(event)
            
            # 创建选择框（初始大小为一个点）：亮蓝实线宽 2 + 淡蓝网点填充，压在网格上才看得清
            self.selection_rect_id = self.canvas.create_rectangle(
                self.drag_start_x, self.drag_start_y, 
                self.drag_start_x, self.drag_start_y,
                outline="#0B6FE0", width=2, dash=(6, 4),
                fill="#5AA9F0", stipple="gray25",
                tags="selection_rect"
            )
            return

        # 3.1：没有命中音符时，看看是不是点在效果器色块上
        # （色块画在音符下面，所以音符优先；点在色块空白处才进入色块操作）
        if not clicked_note:
            blk = self._block_at(x, y)
            if blk is not None:
                self._start_block_press(blk, x, y)
                return

        # 处理音符选择
        if clicked_note:
            self._focus_single_note(clicked_note)
            # 下面会把这一项临时涂成 red/blue/green，记进高亮差异表，
            # 免得下次框选时把它当成“已经是黑的”而漏刷
            if self._hl_state is not None and clicked_note.get('rect_id'):
                self._hl_state[clicked_note['rect_id']] = 3
            # 撤销点延后到真正发生位移时再写（见 on_canvas_drag）
            self._pending_snapshot = True
            
            # 检查是否是特殊拖拽
            if event.state & 0x0001:  # Shift键 - 调整持续时间
                self.dragging = True
                self.drag_type = 'duration'
                # 使用转换后的坐标作为起始点
                self.drag_start_x = x
                self.drag_start_value = clicked_note['duration']
                
                # 高亮显示
                self.canvas.itemconfig(clicked_note['rect_id'], outline="red", width=2)
                
            elif event.state & 0x0004:  # Ctrl键 - 调整音高
                self.dragging = True
                self.drag_type = 'pitch'
                # 使用转换后的坐标作为起始点
                self.drag_start_y = y
                self.drag_start_value = clicked_note['pitch']
                
                # 高亮显示
                self.canvas.itemconfig(clicked_note['rect_id'], outline="blue", width=2)
                
            else:  # 正常拖拽 - 调整起始时间
                self.dragging = True
                self.drag_type = 'start'
                # 使用转换后的坐标作为起始点
                self.drag_start_x = x
                self.drag_start_value = clicked_note['start']
                
                # 高亮显示
                self.canvas.itemconfig(clicked_note['rect_id'], outline="green", width=2)
                
        else:
            self.selected_note = None
            # 清除所有高亮（按 tag 一次设置，别循环一千次 itemconfig）
            self.canvas.itemconfig("notebox", outline="black", width=1)
            self._hl_state = {}     # 全部已归默认，差异表清空
    
    def on_canvas_drag(self, event):
        """画布拖拽事件（使用纵向缩放）"""
        # 获取转换后的画布坐标
        x, y = self.get_canvas_coords(event)

        # 框选优先：此时可能没有 selected_note，不能走下面的 early return
        if self.dragging and self.drag_type == 'multi':
            # 鼠标事件（上百 Hz）远多于屏幕刷新：攒到下一帧再动一次选框，
            # 否则每来一个 motion 都要让 Tk 重排一次画布，音符越多越拖手
            self._rect_pending = (x, y)
            if self._rect_job is None:
                self._rect_job = self.root.after(12, self._flush_selection_rect)
            return

        # 3.1：拖动效果器色块（同样不能落到下面的 selected_note 判断里）
        if self.dragging and self.drag_type in ('block_move', 'block_end') and self._block_press:
            press = self._block_press
            blk = press['block']
            if self._pending_snapshot:
                self.save_state()
                self._pending_snapshot = False
            dx = (x - press['x']) / self.pixels_per_unit
            if abs(x - press['x']) > 2 or abs(y - press['y']) > 2:
                press['moved'] = True
            if self.drag_type == 'block_move':
                length = max(0.5, press['end'] - press['start'])
                new_start = self.snap_units(max(0.0, press['start'] + dx))
                blk['start'] = new_start
                blk['end'] = new_start + length
            else:
                new_end = self.snap_units(press['end'] + dx, minimum=blk['start'] + 0.5)
                blk['end'] = max(blk['start'] + 0.5, new_end)
            self._update_block_geometry(blk)
            self.status_var.set(
                f"效果器块 FX{blk['id']}：{blk['start']:.2f} – {blk['end']:.2f}")
            return

        if not self.dragging or not self.selected_note:
            return

        try:
            # 第一次真正改动音符时才落撤销点（点击只选中不拖动不留快照）
            if self._pending_snapshot:
                self.save_state()
                self._pending_snapshot = False

            label_changed = False

            if self.drag_type == 'start':
                # 计算起始时间变化
                delta_x = x - self.drag_start_x
                delta_time = delta_x / self.pixels_per_unit
                
                # 3.0.0：按当前吸附档位取整（关闭吸附时为 0.25 单位的细调）
                new_start = self.drag_start_value + delta_time
                new_start = self.snap_units(new_start)
                
                self.selected_note['start'] = new_start
                
            elif self.drag_type == 'duration':
                # 计算持续时间变化
                delta_x = x - self.drag_start_x
                delta_duration = delta_x / self.pixels_per_unit
                
                # 3.0.0：长度同样吸附，最短 = 当前格子
                step = self.snap_value() or 0.25
                new_duration = self.drag_start_value + delta_duration
                new_duration = self.snap_units(new_duration, minimum=step)
                
                self.selected_note['duration'] = new_duration
                
            elif self.drag_type == 'pitch':
                # 计算音高变化 - 使用纵向缩放因子
                delta_y = y - self.drag_start_y
                delta_pitch = -delta_y / self.pixels_per_pitch  # 使用纵向缩放
                
                # 更新音高（步长：1个半音）
                new_pitch = self.drag_start_value + round(delta_pitch)
                new_pitch = max(0, min(95, new_pitch))
                
                # 音高真的变了才需要重写音名（Ctrl+拖动变调）
                label_changed = new_pitch != self.selected_note['pitch']
                self.selected_note['pitch'] = new_pitch
            
            # 只挪动被拖的这个音符（整画布重绘随音符数线性变慢，1000 音符要 100+ ms）
            self._update_note_geometry(
                self.selected_note,
                update_label=(self.drag_type == 'pitch' and label_changed))
            
            # 更新状态栏
            status_text = f"编辑: {self.selected_note['wavetype']} @ {self.get_note_from_pitch(self.selected_note['pitch'])}"
            status_text += f" | 起始: {self.selected_note['start']:.2f} | 时长: {self.selected_note['duration']:.2f}"
            self.status_var.set(status_text)
            
        except Exception as e:
            self.status_var.set(f"错误: {str(e)}")
    
    def _flush_selection_rect(self):
        """把节流攒下的最新鼠标位置真正写到选框上（每帧至多一次）"""
        self._rect_job = None
        if not self.selection_rect_id or self._rect_pending is None:
            return
        x, y = self._rect_pending
        self.canvas.coords(self.selection_rect_id,
                           self.drag_start_x, self.drag_start_y, x, y)

    def _cancel_rect_job(self):
        """框选结束时清掉挂起的刷新，避免松手后又动一下旧选框"""
        if self._rect_job is not None:
            try:
                self.root.after_cancel(self._rect_job)
            except Exception:
                pass
            self._rect_job = None
        self._rect_pending = None

    def on_canvas_release(self, event):
        """画布释放事件"""
        # 3.1：色块松手——没移动就是单击，打开设置弹窗
        if self.dragging and self.drag_type in ('block_move', 'block_end'):
            press = self._block_press
            self._block_press = None
            moved = bool(press and press.get('moved'))
            self.dragging = False
            self.drag_type = None
            self._pending_snapshot = False
            if press and not moved:
                self.open_block_dialog(press['block'])
            return

        if self.dragging and self.drag_type == 'multi':
            self._cancel_rect_job()
            # 完成多选
            x, y = self.get_canvas_coords(event)
            self.select_notes_in_rect(self.drag_start_x, self.drag_start_y, x, y)
            
            # 确保删除选择框
            if self.selection_rect_id:
                self.canvas.delete(self.selection_rect_id)
                self.selection_rect_id = None
        
        self.dragging = False
        self.drag_type = None
        self._pending_snapshot = False
        
        # 清除高亮
        if not self.multi_selected_notes and self.selected_note:
            self.canvas.itemconfig(self.selected_note['rect_id'], outline="black", width=1)
            self._hl_state = {}     # 该项已归默认，差异表清空

    def on_right_click(self, event):
        """右键点击事件 - 显示上下文菜单"""
        menu = tk.Menu(self.root, tearoff=0)
        
        # 查找点击的音符
        x, y = self.get_canvas_coords(event)
        clicked_note = self._note_at(x, y)
        
        # 如果有音符被点击
        if clicked_note:
            # 右键点的音符若不在当前框选内，旧框选作废（见 _focus_single_note）
            self._focus_single_note(clicked_note)
            self.canvas.itemconfig(clicked_note['rect_id'], outline="yellow", width=2)
            
            # 单项操作按“点中的音符属于哪个选区”决定作用范围，并显式绑定对象，
            # 不再依赖全局选中状态（否则会作用在上一次选的音符上）
            targets = self._context_targets(clicked_note)
            n_targets = len(targets)
            menu.add_command(
                label=f"复制 {n_targets} 个音符" if n_targets > 1 else "复制音符",
                command=lambda: self.copy_note(targets[0]))
            menu.add_separator()
            menu.add_command(label="编辑音符", command=lambda: self.open_note_editor(clicked_note))
            menu.add_command(
                label=f"删除 {n_targets} 个音符" if n_targets > 1 else "删除音符",
                command=lambda: self.delete_notes(targets))
            menu.add_separator()
            # 3.0.0：剪贴板 / 换轨（与上面“复制 N 个音符”区分：这里只进剪贴板，Ctrl+V 才落地）
            menu.add_command(label="复制到剪贴板 (Ctrl+C)", command=self.copy_to_clipboard)
            menu.add_command(label="粘贴到此处 (Ctrl+V)",
                             command=lambda: self.paste_at(x))
            track_sub = tk.Menu(menu, tearoff=0)
            for i, track in enumerate(self.tracks):
                track_sub.add_command(
                    label=f"{i + 1}. {track['name']}",
                    command=lambda i=i: self.assign_selection_to_track(i))
            menu.add_cascade(label="移到音轨", menu=track_sub)
        else:
            menu.add_command(label="在此位置添加音符", 
                    command=lambda: self.add_note_at_position(x, y))
            menu.add_command(label="粘贴到此处 (Ctrl+V)",
                             command=lambda: self.paste_at(x))
        
        # 框选（Alt 拖动）后的额外选项
        if self.multi_selected_notes:
            menu.add_separator()
            menu.add_command(label="多选删除", command=self.delete_selected_notes)
            menu.add_command(label="多选变调", command=self.transpose_selected)
            # 新增：统一设定振幅选项
            menu.add_command(label="统一设定振幅", command=self.unify_amplitude_for_selected)
            menu.add_command(label="向左移动所有选中音符", command=self.move_selected_notes_left)
            menu.add_command(label="向右移动所有选中音符", command=self.move_selected_notes_right)
            menu.add_separator()
            menu.add_command(label="清除选择", command=self.clear_multi_selection)
        
        # 显示菜单
        menu.post(event.x_root, event.y_root)

    def on_double_click(self, event):
        """双击事件 - 打开音符编辑器（修正音高计算）"""
        x, y = self.get_canvas_coords(event)

        # 双击落在效果器色块的空白处：不新增音符。色块是单击就打开设置的，
        # 双击多半是手快点重了，落到“添加音符”上会莫名其妙多出一个音符
        if not self._note_at(x, y) and self._block_at(x, y) is not None:
            return

        num_pitches = 96 
        if self.selected_note:
            self.open_note_editor(self.selected_note)
        else:
            # 计算音高编号（3.0.0：改用纵向缩放因子，原来写死 20 会错位）
            pitch = num_pitches - 1 - int(y / self.pixels_per_pitch)
            pitch = max(0, min(95, pitch))  # 限制在0-95范围内
            start = self.snap_units(x / self.pixels_per_unit)
            self.cursor_unit = start
            track = self.get_track(self.current_track)

            # 默认音符参数
            new_note = {
                'wavetype': track['wavetype'],
                'start': start,
                'duration': 4.0,  # 默认1拍（4个十六分音符）
                'pitch': pitch,
                'start_amp': 0.7,
                'amp_change_rate': 0.0,
                'kwargs': {},
                'track': self.current_track,
            }
            
            # 打开编辑器
            self.open_note_editor(new_note, is_new=True)
    
    def open_note_editor(self, note, is_new=False):
        """打开音符编辑器窗口

        3.0.1：编辑已有音符后也要重绘。音符块的颜色是画的时候按 wavetype 算的
        （见 _note_colors），只改数据不重绘，音符就会一直顶着旧音色色块。
        """
        # 编辑器是直接就地改 note 这个 dict 的，改完就找不回旧值了，
        # 所以改动前先自己留一份；值都是标量/整个换掉的 kwargs，浅拷贝够用
        before = None if is_new else dict(note)

        editor = NoteEditor(self.root, note, self)
        self.root.wait_window(editor)

        if is_new:
            if editor.saved:
                # 添加新音符
                self.save_state()
                self.sequence.append(note)
                self.selected_note = note
                self.draw_notes()
            return

        if not editor.saved or dict(note) == before:
            return

        # 撤销栈里要存“改之前”的工程状态：临时把旧值塞回去拍快照，再写回新值
        after = dict(note)
        note.clear()
        note.update(before)
        self.save_state()
        note.update(after)

        self.selected_note = note
        self.draw_notes()
        self.status_var.set(f"音符已更新（音色 {note.get('wavetype')}）")
    
    def segment_copy_paste(self):
        """编辑：按时间段选中音符，并把这批复刻到新的起点"""
        SegmentCopyDialog(self)

    def add_note(self):
        """添加新音符"""
        track = self.get_track(self.current_track)
        new_note = {
            'wavetype': track['wavetype'],
            'start': self.snap_units(self.cursor_unit),
            'duration': 4.0,
            'pitch': 48,
            'start_amp': 0.5,
            'amp_change_rate': 0.0,  # 关键修改：默认振幅变化率为0
            'kwargs': {},
            'track': self.current_track,
        }
        self.open_note_editor(new_note, is_new=True)
    
    def add_note_at_position(self, canvas_x, canvas_y):
        """在指定位置添加新音符"""
        # 3.0.0：起点吸附到网格，默认长度 = 一拍
        start_time = self.snap_units(canvas_x / self.pixels_per_unit)

        num_pitches = 96  # 总音高数量
        
        # 计算音高编号使用纵向缩放因子
        pitch = num_pitches - 1 - int(canvas_y / self.pixels_per_pitch)
        pitch = max(0, min(95, pitch))

        self.cursor_unit = start_time
        track = self.get_track(self.current_track)
        
        # 默认音符参数 - 关键修改：更新数据结构
        new_note = {
            'wavetype': track['wavetype'],
            'start': start_time,
            'duration': 4.0,  # 默认1拍（4个十六分音符）
            'pitch': pitch,
            'start_amp': 0.7,
            'amp_change_rate': 0.0,  # 关键修改：添加振幅变化率
            'kwargs': {},
            'track': self.current_track,
        }
        
        # 打开编辑器
        self.open_note_editor(new_note, is_new=True)

    def delete_notes(self, notes):
        """删除指定的音符（不经过选区状态，供右键菜单等直接指定对象）"""
        notes = [n for n in notes if n]
        if not notes:
            return
        # 保存当前状态到撤销栈
        self.save_state()

        # 按对象身份过滤，别用 sequence.remove（音符是 dict，会做逐键深比较）
        drop = {id(n) for n in notes}
        self.sequence = [n for n in self.sequence if id(n) not in drop]

        # 清除选择状态
        if self.selected_note is not None and id(self.selected_note) in drop:
            self.selected_note = None
        if self.multi_selected_notes:
            self.multi_selected_notes = [n for n in self.multi_selected_notes
                                         if id(n) not in drop]

        # 重新绘制
        self.draw_notes()
        self.status_var.set(f"已删除 {len(notes)} 个音符")

    def delete_selected_notes(self, event=None):
        """删除选中的音符（支持多选）"""
        # 3.0.0：焦点在输入框时，Delete 应该删字符而不是删音符
        if event is not None and self._text_has_focus():
            return
        # 确定要删除的音符集合
        notes_to_delete = self.multi_selected_notes or (
            [self.selected_note] if self.selected_note else [])
        
        if not notes_to_delete:
            self.status_var.set("没有选中任何音符")
            return
        
        self.delete_notes(notes_to_delete)

    def delete_note(self):
        """删除音符（兼容旧版本，实际调用多选删除）"""
        self.delete_selected_notes()
    
    def _snapshot(self):
        """当前工程快照（音轨里的 tk 变量不进快照）"""
        return {
            'sequence': copy.deepcopy(self.sequence),
            'tracks': [{k: v for k, v in track.items() if not k.startswith('_')}
                       for track in self.tracks],
            # 效果器块：去掉画布图元 id（快照恢复后图元会重建，留着是脏的）
            'effect_blocks': [{k: v for k, v in copy.deepcopy(blk).items()
                               if k not in ('rect_id', 'label_id')}
                              for blk in self.effect_blocks],
        }

    _TRACK_DATA_KEYS = ('name', 'wavetype', 'color', 'volume', 'pan', 'mute', 'solo')

    def _tracks_differ(self, new_tracks):
        """比较音轨数据是否真的变了（忽略 _mute_var 之类的控件变量）"""
        if len(new_tracks) != len(self.tracks):
            return True
        for old, new in zip(self.tracks, new_tracks):
            for key in self._TRACK_DATA_KEYS:
                if old.get(key) != new.get(key):
                    return True
        return False

    def restore_snapshot(self, state):
        """恢复快照（3.0.0：撤销/重做同时恢复音轨）"""
        if isinstance(state, list):          # 兼容旧版纯音符栈
            self.sequence = copy.deepcopy(state)
        else:
            self.sequence = copy.deepcopy(state.get('sequence', []))
            new_tracks = []
            for index, data in enumerate(state.get('tracks') or []):
                track = self._make_track(
                    data.get('name', f"轨道 {index + 1}"),
                    data.get('wavetype', 'triangle'),
                    data.get('color'))
                track.update({k: v for k, v in data.items() if not k.startswith('_')})
                new_tracks.append(track)
            new_tracks = new_tracks or [self._make_track("轨道 1")]
            # 撤销/重做绝大多数只动音符，音轨没变时重建面板毫无意义：
            # build_track_panel 建控件 + 布局要 600+ ms，是撤销卡顿的真正来源
            if self._tracks_differ(new_tracks):
                self.tracks = new_tracks
                self.current_track = max(0, min(self.current_track, len(self.tracks) - 1))
                self.refresh_track_panel()
            # 数据一致时保留原 track dict（里面有 _mute_var 等控件变量），不动面板
        # 3.1：效果器块跟着一起撤销/重做
        if isinstance(state, dict) and 'effect_blocks' in state:
            self.effect_blocks = copy.deepcopy(state.get('effect_blocks') or [])
            self.selected_block = None
            self._fx_next_id = 1 + max([b.get('id', 0) for b in self.effect_blocks], default=0)
        self.selected_note = None
        self.multi_selected_notes = []
        self.draw_effect_blocks()
        self.draw_notes()

    def save_state(self):
        """保存当前状态到撤销栈"""
        self.undo_stack.append(self._snapshot())
        # 快照是整份工程的深拷贝，音符多时单份就不小，栈不封顶会一路吃内存
        if len(self.undo_stack) > 60:
            del self.undo_stack[0]
        self.redo_stack = []  # 清空重做栈
        self.status_var.set(f"状态已保存 ({len(self.undo_stack)})")
    
    def transpose_all(self):
        """一键变调所有音符"""
        # 创建对话框获取变调量
        dialog = tk.Toplevel(self.root)
        dialog.title("一键变调")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="输入要升高的半音数（正数升高，负数降低）:").pack(pady=10)
        
        semitones_var = tk.StringVar(value="0")
        entry = ttk.Entry(dialog, textvariable=semitones_var, width=10)
        entry.pack(pady=5)
        entry.select_range(0, tk.END)  # 选中文本方便输入
        entry.focus_set()  # 设置焦点
        
        status_var = tk.StringVar(value="")
        ttk.Label(dialog, textvariable=status_var).pack(pady=5)
        
        def apply_transpose():
            try:
                semitones = int(semitones_var.get())
                if not self.sequence:
                    status_var.set("没有音符可调整")
                    return
                    
                # 保存当前状态到撤销栈
                self.save_state()
                
                # 执行变调操作
                for note in self.sequence:
                    new_pitch = note['pitch'] + semitones
                    # 确保音高在0-95范围内（修正范围）
                    note['pitch'] = max(0, min(95, new_pitch))
                
                # 重新绘制音符
                self.draw_notes()
                self.status_var.set(f"所有音符已升高 {semitones} 个半音")
                dialog.destroy()
                
            except ValueError:
                status_var.set("请输入有效的整数")
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="应用", command=apply_transpose).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
        
        # 绑定回车键
        dialog.bind("<Return>", lambda e: apply_transpose())

    def unify_amplitude_for_selected(self):
        """统一设定选中音符的振幅（按倍数调整）"""
        # 确定要设置的音符集合
        notes_to_set = []
        if self.multi_selected_notes:
            notes_to_set = self.multi_selected_notes
        elif self.selected_note:
            notes_to_set = [self.selected_note]
        else:
            messagebox.showinfo("提示", "请先选择要设置振幅的音符")
            return
            
        # 创建对话框
        dialog = tk.Toplevel(self.root)
        dialog.title("统一设定振幅（按倍数调整）")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 振幅倍数设置
        ttk.Label(dialog, text="振幅倍数 (例如: 0.5=减半, 2.0=加倍):").pack(pady=5)
        amplitude_multiplier_var = tk.StringVar(value="1.0")
        amplitude_multiplier_entry = ttk.Entry(dialog, textvariable=amplitude_multiplier_var, width=10)
        amplitude_multiplier_entry.pack(pady=5)
        
        # 振幅变化率倍数设置
        ttk.Label(dialog, text="振幅变化率倍数:").pack(pady=5)
        amp_change_multiplier_var = tk.StringVar(value="1.0")
        amp_change_multiplier_entry = ttk.Entry(dialog, textvariable=amp_change_multiplier_var, width=10)
        amp_change_multiplier_entry.pack(pady=5)
        
        status_var = tk.StringVar(value="")
        ttk.Label(dialog, textvariable=status_var).pack(pady=5)
        
        def apply_multiplier_amplitude():
            try:
                amplitude_multiplier = float(amplitude_multiplier_var.get())
                amp_change_multiplier = float(amp_change_multiplier_var.get())
                
                # 验证输入范围
                if amplitude_multiplier <= 0:
                    status_var.set("振幅倍数必须大于0")
                    return
                    
                # 保存当前状态到撤销栈
                self.save_state()
                
                # 按倍数调整振幅和变化率
                for note in notes_to_set:
                    # 调整起始振幅
                    new_start_amp = note['start_amp'] * amplitude_multiplier
                    note['start_amp'] = max(0.0, min(1.0, new_start_amp))  # 限制在0-1范围内
                    
                    # 调整振幅变化率
                    note['amp_change_rate'] = note.get('amp_change_rate', 0.0) * amp_change_multiplier
                
                # 重新绘制
                self.draw_notes()
                self.status_var.set(f"已将 {len(notes_to_set)} 个音符的振幅调整为原来的{amplitude_multiplier}倍，变化率调整为{amp_change_multiplier}倍")
                dialog.destroy()
                
            except ValueError:
                status_var.set("请输入有效的数字")
        
        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="应用", command=apply_multiplier_amplitude).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
        
        # 绑定回车键
        dialog.bind("<Return>", lambda e: apply_multiplier_amplitude())
        
        # 设置焦点并选中文本
        amplitude_multiplier_entry.select_range(0, tk.END)
        amplitude_multiplier_entry.focus_set()

    def set_timbre_amplitude(self):
        """宏操作：按音色统一设定起始振幅（倍数 / 定值两种方式）"""
        if not self.sequence:
            messagebox.showinfo("宏操作", "当前工程里没有音符")
            return
        TimbreAmplitudeDialog(self)

    def sync_move_notes(self, scope='all'):
        """宏操作：整体平移一批音符，使最早的那个从指定 16 分音符开始

        scope='all' 取全部音符，'sel' 取当前框选的音符。只改 start，
        音符之间的距离、时长、力度全都不动，所以节奏不会走样。
        """
        if scope == 'sel':
            targets = (list(self.multi_selected_notes)
                       or ([self.selected_note] if self.selected_note else []))
            if not targets:
                messagebox.showinfo("同步移动音符",
                                    "请先框选要移动的音符（按住 Alt 拖动）")
                return
        else:
            targets = list(self.sequence)
            if not targets:
                messagebox.showinfo("同步移动音符", "当前工程里没有音符")
                return
        SyncMoveDialog(self, scope, targets)

    def undo(self, event=None):
        """撤销操作"""
        if self.undo_stack:
            # 保存当前状态到重做栈
            self.redo_stack.append(self._snapshot())
            
            # 恢复上一个状态
            self.restore_snapshot(self.undo_stack.pop())
            self.status_var.set(f"撤销完成 ({len(self.undo_stack)}/{len(self.redo_stack)})")
    
    def redo(self, event=None):
        """重做操作"""
        if self.redo_stack:
            # 保存当前状态到撤销栈
            self.undo_stack.append(self._snapshot())
            
            # 恢复重做状态
            self.restore_snapshot(self.redo_stack.pop())
            self.status_var.set(f"重做完成 ({len(self.undo_stack)}/{len(self.redo_stack)})") 
    
    def process_note_line(self, line, notes):
        """优化后的音符解析逻辑"""
        line = line.strip()
        if not line:
            return
            
        # 快速检查是否为节拍信息行
        if line.startswith("TimeSignature;"):
            return
            
        # 使用更高效的分割方式
        try:
            parts = line.split(',', 6)  # 最多分割6次，避免过多分割操作
            if len(parts) < 7:
                return
                
            # 直接解析基本参数
            wavetype = parts[0].strip()
            freq = float(parts[1])
            start = float(parts[2])
            duration = float(parts[3])
            start_amp = float(parts[4])
            amp_change_rate = float(parts[5])
            
            # 快速解析JSON参数（3.0.0：JSON 后面可能还带着音轨号，先切出来）
            kwargs_str = parts[6].strip() if len(parts) > 6 else "{}"
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
            
            if kwargs_str and kwargs_str != "{}":
                # 优化JSON解析：先检查是否为简单格式
                if kwargs_str.startswith('{') and kwargs_str.endswith('}'):
                    try:
                        kwargs = json.loads(kwargs_str)
                    except json.JSONDecodeError:
                        # 快速修复常见JSON格式问题
                        kwargs_str = kwargs_str.replace("'", '"')
                        kwargs_str = kwargs_str.replace("True", "true").replace("False", "false")
                        try:
                            kwargs = json.loads(kwargs_str)
                        except:
                            kwargs = {}
            
            # 计算音高
            midi_pitch = int(round(12 * math.log2(freq / 440) + 69))
            pitch = max(0, min(95, midi_pitch - 12))
            
            # 直接创建音符对象
            note_obj = {
                'wavetype': wavetype,
                'start': start,
                'duration': duration,
                'pitch': pitch,
                'start_amp': start_amp,
                'amp_change_rate': amp_change_rate,
                'kwargs': kwargs,
                'track': track_index,   # 3.0.0
            }
            
            notes.append(note_obj)
            
        except (ValueError, IndexError) as e:
            # 快速错误处理，避免异常影响性能
            print(f"快速解析失败，行内容: {line[:50]}... 错误: {e}")

    def fix_json_format(self, json_str):
        """修复JSON格式问题"""
        # 移除可能的空白字符问题
        json_str = json_str.strip()
        
        # 确保使用双引号
        json_str = json_str.replace("'", '"')
        
        # 修复布尔值（true/false应该小写）
        json_str = json_str.replace("True", "true").replace("False", "false")
        
        return json_str

    def try_fix_json(self, json_str):
        """尝试修复JSON字符串"""
        try:
            # 如果JSON以{开头但可能格式不完整，尝试修复
            if json_str.startswith('{') and not json_str.endswith('}'):
                json_str = json_str + '}'
            elif not json_str.startswith('{'):
                json_str = '{' + json_str + '}'
                
            return json.loads(json_str)
        except:
            # 如果无法修复，返回空字典
            return {}

    def new_file(self):
        """创建新文件"""
        self.save_state()
        self.sequence = []
        self.selected_note = None
        self.multi_selected_notes = []
        # 3.0.0：重置为单轨
        self.tracks = [self._make_track("轨道 1", "triangle", self.TRACK_COLORS[0])]
        self.current_track = 0
        self.annotations = []
        # 3.1：效果器块也清空
        self.effect_blocks = []
        self.selected_block = None
        self._fx_next_id = 1
        self.refresh_track_panel()
        self.draw_effect_blocks()
        self.draw_notes()
        self.measure_range_manual = False
        self.sync_measure_range(force=True)
        self.current_file = None
        self.root.title("钢琴卷帘音序器 3.0 - 未命名")
        self.status_var.set("已创建新文件")
    
    # 在 open_file 方法中保持批注加载功能
    def open_file(self):
        """打开文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("音序文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                notes = []
                blocks = []          # 3.1：效果器块
                time_signature = [4, 4]  # 默认4/4拍
                self.annotations = []  # 清空现有批注
                # 3.0.0：先清空音轨，读到 Track 行再建；没有 Track 行就是单轨旧格式
                self.tracks = [self._make_track("轨道 1", "triangle", self.TRACK_COLORS[0])]
                self.current_track = 0
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # 检查是否是节拍信息
                        if line.startswith("TimeSignature;"):
                            parts = line.split(';')
                            if len(parts) >= 3:
                                try:
                                    numerator = int(parts[1])
                                    denominator = int(parts[2])
                                    time_signature = [numerator, denominator]
                                except ValueError:
                                    self.status_var.set("警告: 无效的节拍信息格式，使用默认4/4拍")
                        
                        # 检查是否是批注信息
                        elif line.startswith("Annotation;"):
                            parts = line.split(';', 2)
                            if len(parts) >= 3:
                                try:
                                    measure = int(parts[1])
                                    content = parts[2]
                                    self.annotations.append({
                                        'measure': measure,
                                        'content': content
                                    })
                                except ValueError:
                                    pass

                        # 3.1：效果器块  EffectBlock;起点;终点;最低音;最高音;JSON
                        elif line.startswith("EffectBlock;"):
                            parts = line.split(';', 5)
                            if len(parts) >= 6:
                                try:
                                    payload = json.loads(parts[5]) if parts[5].strip() else {}
                                    block = self._new_effect_block(
                                        float(parts[1]), float(parts[2]),
                                        int(float(parts[3] or 0)), int(float(parts[4] or 95)))
                                    if payload.get('color'):
                                        block['color'] = payload['color']
                                    block['timbres'] = payload.get('timbres')
                                    fx = payload.get('fx') or {}
                                    block['fx']['lowpass_1st']['on'] = bool(
                                        fx.get('lowpass_1st', {}).get('on'))
                                    block['fx']['sustain'].update({
                                        'on': bool(fx.get('sustain', {}).get('on')),
                                        'value': fx.get('sustain', {}).get('value', 0.2)})
                                    block['fx']['reverb'].update({
                                        'on': bool(fx.get('reverb', {}).get('on')),
                                        'ir': fx.get('reverb', {}).get('ir', ''),
                                        'mix': fx.get('reverb', {}).get('mix', 0.3)})
                                    blocks.append(block)
                                except (ValueError, json.JSONDecodeError):
                                    pass

                        # 3.1：音轨定义  Track;序号;名称;音色;音量;声像;静音;独奏
                        elif line.startswith("Track;"):
                            parts = line.split(';')
                            if len(parts) >= 4:
                                try:
                                    index = int(parts[1])
                                    name = parts[2] if len(parts) > 2 else f"轨道 {index + 1}"
                                    wavetype = parts[3] if len(parts) > 3 else "triangle"
                                    volume = float(parts[4]) if len(parts) > 4 and parts[4] else 0.8
                                    pan = float(parts[5]) if len(parts) > 5 and parts[5] else 0.0
                                    mute = bool(int(parts[6])) if len(parts) > 6 and parts[6] else False
                                    solo = bool(int(parts[7])) if len(parts) > 7 and parts[7] else False
                                    color = self.TRACK_COLORS[index % len(self.TRACK_COLORS)]
                                    track = self._make_track(name, wavetype, color)
                                    track.update({'volume': volume, 'pan': pan,
                                                  'mute': mute, 'solo': solo})
                                    while len(self.tracks) <= index:
                                        fill = len(self.tracks)
                                        self.tracks.append(self._make_track(
                                            f"轨道 {fill + 1}", "triangle",
                                            self.TRACK_COLORS[fill % len(self.TRACK_COLORS)]))
                                    self.tracks[index] = track
                                except ValueError:
                                    pass

                        else:
                            # 处理音符数据
                            self.process_note_line(line, notes)
                
                # 新增：按起始时间对音符事件进行排序
                notes.sort(key=lambda note: note['start'])

                # 3.0.0：万一音符引用了不存在的音轨，补齐轨道表并夹住索引
                max_track = max([n.get('track', 0) for n in notes], default=0)
                while len(self.tracks) <= max_track:
                    fill = len(self.tracks)
                    self.tracks.append(self._make_track(
                        f"轨道 {fill + 1}", "triangle",
                        self.TRACK_COLORS[fill % len(self.TRACK_COLORS)]))
                for note in notes:
                    if note.get('track', 0) >= len(self.tracks):
                        note['track'] = 0

                # 设置节拍
                self.set_time_signature(time_signature[0], time_signature[1])
                
                # 保存当前状态到撤销栈
                self.save_state()
                
                # 加载音符数据
                self.sequence = notes
                self.effect_blocks = blocks      # 3.1
                self.selected_block = None
                self.selected_note = None
                self.refresh_track_panel()   # 3.0.0
                self.draw_effect_blocks()
                self.draw_notes()
                # 3.0.0：播放范围自动对齐乐曲长度，避免长文件被默认的 64 小节截断
                self.measure_range_manual = False
                total_measures = self.sync_measure_range(force=True)
                self.current_file = file_path
                self.root.title(f"钢琴卷帘音序器 3.0 - {os.path.basename(file_path)}")
                
                # 更新状态栏
                if time_signature != [4, 4]:
                    self.status_var.set(f"已打开文件: {os.path.basename(file_path)} | "
                                        f"节拍: {time_signature[0]}/{time_signature[1]} | "
                                        f"共 {total_measures} 小节 | 音符 {len(notes)} 个")
                else:
                    self.status_var.set(f"已打开文件: {os.path.basename(file_path)} | "
                                        f"共 {total_measures} 小节 | 音符 {len(notes)} 个")
                    
            except Exception as e:
                messagebox.showerror("错误", f"打开文件时出错: {str(e)}")
                import traceback
                traceback.print_exc()
    
    def save_file(self):
        """保存文件"""
        if not self.current_file:
            self.save_as_file()
            return
        
        self.save_to_file(self.current_file)
    
    def save_as_file(self):
        """另存为文件"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("音序文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.save_to_file(file_path)
            self.current_file = file_path
            self.root.title(f"钢琴卷帘音序器 3.0 - {os.path.basename(file_path)}")

    # 在 save_to_file 方法中保持批注保存功能
    def save_to_file(self, file_path):
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"TimeSignature;{self.time_signature[0]};{self.time_signature[1]}\n")
                # 3.0.0：音轨定义（排在批注之前，旧版程序会忽略以 Track; 开头的行）
                for index, track in enumerate(self.tracks):
                    name = str(track['name']).replace(';', '；').replace('\n', ' ')
                    file.write("Track;{};{};{};{:.2f};{:.2f};{};{}\n".format(
                        index, name, track['wavetype'],
                        track.get('volume', 1.0), track.get('pan', 0.0),
                        1 if track.get('mute') else 0,
                        1 if track.get('solo') else 0))
                # 保存批注
                for annotation in self.annotations:
                    file.write(f"Annotation;{annotation['measure']};{annotation['content']}\n")

                # 3.1：效果器块（EffectBlock;起点;终点;最低音;最高音;JSON）
                # 旧版程序读这行时会因为逗号不足 7 段而跳过，格式向后兼容
                for blk in self.effect_blocks:
                    payload = {
                        'color': blk.get('color'),
                        'timbres': blk.get('timbres'),
                        'fx': blk.get('fx', {}),
                    }
                    file.write("EffectBlock;{:.4f};{:.4f};{};{};{}\n".format(
                        blk['start'], blk['end'], blk.get('pitch_low', 0),
                        blk.get('pitch_high', 95), json.dumps(payload)))

                for note in self.sequence:
                    # 计算频率（从音高）
                    midi_pitch = note['pitch'] + 12  # C0在MIDI系统中是12
                    freq = 440 * (2 ** ((midi_pitch - 69) / 12))
                    
                    # 关键修改：直接使用振幅变化率
                    start_amp = note['start_amp']
                    amp_change_rate = note.get('amp_change_rate', 0.0)
                    
                    # 处理附加参数
                    kwargs_str = json.dumps(note['kwargs'])
                    
                    # 关键修改：使用振幅变化率；3.0.0 末尾追加音轨号
                    note_str = f"{note['wavetype']},{freq:.2f},{note['start']:.2f},{note['duration']:.2f}," \
                            f"{start_amp:.2f},{amp_change_rate:.4f},{kwargs_str},{note.get('track', 0)}"
                    
                    file.write(note_str + "\n")
            
            self.status_var.set(f"文件已保存: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"保存文件时出错: {str(e)}")

    def show_about(self):
        """显示关于对话框"""
        messagebox.showinfo("关于", "钢琴卷帘音序器\n\n"
                               "用于创建符合电子音乐程序要求的音序文件\n"
                               "版本 3.0\n\n"
                               "操作说明:\n"
                               "- 双击添加新音符\n"
                               "- 拖拽音符调整位置（按“吸附”档位对齐）\n"
                               "- Shift+拖拽调整持续时间\n"
                               "- Ctrl+拖拽调整音高\n"
                               "- 双击音符打开编辑器\n"
                               "-----------------------\n"
                               "3.0 新增：\n"
                               "- 空格：播放 / 停止\n"
                               "- Delete：删除选中音符\n"
                               "- Ctrl+C / Ctrl+X / Ctrl+V：复制 / 剪切 / 粘贴到光标\n"
                               "- Ctrl+D：原地复制\n"
                               "- 循环：按小节区间反复播放\n"
                               "- 节拍器：播放时叠加节拍音\n"
                               "- 音轨：左侧面板可增删轨道、设音量声像、静音独奏\n"
                               "- 音符填充 = 音色专属色块 × 力度，边框 = 音轨颜色\n"
                               "-----------------------\n"
                               "常见参数：\n"
                               "幅值：\n"
                               "min_amp：最小幅值\n"
                               "max_amp：最大幅值\n"
                               "fi、fo：管弦乐音色的淡入淡出\n"
                               "-----\n"
                               "颤音：\n"
                               "vibrato_enabled：启用颤音\n"
                               "spread_period：颤音周期\n"
                               "vibrato_depth：颤音幅度\n"
                               "vibrato_continuous：颤音是否连续\n"
                               "-----\n"
                               "方波：\n"
                               "start_duty：起始占空比\n"
                               "duty_change_rate：占空比变化率\n"
                               "min_duty：最小占空比\n"
                               "max_duty：最大占空比\n"
                               "-----\n"
                               "节拍音色：\n"
                               "rand：扩散幅度\n"
                               "-----\n"
                               "滤波：\n"
                               "R：电阻大小\n"
                               "C：电容大小\n"
                               "提示：截止频率fc=1/(2pi*R*c)"
                               )
        

class NoteEditor(tk.Toplevel):
    def __init__(self, parent, note_data, app):
        super().__init__(parent)
        self.note_data = note_data
        self.app = app
        self.saved = False
        
        # 加载音频样本
        self.app.load_audio_samples()
        
        self.title("音符参数编辑器")
        self.geometry("620x670")  # 3.0.0：新增延音、一阶低通两行，加高
        self.transient(parent)
        self.grab_set()
        
        self.create_interface()
    
    def create_interface(self):
        # 主框架
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 音色 - 添加音频采样选项
        ttk.Label(main_frame, text="音色:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.wave_type = tk.StringVar(value=self.note_data["wavetype"])
        wave_types = ["custom_harmonics", "harmonics", "harmonica","violin","string", "ocarina", "organ", "steel-tongue-drum", "sine", 
              "square", "triangle", "sawtooth", "noise", "filtered_sq", 
              "filtered_noise", "double_filtered_noise", "double_filtered_sq", 
              "beat", "empty", "audio_sample"]  # 添加audio_sample选项
        wave_combo = ttk.Combobox(main_frame, textvariable=self.wave_type, values=wave_types, state="readonly")
        wave_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        wave_combo.bind('<<ComboboxSelected>>', self.on_wave_type_changed)
        
        # 3.0.0：当前音色的色块（与卷帘里该音色音符的填充色一致）
        self.wave_color_patch = tk.Label(
            main_frame, text="   ", bg=self.app.wavetype_color(self.wave_type.get()),
            bd=1, relief=tk.SOLID)
        self.wave_color_patch.grid(row=0, column=3, padx=(2, 5))
        
        # 音频采样按钮 - 只在选择audio_sample时启用
        self.audio_sample_btn = ttk.Button(main_frame, text="音频采样", command=self.open_audio_sample_list)
        self.audio_sample_btn.grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        
        # 音频样本显示标签
        self.sample_label_var = tk.StringVar(value="未选择样本")
        ttk.Label(main_frame, textvariable=self.sample_label_var, foreground="blue").grid(
            row=1, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        
        # 起始时间
        ttk.Label(main_frame, text="起始时间 (十六分音符):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_time = tk.DoubleVar(value=self.note_data["start"])
        start_frame = ttk.Frame(main_frame)
        start_frame.grid(row=1, column=1, sticky=tk.W)
        
        ttk.Entry(start_frame, textvariable=self.start_time, width=8).pack(side=tk.LEFT)
        
        # 起始时间调整按钮（步长：1/32音符 = 0.5单位）
        btn_frame = ttk.Frame(start_frame)
        btn_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="◀", width=2, 
                  command=lambda: self.adjust_value(self.start_time, -0.5)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="▶", width=2, 
                  command=lambda: self.adjust_value(self.start_time, 0.5)).pack(side=tk.LEFT)
        
        # 持续时间
        ttk.Label(main_frame, text="持续时间 (十六分音符):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.duration = tk.DoubleVar(value=self.note_data["duration"])
        duration_frame = ttk.Frame(main_frame)
        duration_frame.grid(row=2, column=1, sticky=tk.W)
        
        ttk.Entry(duration_frame, textvariable=self.duration, width=8).pack(side=tk.LEFT)
        
        # 持续时间调整按钮（步长：1/32音符 = 0.5单位）
        btn_frame = ttk.Frame(duration_frame)
        btn_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="◀", width=2, 
                  command=lambda: self.adjust_value(self.duration, -0.5, min_val=0.5)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="▶", width=2, 
                  command=lambda: self.adjust_value(self.duration, 0.5)).pack(side=tk.LEFT)
        
        # 音高
        ttk.Label(main_frame, text="音高:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.pitch = tk.IntVar(value=self.note_data["pitch"])
        pitch_frame = ttk.Frame(main_frame)
        pitch_frame.grid(row=3, column=1, sticky=tk.W)
        
        # 获取音符名称
        note = self.app.get_note_from_pitch(self.pitch.get())
        self.note_var = tk.StringVar(value=note)
        ttk.Label(pitch_frame, textvariable=self.note_var, width=5).pack(side=tk.LEFT)
        
        # 音高调整按钮（步长：1个半音）
        btn_frame = ttk.Frame(pitch_frame)
        btn_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="▼", width=2, 
                  command=lambda: self.adjust_pitch(-1)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="▲", width=2, 
                  command=lambda: self.adjust_pitch(1)).pack(side=tk.LEFT)
        
        # 起始振幅
        ttk.Label(main_frame, text="起始振幅 (0.0-1.0):").grid(row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.start_amp = tk.DoubleVar(value=self.note_data["start_amp"])
        start_amp_frame = ttk.Frame(main_frame)
        start_amp_frame.grid(row=4, column=1, sticky=tk.W)
        
        ttk.Entry(start_amp_frame, textvariable=self.start_amp, width=8).pack(side=tk.LEFT)
        
        # 起始振幅调整按钮（步长：0.1）
        btn_frame = ttk.Frame(start_amp_frame)
        btn_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="-", width=2, 
                  command=lambda: self.adjust_value(self.start_amp, -0.1, min_val=0.0, max_val=1.0)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="+", width=2, 
                  command=lambda: self.adjust_value(self.start_amp, 0.1, min_val=0.0, max_val=1.0)).pack(side=tk.LEFT)
        
        # 将"结束振幅"改为"振幅变化率"
        ttk.Label(main_frame, text="振幅变化率:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=5)
        self.amp_change_rate = tk.DoubleVar(value=self.note_data.get("amp_change_rate", 0.0))
        amp_change_frame = ttk.Frame(main_frame)
        amp_change_frame.grid(row=5, column=1, sticky=tk.W)
        
        ttk.Entry(amp_change_frame, textvariable=self.amp_change_rate, width=8).pack(side=tk.LEFT)
        
        # 振幅变化率调整按钮
        btn_frame = ttk.Frame(amp_change_frame)
        btn_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="-", width=2, 
                command=lambda: self.adjust_value(self.amp_change_rate, -0.1)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="+", width=2, 
                command=lambda: self.adjust_value(self.amp_change_rate, 0.1)).pack(side=tk.LEFT)
        
        # 3.0.0：延音（主时长结束后，衰减到静音的时间 = 主时长 × 该系数）
        ttk.Label(main_frame, text="延音 (0.0-1.0，×主时长):").grid(row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.sustain_ratio = tk.DoubleVar(
            value=self.note_data.get("sustain_ratio",
                                     self.note_data.get("kwargs", {}).get("sustain_ratio", 0.0)))
        sustain_frame = ttk.Frame(main_frame)
        sustain_frame.grid(row=6, column=1, sticky=tk.W)
        
        ttk.Entry(sustain_frame, textvariable=self.sustain_ratio, width=8).pack(side=tk.LEFT)
        
        # 延音调整按钮（步长：0.05）
        btn_frame = ttk.Frame(sustain_frame)
        btn_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(btn_frame, text="-", width=2, 
                command=lambda: self.adjust_value(self.sustain_ratio, -0.05, min_val=0.0, max_val=1.0)).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="+", width=2, 
                command=lambda: self.adjust_value(self.sustain_ratio, 0.05, min_val=0.0, max_val=1.0)).pack(side=tk.LEFT)
        
        
        # 3.0.0：一阶低通（仅非采样音色）。截止频率跟着基波走，fc = 1.5 × f0
        self.lowpass_1st = tk.BooleanVar(
            value=bool(self.note_data.get("kwargs", {}).get("lowpass_1st", False)))
        self.lowpass_chk = ttk.Checkbutton(
            main_frame, text="一阶低通（截止 = 1.5 × 基波）",
            variable=self.lowpass_1st)
        self.lowpass_chk.grid(row=7, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(2, 0))
        # 说明单独占一行：挤在复选框右边会被 620px 的窗口裁掉
        ttk.Label(main_frame, foreground="#666666", font=("", 7),
                  text="磨圆高频、接近积分器效果；直流增益为 1，不会发散").grid(
            row=8, column=0, columnspan=4, sticky=tk.W, padx=25, pady=(0, 2))

        # 自定义谐波系数文件选择（仅在custom_harmonics时启用）
        self.harm_coeff_label_var = tk.StringVar(value="未选择波形文件")
        self.harm_coeff_label = ttk.Label(main_frame, textvariable=self.harm_coeff_label_var, foreground="magenta")
        self.harm_coeff_label.grid(row=9, column=0, sticky=tk.W, padx=5, pady=2)
        self.harm_coeff_btn = ttk.Button(main_frame, text="选择波形/谐波文件", command=self.select_harm_coeff)
        self.harm_coeff_btn.grid(row=9, column=1, sticky=tk.W, padx=5, pady=2)
        
        # 附加参数
        ttk.Label(main_frame, text="附加参数 (JSON格式):").grid(row=10, column=0, sticky=tk.W, padx=5, pady=5)
        self.kwargs_text = tk.Text(main_frame, height=3, width=40)
        self.kwargs_text.grid(row=10, column=1, sticky=tk.W, padx=5, pady=5)
        # 3.0.0：延音与一阶低通都有独立控件，不再重复出现在 JSON 里
        kwargs_display = {k: v for k, v in (self.note_data.get("kwargs") or {}).items()
                          if k not in ('sustain_ratio', 'lowpass_1st')}
        self.kwargs_text.insert(tk.END, json.dumps(kwargs_display, indent=2))
        
        # 底部按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=11, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="确定", command=self.save_and_close).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=5)

        # 初始化控件状态
        self.on_wave_type_changed()
        
        # 如果已是custom_harmonics，加载已保存的路径
        if self.note_data["wavetype"] == "custom_harmonics":
            saved_path = self.note_data.get("kwargs", {}).get("coeff_path", "")
            if saved_path:
                self.harm_coeff_label_var.set(f"波形文件: {os.path.basename(saved_path)}")
        # 如果已经是音频样本，显示当前样本信息
        if self.note_data["wavetype"] == "audio_sample":
            sample_id = self.note_data.get("kwargs", {}).get("sample_id")
            if sample_id:
                sample = self.app.get_audio_sample_by_id(sample_id)
                if sample:
                    self.sample_label_var.set(f"当前样本: {sample['name']} (采样{sample_id})")

    def on_wave_type_changed(self, event=None):
        """音色改变时的处理"""
        wt = self.wave_type.get()
        # 3.0.0：同步色块
        try:
            self.wave_color_patch.config(bg=self.app.wavetype_color(wt))
        except Exception:
            pass
        # 自定义谐波
        if wt == "custom_harmonics":
            self.harm_coeff_btn.config(state=tk.NORMAL)
            self.harm_coeff_label.config(state=tk.NORMAL)
        else:
            self.harm_coeff_btn.config(state=tk.DISABLED)
            self.harm_coeff_label.config(state=tk.DISABLED)
        # 音频样本
        if wt == "audio_sample":
            self.audio_sample_btn.config(state=tk.NORMAL)
            # 一阶低通只作用于合成波形，采样本来就是录好的声音，没得滤
            self.lowpass_chk.config(state=tk.DISABLED)
            for widget in self.grid_slaves(row=4):
                if isinstance(widget, (ttk.Frame, ttk.Label)):
                    continue
                widget.config(state=tk.DISABLED)
        else:
            self.audio_sample_btn.config(state=tk.DISABLED)
            self.lowpass_chk.config(state=tk.NORMAL)
            for widget in self.grid_slaves(row=4):
                if isinstance(widget, (ttk.Frame, ttk.Label)):
                    continue
                widget.config(state=tk.NORMAL)
    
    def select_harm_coeff(self):
        """选择自定义波形文件（-wave.npz / -har.npz，自动检测格式）"""
        path = filedialog.askopenfilename(
            title="选择波形/谐波文件（-wave.npz 或 -har.npz）",
            filetypes=[("NPZ 波形/系数文件", "*.npz"), ("所有文件", "*.*")]
        )
        if not path:
            return
        self.harm_coeff_label_var.set(f"波形文件: {os.path.basename(path)}")
        # 写入kwargs
        try:
            kwargs = json.loads(self.kwargs_text.get("1.0", tk.END).strip())
        except:
            kwargs = {}
        kwargs['coeff_path'] = path
        self.kwargs_text.delete("1.0", tk.END)
        self.kwargs_text.insert(tk.END, json.dumps(kwargs, indent=2))
    
    def open_audio_sample_list(self):
        """打开音频样本列表窗口"""
        sample_window = AudioSampleListWindow(self, self.app)
        self.wait_window(sample_window)
        
        if sample_window.selected_sample:
            sample = sample_window.selected_sample
            self.sample_label_var.set(f"已选择: {sample['name']} (采样{sample['id']})")
            
            # 更新kwargs中的样本ID
            try:
                kwargs = json.loads(self.kwargs_text.get("1.0", tk.END).strip())
            except:
                kwargs = {}
            kwargs['sample_id'] = sample['id']
            kwargs['sample_file'] = sample['file_path']
            
            self.kwargs_text.delete("1.0", tk.END)
            self.kwargs_text.insert(tk.END, json.dumps(kwargs, indent=2))

    def adjust_value(self, var, increment, min_val=None, max_val=None):
        """调整值"""
        current = var.get()
        new_value = current + increment
        
        # 应用范围限制
        if min_val is not None:
            new_value = max(min_val, new_value)
        if max_val is not None:
            new_value = min(max_val, new_value)
        
        var.set(new_value)
    
    def adjust_pitch(self, increment):
        """调整音高"""
        current = self.pitch.get()
        new_value = current + increment
        
        # 限制音高范围 (0-95)
        new_value = max(0, min(95, new_value))
        
        self.pitch.set(new_value)
        self.note_var.set(self.app.get_note_from_pitch(new_value))
    
    def save_and_close(self):
        """保存并关闭窗口"""
        try:
            kwargs = json.loads(self.kwargs_text.get("1.0", tk.END).strip() or "{}")
            # 3.0.0：延音写进 kwargs，文件格式不变，旧版程序/旧文件都兼容
            try:
                sustain_ratio = max(0.0, min(1.0, float(self.sustain_ratio.get())))
            except (tk.TclError, ValueError):
                sustain_ratio = 0.0
            if sustain_ratio > 0:
                kwargs['sustain_ratio'] = round(sustain_ratio, 4)
            else:
                kwargs.pop('sustain_ratio', None)
            # 3.0.0：一阶低通开关（截止 = 1.5 × 基波，由渲染端按 freq 算）
            if self.lowpass_1st.get() and self.wave_type.get() != 'audio_sample':
                kwargs['lowpass_1st'] = True
            else:
                kwargs.pop('lowpass_1st', None)
            self.note_data.update({
                "wavetype": self.wave_type.get(),
                "start": self.start_time.get(),
                "duration": self.duration.get(),
                "pitch": self.pitch.get(),
                "start_amp": self.start_amp.get(),
                "amp_change_rate": self.amp_change_rate.get(),  # 关键修改
                "sustain_ratio": sustain_ratio,                 # 3.0.0
                "kwargs": kwargs
            })
            self.saved = True
            self.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"保存音符参数时出错: {str(e)}")

class AudioSampleListWindow(tk.Toplevel):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.selected_sample = None
        
        self.title("音频样本列表")
        self.geometry("600x400")
        self.transient(parent)
        self.grab_set()
        
        self.create_interface()
        self.load_samples()
    
    def create_interface(self):
        # 主框架
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 顶部按钮框架
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="导入音频文件", command=self.import_audio_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="刷新列表", command=self.load_samples).pack(side=tk.LEFT, padx=5)
        
        # 样本列表
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建树形视图显示样本
        columns = ("ID", "名称", "文件")
        self.tree = ttk.Treeview(list_frame, columns=columns, show="headings", height=15)
        
        # 设置列标题
        self.tree.heading("ID", text="采样ID")
        self.tree.heading("名称", text="样本名称")
        self.tree.heading("文件", text="文件名")
        
        # 设置列宽
        self.tree.column("ID", width=80)
        self.tree.column("名称", width=150)
        self.tree.column("文件", width=250)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定双击事件
        self.tree.bind("<Double-1>", self.on_sample_selected)
        
        # 底部按钮
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(bottom_frame, text="选择", command=self.select_sample).pack(side=tk.RIGHT, padx=5)
        ttk.Button(bottom_frame, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=5)
    
    def load_samples(self):
        """加载样本列表"""
        # 清空现有列表
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # 加载样本
        self.app.load_audio_samples()
        
        # 添加样本到列表
        for sample in self.app.audio_samples:
            self.tree.insert("", tk.END, values=(
                f"采样{sample['id']}",
                sample['name'],
                sample['filename']
            ), tags=(sample['id'],))
    
    def import_audio_file(self):
        """导入音频文件"""
        file_types = [
            ("音频文件", "*.wav *.mp3 *.ogg"),
            ("WAV文件", "*.wav"),
            ("MP3文件", "*.mp3"),
            ("OGG文件", "*.ogg"),
            ("所有文件", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=file_types
        )
        
        if file_path:
            sample_info = self.app.import_audio_sample(file_path)
            if sample_info:
                self.load_samples()
                messagebox.showinfo("成功", f"已导入音频样本: {sample_info['name']}")
    
    def on_sample_selected(self, event):
        """双击选择样本"""
        self.select_sample()
    
    def select_sample(self):
        """选择当前选中的样本"""
        selection = self.tree.selection()
        if selection:
            item = selection[0]
            values = self.tree.item(item, 'values')
            sample_id = int(values[0].replace("采样", ""))
            
            self.selected_sample = self.app.get_audio_sample_by_id(sample_id)
            self.destroy()


class EffectBlockDialog(tk.Toplevel):
    """效果器块设置弹窗（3.1）

    上半部分是这个块能用哪些效果（一阶低通 / 延音 / 卷积混响），
    下半部分列出“出现在这个色块时间范围内的、除音频采样外的所有音色”，
    默认全勾 = 对所有音色生效；取消勾选就把该音色排除在外。

    勾选结果同时决定色块大小：上界 = 勾选音色在全曲的最高音，
    下界 = 最低音（勾/取消勾时实时预览，取消关窗则还原）。
    """

    def __init__(self, app, block):
        super().__init__(app.root)
        self.app = app
        self.block = block
        # 弹窗里勾/取消勾音色会实时改色块高度，没点“确定”就关窗要还原
        self._orig_pitch = (int(block.get('pitch_low', 0)),
                            int(block.get('pitch_high', 95)))
        self._applied = False
        self.title(f"效果器块 FX{block['id']}")
        self.transient(app.root)
        self.resizable(False, False)
        self._build()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ---------------------------------------------------------------- 界面
    def _build(self):
        blk = self.block
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 作用范围
        range_frame = ttk.LabelFrame(main, text="作用范围（十六分音符）", padding=8)
        range_frame.pack(fill=tk.X, pady=(0, 8))
        self.start_var = tk.DoubleVar(value=round(blk['start'], 4))
        self.end_var = tk.DoubleVar(value=round(blk['end'], 4))
        ttk.Label(range_frame, text="起点").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(range_frame, textvariable=self.start_var, width=10).grid(
            row=0, column=1, padx=4)
        ttk.Label(range_frame, text="终点").grid(row=0, column=2, sticky=tk.W, padx=(8, 0))
        ttk.Entry(range_frame, textvariable=self.end_var, width=10).grid(
            row=0, column=3, padx=4)
        self.pitch_label = ttk.Label(range_frame, text=self._pitch_text(),
                                     foreground="#777")
        self.pitch_label.grid(row=1, column=0, columnspan=4,
                              sticky=tk.W, pady=(4, 0))

        # 效果列表
        fx_frame = ttk.LabelFrame(main, text="效果（勾选即对块内音符生效）", padding=8)
        fx_frame.pack(fill=tk.X, pady=(0, 8))
        fx = blk['fx']

        self.lp_var = tk.BooleanVar(value=bool(fx['lowpass_1st'].get('on')))
        ttk.Checkbutton(fx_frame,
                        text="一阶低通（截止 = 基波 × 1.5，类积分器，不发散）",
                        variable=self.lp_var).grid(row=0, column=0, columnspan=3,
                                                   sticky=tk.W, pady=(0, 6))

        self.sus_var = tk.BooleanVar(value=bool(fx['sustain'].get('on')))
        self.sus_value = tk.DoubleVar(value=float(fx['sustain'].get('value', 0.2)))
        ttk.Checkbutton(fx_frame, text="延音", variable=self.sus_var).grid(
            row=1, column=0, sticky=tk.W)
        ttk.Label(fx_frame, text="系数 × 主时长").grid(row=1, column=1, sticky=tk.E)
        ttk.Entry(fx_frame, textvariable=self.sus_value, width=8).grid(
            row=1, column=2, sticky=tk.W, padx=4)

        self.rv_var = tk.BooleanVar(value=bool(fx['reverb'].get('on')))
        self.rv_mix = tk.DoubleVar(value=float(fx['reverb'].get('mix', 0.3)))
        self.rv_ir = fx['reverb'].get('ir', '') or ''
        ttk.Checkbutton(fx_frame, text="卷积混响", variable=self.rv_var).grid(
            row=2, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Label(fx_frame, text="干湿比").grid(row=2, column=1, sticky=tk.E, pady=(6, 0))
        ttk.Entry(fx_frame, textvariable=self.rv_mix, width=8).grid(
            row=2, column=2, sticky=tk.W, padx=4, pady=(6, 0))
        self.ir_label = ttk.Label(fx_frame, text=self._ir_text(), foreground="#555")
        self.ir_label.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))
        ttk.Button(fx_frame, text="选择 IR 音频…", command=self.choose_ir).grid(
            row=3, column=2, sticky=tk.W, pady=(2, 0))

        # 预置 IR：impulse_responses/ 里的三条，点一下就换
        ttk.Label(fx_frame, text="预置：", foreground="#555").grid(
            row=4, column=0, sticky=tk.W, pady=(4, 0))
        preset_row = ttk.Frame(fx_frame)
        preset_row.grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=(4, 0))
        presets = self._preset_irs()
        if presets:
            for name, path in presets:
                ttk.Button(preset_row, text=name, width=7,
                           command=lambda p=path, n=name: self.use_preset(p, n)
                           ).pack(side=tk.LEFT, padx=(0, 4))
        else:
            ttk.Label(preset_row, text="（没找到 impulse_responses 目录）",
                      foreground="#999").pack(side=tk.LEFT)

        # 音色列表
        tim_frame = ttk.LabelFrame(
            main, text="应用到哪些音色（默认全选；音频采样不参与）", padding=8)
        tim_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        timbres = self.enclosed_timbres()
        allowed = blk.get('timbres')
        self.tim_vars = {}
        if not timbres:
            ttk.Label(tim_frame, text="块内还没有非采样音符",
                      foreground="#777").grid(row=0, column=0, sticky=tk.W)
        for i, wavetype in enumerate(timbres):
            var = tk.BooleanVar(value=(allowed is None or wavetype in allowed))
            self.tim_vars[wavetype] = var
            tk.Frame(tim_frame, width=12, height=12,
                     bg=self.app.wavetype_color(wavetype),
                     highlightthickness=1,
                     highlightbackground="#999").grid(row=i, column=0,
                                                      padx=(0, 6), pady=1, sticky=tk.W)
            ttk.Checkbutton(tim_frame, text=wavetype, variable=var,
                            command=self.preview_pitch).grid(
                row=i, column=1, sticky=tk.W)

        # 按钮
        btn_row = ttk.Frame(main)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="全选", command=lambda: self.set_all(True)).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="全不选", command=lambda: self.set_all(False)).pack(
            side=tk.LEFT, padx=4)
        ttk.Button(btn_row, text="删除此块", command=self.on_delete).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="取消", command=self.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text="确定", command=self.on_apply).pack(side=tk.RIGHT)

    def enclosed_timbres(self):
        """出现在这个色块时间范围内的音色（去掉 audio_sample），按首次出现排序

        只按时间判断：色块的上下界本身就是按音色算出来的，
        再用音高去筛音色就成了循环依赖，还会让被勾掉的音色在重开弹窗时消失。
        """
        seen = []
        for note in self.app.sequence:
            wavetype = note.get('wavetype')
            if wavetype == 'audio_sample':
                continue
            if (self.block['start'] - 1e-6 <= note['start']
                    < self.block['end'] - 1e-6 and wavetype not in seen):
                seen.append(wavetype)
        return seen

    def current_timbres(self):
        """弹窗里当前的勾选结果：全勾 = None（不限制音色），否则是列表"""
        if not self.tim_vars:
            return self.block.get('timbres')
        chosen = [wt for wt, var in self.tim_vars.items() if var.get()]
        if len(chosen) == len(self.tim_vars):
            return None
        return chosen

    def _pitch_text(self):
        low = int(self.block.get('pitch_low', 0))
        high = int(self.block.get('pitch_high', 95))
        return (f"音高 {self.app.get_note_from_pitch(low)}–"
                f"{self.app.get_note_from_pitch(high)}（{low}–{high}）："
                "上下界 = 勾选音色在全曲的最高/最低音")

    def preview_pitch(self):
        """勾选变化时立刻把色块高度贴到所选音色的音域（所见即所得）"""
        if not self.app.fit_block_pitch(self.block, self.current_timbres()):
            return          # 全不选 / 该音色没音符：保持原高度
        self.app._update_block_geometry(self.block)
        self.pitch_label.config(text=self._pitch_text())

    def _preset_irs(self):
        """impulse_responses/ 下可用的预置 IR：[(显示名, 路径), ...]"""
        base = getattr(self.app, 'ir_preset_dir', '') or ''
        found = []
        for name in ("环绕声", "浴室", "大堂"):
            path = os.path.join(base, name + ".wav")
            if os.path.isfile(path):
                found.append((name, path))
        return found

    def use_preset(self, path, name):
        """点预置按钮：换成这条 IR，并顺手勾上混响"""
        self.rv_ir = path
        self.rv_var.set(True)
        self.ir_label.config(text=self._ir_text())
        self.app.status_var.set(f"混响 IR 已切换为预置「{name}」")

    def _ir_text(self):
        if self.rv_ir:
            return "IR：" + os.path.basename(self.rv_ir)
        return "IR：（未选择，混响不会生效）"

    def set_all(self, value):
        for var in self.tim_vars.values():
            var.set(value)
        self.preview_pitch()

    # ---------------------------------------------------------------- 动作
    def choose_ir(self):
        path = filedialog.askopenfilename(
            parent=self, title="选择混响脉冲响应 / 音频",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac *.ogg"), ("所有文件", "*.*")])
        if path:
            self.rv_ir = path
            self.rv_var.set(True)
            self.ir_label.config(text=self._ir_text())

    def on_apply(self):
        blk = self.block
        try:
            start = max(0.0, float(self.start_var.get()))
            end = float(self.end_var.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("错误", "起点/终点必须是数字", parent=self)
            return
        if end <= start:
            messagebox.showerror("错误", "终点必须大于起点", parent=self)
            return

        self.app.save_state()
        blk['start'], blk['end'] = start, end
        blk['fx']['lowpass_1st']['on'] = bool(self.lp_var.get())
        blk['fx']['sustain']['on'] = bool(self.sus_var.get())
        try:
            blk['fx']['sustain']['value'] = max(0.0, min(1.0, float(self.sus_value.get())))
        except (tk.TclError, ValueError):
            pass
        blk['fx']['reverb']['on'] = bool(self.rv_var.get())
        blk['fx']['reverb']['ir'] = self.rv_ir
        try:
            blk['fx']['reverb']['mix'] = max(0.0, min(1.0, float(self.rv_mix.get())))
        except (tk.TclError, ValueError):
            pass

        checked = [wt for wt, var in self.tim_vars.items() if var.get()]
        # 全部勾上就等于“不限制音色”，存 None（以后新增的音色也能吃到效果）
        blk['timbres'] = None if len(checked) == len(self.tim_vars) else checked
        # 色块大小 = 勾选音色的音域（只改显示尺寸，不改作用逻辑）
        self.app.fit_block_pitch(blk, blk['timbres'])

        if self.rv_var.get() and not self.rv_ir:
            messagebox.showwarning("提示", "卷积混响还没选 IR 音频，本次渲染不会生效",
                                   parent=self)

        self.app.draw_effect_blocks()
        self.app.status_var.set(
            f"效果器块 FX{blk['id']} 已更新："
            + ("/".join(l for k, l in self.app.FX_LABELS
                        if blk['fx'].get(k, {}).get('on')) or "无效果"))
        self._applied = True        # 已确定，关窗时不要还原高度
        self.destroy()

    def on_delete(self):
        if not messagebox.askyesno("删除", f"删除效果器块 FX{self.block['id']}？", parent=self):
            return
        self.app.save_state()
        self.app.effect_blocks = [b for b in self.app.effect_blocks if b is not self.block]
        self.app.selected_block = None
        self._applied = True        # 块都删了，别再去还原它
        self.app.draw_effect_blocks()
        self.destroy()

    def destroy(self):
        """没点“确定”就关窗：把预览时改过的色块高度还原"""
        if not self._applied:
            self._applied = True
            low, high = self._orig_pitch
            self.block['pitch_low'], self.block['pitch_high'] = low, high
            try:
                self.app._update_block_geometry(self.block)
            except tk.TclError:
                pass
        super().destroy()


class SettingsDialog(tk.Toplevel):
    """设置窗口（目前只有音频合成进程数）"""

    def __init__(self, app, parent=None):
        super().__init__(parent or app.root)
        self.app = app
        self.title("设置")
        self.resizable(False, False)
        self.transient(parent or app.root)

        body = ttk.Frame(self, padding=(16, 14, 16, 10))
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text="音频合成", font=("", 10, "bold")).pack(anchor=tk.W)

        row = ttk.Frame(body)
        row.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(row, text="音频合成进程数：").pack(side=tk.LEFT)
        self.workers_var = tk.StringVar(value=str(app.synth_workers))
        self.spin = ttk.Spinbox(row, from_=1, to=app.max_synth_workers, width=4,
                                textvariable=self.workers_var, justify=tk.CENTER)
        self.spin.pack(side=tk.LEFT, padx=(2, 6))
        self.spin.bind("<KeyRelease>", lambda e: self._refresh_hint())
        self.spin.bind("<<Increment>>", lambda e: self._refresh_hint())
        self.spin.bind("<<Decrement>>", lambda e: self._refresh_hint())
        ttk.Label(row, text=f"本机 {os.cpu_count()} 核，最多 {app.max_synth_workers}").pack(side=tk.LEFT)

        self.hint_var = tk.StringVar()
        ttk.Label(body, textvariable=self.hint_var, wraplength=440, justify=tk.LEFT,
                  foreground="#555555").pack(anchor=tk.W, pady=(10, 0))

        btns = ttk.Frame(body)
        btns.pack(fill=tk.X, pady=(14, 0))
        ttk.Button(btns, text="恢复默认", command=self._restore_default).pack(side=tk.LEFT)
        ttk.Button(btns, text="取消", command=self.destroy).pack(side=tk.RIGHT)
        ttk.Button(btns, text="确定", command=self._ok).pack(side=tk.RIGHT, padx=(0, 6))

        self._refresh_hint()
        self._center(parent or app.root)
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _center(self, master):
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        try:
            x = master.winfo_rootx() + (master.winfo_width() - w) // 2
            y = master.winfo_rooty() + (master.winfo_height() - h) // 3
        except tk.TclError:
            x = y = 200
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _value(self):
        """当前输入值；不合法时返回 None"""
        try:
            v = int(str(self.workers_var.get()).strip())
        except (TypeError, ValueError):
            return None
        return max(1, min(self.app.max_synth_workers, v))

    def _refresh_hint(self):
        v = self._value()
        if v is None:
            self.hint_var.set(f"请输入 1 到 {self.app.max_synth_workers} 之间的整数。")
            return
        if v == 1:
            head = "当前：串行（只用主进程）"
            body = "不开子进程。最省内存，长曲子也最慢。"
        else:
            head = f"当前：{v} 进程并行"
            body = ("进程越多合成越快。不过混音和进程间传输由主程序串行完成，"
                    "超过 6~8 个以后提速会明显变小；每个子进程约占 100 MB 内存。\n"
                    "第一次用这个数量播放或导出时，子进程要先启动（约 1~3 秒），"
                    "之后整个运行期间都复用，不再重新启动。")
        self.hint_var.set(head + "\n" + body)

    def _restore_default(self):
        self.workers_var.set(str(self.app.default_settings()['synth_workers']))
        self._refresh_hint()

    def _ok(self):
        v = self._value()
        if v is None:
            messagebox.showwarning(
                "设置", f"请输入 1 到 {self.app.max_synth_workers} 之间的整数。", parent=self)
            return
        self.app.apply_synth_workers(v)
        self.destroy()


class TimbreAmplitudeDialog(tk.Toplevel):
    """宏操作：按音色统一设定起始振幅

    两种方式：
      · 统一调节倍数 —— 同一音色的所有音符乘同一个系数，原有的强弱关系保留；
      · 统一设定幅值 —— 全部压到同一个值，原有的强弱关系抹平。
    作用范围可以是「全部音符 / 仅当前音轨 / 仅框选音符」。
    """

    def __init__(self, app, parent=None):
        super().__init__(parent or app.root)
        self.app = app
        self.title("设定同一音色幅值")
        self.resizable(False, False)
        self.transient(parent or app.root)

        self._stats = self._collect_stats()
        self._build()
        self._center(parent or app.root)
        try:
            self.grab_set()
        except tk.TclError:
            pass
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ------------------------------------------------------------ 数据
    def _collect_stats(self):
        """按音色汇总：总数、当前轨个数、选中个数、振幅范围"""
        app = self.app
        sel_ids = {id(n) for n in app.multi_selected_notes}
        acc = {}
        for n in app.sequence:
            wt = n.get('wavetype', 'sine')
            d = acc.get(wt)
            if d is None:
                d = acc[wt] = {'all': 0, 'track': 0, 'sel': 0, 'lo': None, 'hi': None}
            amp = float(n.get('start_amp', 0.7))
            d['all'] += 1
            if n.get('track', 0) == app.current_track:
                d['track'] += 1
            if id(n) in sel_ids:
                d['sel'] += 1
            d['lo'] = amp if d['lo'] is None else min(d['lo'], amp)
            d['hi'] = amp if d['hi'] is None else max(d['hi'], amp)
        return sorted(acc.items(), key=lambda kv: -kv[1]['all'])

    def _targets(self, wt):
        """该音色在所选范围内的音符"""
        app = self.app
        scope = self.scope_var.get()
        if scope == 'sel':
            sel_ids = {id(n) for n in app.multi_selected_notes}
            return [n for n in app.sequence
                    if n.get('wavetype') == wt and id(n) in sel_ids]
        if scope == 'track':
            return [n for n in app.sequence
                    if n.get('wavetype') == wt and n.get('track', 0) == app.current_track]
        return [n for n in app.sequence if n.get('wavetype') == wt]

    def _plan(self):
        """(音色, 音符列表, 新振幅列表, 错误信息, 额外提示)"""
        wt = self._selected_timbre()
        if wt is None:
            return None, [], [], "请先在上表选一个音色", ""
        targets = self._targets(wt)
        if not targets:
            return wt, [], [], "这个范围里没有该音色的音符", ""

        hint = ""
        if self.mode_var.get() == 'mult':
            try:
                k = float(self.mult_var.get().strip())
            except ValueError:
                return wt, [], [], "倍数请填数字（0.5 = 减半，2 = 加倍）", ""
            if k <= 0:
                return wt, [], [], "倍数必须大于 0", ""
            old = [float(n.get('start_amp', 0.7)) for n in targets]
            new = [max(0.0, min(1.0, v * k)) for v in old]
            clipped = sum(1 for v in old if v * k > 1.0)
            if clipped:
                hint = "（%d 个超上限已截断）" % clipped
            if self.scale_rate_var.get():
                hint += "（变化率同时 ×%g）" % k
        else:
            try:
                v = float(self.set_var.get().strip())
            except ValueError:
                return wt, [], [], "幅值请填数字（0 ~ 1）", ""
            if not (0.0 <= v <= 1.0):
                return wt, [], [], "幅值需要落在 0 ~ 1 之间", ""
            new = [v] * len(targets)
        return wt, targets, new, "", hint

    def _selected_timbre(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    # ------------------------------------------------------------ 界面
    def _build(self):
        body = ttk.Frame(self, padding=(16, 14, 16, 10))
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text="音色（当前工程里出现过的）",
                  font=("", 10, "bold")).pack(anchor=tk.W)

        wrap = ttk.Frame(body)
        wrap.pack(fill=tk.X, pady=(6, 0))
        self.tree = ttk.Treeview(
            wrap, columns=("timbre", "n_all", "n_track", "n_sel", "amp"),
            show="headings", height=7, selectmode="browse")
        for cid, text, width, anchor in (
                ("timbre", "音色", 128, tk.W),
                ("n_all", "全部", 52, tk.CENTER),
                ("n_track", "当前轨", 58, tk.CENTER),
                ("n_sel", "选中", 48, tk.CENTER),
                ("amp", "当前振幅", 96, tk.CENTER)):
            self.tree.heading(cid, text=text)
            self.tree.column(cid, width=width, anchor=anchor, stretch=False)
        for wt, d in self._stats:
            self.tree.insert("", tk.END, iid=wt, values=(
                wt, d['all'], d['track'], d['sel'],
                "%.2f ~ %.2f" % (d['lo'], d['hi'])))
        self.tree.pack(fill=tk.X)
        self.tree.bind("<<TreeviewSelect>>", self._refresh)

        # 默认选中：正在编辑的音符的同音色，否则数量最多的那个
        default_wt = None
        if self.app.selected_note is not None:
            default_wt = self.app.selected_note.get('wavetype')
        if default_wt not in [wt for wt, _d in self._stats]:
            default_wt = self._stats[0][0]
        self.tree.selection_set(default_wt)
        self.tree.see(default_wt)

        # 范围
        rf = ttk.LabelFrame(body, text="范围", padding=8)
        rf.pack(fill=tk.X, pady=(10, 0))
        self.scope_var = tk.StringVar(value='all')
        n_total = len(self.app.sequence)
        n_sel = len(self.app.multi_selected_notes)
        track_name = self.app.get_track(self.app.current_track).get('name', '')
        ttk.Radiobutton(rf, text="全部音符（%d）" % n_total, value='all',
                        variable=self.scope_var, command=self._refresh).grid(
            row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(rf, text="仅当前音轨「%s」" % track_name, value='track',
                        variable=self.scope_var, command=self._refresh).grid(
            row=0, column=1, sticky=tk.W, padx=(12, 0))
        sel_btn = ttk.Radiobutton(rf, text="仅选中音符（%d）" % n_sel, value='sel',
                                  variable=self.scope_var, command=self._refresh)
        sel_btn.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        if n_sel == 0:
            sel_btn.configure(state=tk.DISABLED)

        # 方式
        mf = ttk.LabelFrame(body, text="方式", padding=8)
        mf.pack(fill=tk.X, pady=(10, 0))
        self.mode_var = tk.StringVar(value='mult')
        ttk.Radiobutton(mf, text="统一调节倍数", value='mult',
                        variable=self.mode_var, command=self._refresh).grid(
            row=0, column=0, sticky=tk.W)
        self.mult_var = tk.StringVar(value="1.00")
        self.mult_entry = ttk.Entry(mf, textvariable=self.mult_var, width=8,
                                    justify=tk.CENTER)
        self.mult_entry.grid(row=0, column=1, padx=(6, 4))
        ttk.Label(mf, text="× 原值（0.5 = 减半，2 = 加倍）",
                  foreground="#666").grid(row=0, column=2, sticky=tk.W)

        ttk.Radiobutton(mf, text="统一设定幅值", value='set',
                        variable=self.mode_var, command=self._refresh).grid(
            row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.set_var = tk.StringVar(value="0.70")
        self.set_entry = ttk.Entry(mf, textvariable=self.set_var, width=8,
                                   justify=tk.CENTER)
        self.set_entry.grid(row=1, column=1, padx=(6, 4), pady=(6, 0))
        ttk.Label(mf, text="全部改成这个值（0 ~ 1）",
                  foreground="#666").grid(row=1, column=2, sticky=tk.W, pady=(6, 0))

        self.scale_rate_var = tk.BooleanVar(value=False)
        self.rate_chk = ttk.Checkbutton(
            mf, text="同时按同一倍数缩放振幅变化率（amp_change_rate）",
            variable=self.scale_rate_var, command=self._refresh)
        self.rate_chk.grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

        # 预览
        self.preview_var = tk.StringVar()
        ttk.Label(body, textvariable=self.preview_var, wraplength=430,
                  justify=tk.LEFT, foreground="#444").pack(anchor=tk.W, pady=(10, 0))

        btns = ttk.Frame(body)
        btns.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(btns, text="取消", command=self.destroy).pack(side=tk.RIGHT)
        self.apply_btn = ttk.Button(btns, text="应用", command=self._apply)
        self.apply_btn.pack(side=tk.RIGHT, padx=(0, 6))

        for w in (self.mult_entry, self.set_entry):
            w.bind("<KeyRelease>", self._refresh)
        self.bind("<Return>", lambda e: self._apply())

        self._refresh()

    def _refresh(self, *_):
        mult = self.mode_var.get() == 'mult'
        self.mult_entry.configure(state=tk.NORMAL if mult else tk.DISABLED)
        self.set_entry.configure(state=tk.NORMAL if mult else tk.DISABLED)
        self.rate_chk.configure(state=tk.NORMAL if mult else tk.DISABLED)

        wt, targets, new, err, hint = self._plan()
        if err:
            self.preview_var.set(err)
            self.apply_btn.configure(state=tk.DISABLED)
            return
        old = [float(n.get('start_amp', 0.7)) for n in targets]
        self.preview_var.set(
            "影响 %d 个 %s 音符\n起始振幅 %.2f~%.2f → %.2f~%.2f%s"
            % (len(targets), wt, min(old), max(old), min(new), max(new), hint))
        self.apply_btn.configure(state=tk.NORMAL)

    def _apply(self):
        wt, targets, new, err, _hint = self._plan()
        if err:
            messagebox.showwarning("设定同一音色幅值", err, parent=self)
            return

        mult = self.mode_var.get() == 'mult'
        try:
            k = float(self.mult_var.get().strip())
        except ValueError:
            k = 1.0
        old = [float(n.get('start_amp', 0.7)) for n in targets]

        self.app.save_state()          # 撤销快照取“改之前”
        for n, v in zip(targets, new):
            n['start_amp'] = v         # 已在 _plan 里夹到 0~1
            if mult and self.scale_rate_var.get():
                n['amp_change_rate'] = n.get('amp_change_rate', 0.0) * k
        self.app.draw_notes()
        self.app.status_var.set(
            "宏操作：%s %d 个音符，起始振幅 %.2f~%.2f → %.2f~%.2f"
            % (wt, len(targets), min(old), max(old), min(new), max(new)))
        self.destroy()

    def _center(self, master):
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        try:
            x = master.winfo_rootx() + (master.winfo_width() - w) // 2
            y = master.winfo_rooty() + (master.winfo_height() - h) // 3
        except tk.TclError:
            x = y = 200
        self.geometry("+%d+%d" % (max(0, x), max(0, y)))


def meter_units(app):
    """当前拍号下 (每小节单位数, 每拍单位数)，单位 = 16 分音符

    与 set_time_signature 的口径一致：4/4 → (16, 4)、3/4 → (12, 4)、
    6/8 → (12, 2)。弹窗里的小节 / 拍换算都得走这里，
    不能在类里写死 16，否则换成 3/4 或 6/8 后小节号就全错了。
    """
    bar = float(getattr(app, 'units_per_measure', 16) or 16)
    ts = getattr(app, 'time_signature', None) or [4, 4]
    try:
        denominator = int(ts[1])
    except (IndexError, TypeError, ValueError):
        denominator = 4
    beat = 4.0 if denominator == 4 else 2.0
    return bar, beat


class SyncMoveDialog(tk.Toplevel):
    """宏操作：把一批音符整体平移，使最早的一个从指定的 16 分音符开始

    位置一律用「第 xx 个 16 分音符」表示，从 1 数起（第 1 个就是工程起点 0）。
    范围由菜单决定（全部 / 仅选中），对话框里只填目标位置并预览位移。
    """

    def __init__(self, app, scope, targets, parent=None):
        super().__init__(parent or app.root)
        self.app = app
        self.scope = scope
        self.targets = list(targets)
        self.title("同步移动音符")
        self.resizable(False, False)
        self.transient(parent or app.root)

        self._anchor = min(float(n.get('start', 0.0)) for n in self.targets)
        self._latest_end = max(float(n.get('start', 0.0)) + float(n.get('duration', 0.0))
                               for n in self.targets)

        self._build()
        self._center(parent or app.root)
        try:
            self.grab_set()
        except tk.TclError:
            pass
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ------------------------------------------------------------ 位置换算
    def _bar_text(self, units):
        """「第 b 小节 第 p 拍 第 s 个」，按当前拍号换算；不在网格上返回空串"""
        u = float(units)
        if abs(u - round(u)) > 1e-6:
            return ""
        bar_len, beat_len = meter_units(self.app)
        bar = int(u // bar_len) + 1
        beat = int((u % bar_len) // beat_len) + 1
        sub = int(u % beat_len) + 1
        return "第 %d 小节 第 %d 拍 第 %d 个" % (bar, beat, sub)

    def _pos_text(self, units, short=False):
        """把单位数写成「第 n 个 16 分音符（第 b 小节 第 p 拍 第 s 个）」

        short=True 只给「第 n 个」，用于预览行，避免一行塞太多字被折成三行。
        """
        u = float(units)
        if abs(u - round(u)) < 1e-6:
            n = int(round(u)) + 1
            if short:
                return "第 %d 个" % n
            bar_len, beat_len = meter_units(self.app)
            bar = int(u // bar_len) + 1
            beat = int((u % bar_len) // beat_len) + 1
            sub = int(u % beat_len) + 1
            return "第 %d 个 16 分音符（第 %d 小节 第 %d 拍 第 %d 个）" % (n, bar, beat, sub)
        return "第 %.2f 个（未对齐网格）" % (u + 1.0) if short else \
            "第 %.2f 个 16 分音符（未对齐网格）" % (u + 1.0)

    def _plan(self):
        """返回 (位移量, 错误信息)；位移 = 目标起点 - 当前最早起点"""
        try:
            raw = float(self.pos_var.get().strip())
        except ValueError:
            return None, "请填一个整数（第几个 16 分音符）"
        if abs(raw - round(raw)) > 1e-6:
            return None, "16 分音符的序号是整数，请填 1、2、3 …"
        if raw < 1:
            return None, "序号从 1 起算，不能小于 1"
        return float(int(round(raw)) - 1) - self._anchor, ""

    # ------------------------------------------------------------ 界面
    def _build(self):
        body = ttk.Frame(self, padding=(16, 14, 16, 10))
        body.pack(fill=tk.BOTH, expand=True)

        n = len(self.targets)
        head = ("范围：选中的 %d 个音符" % n if self.scope == 'sel'
                else "范围：全部 %d 个音符" % n)
        ttk.Label(body, text=head, font=("", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(body, foreground="#666", justify=tk.LEFT, wraplength=410,
                  text="所有目标音符平移同样的距离，彼此之间的间隔、时长和力度都不变。"
                  ).pack(anchor=tk.W, pady=(4, 0))

        info = ttk.LabelFrame(body, text="现在的位置", padding=8)
        info.pack(fill=tk.X, pady=(10, 0))
        self.now_var = tk.StringVar()
        ttk.Label(info, textvariable=self.now_var, justify=tk.LEFT).pack(anchor=tk.W)

        pos = ttk.LabelFrame(body, text="目标位置", padding=8)
        pos.pack(fill=tk.X, pady=(10, 0))
        line = ttk.Frame(pos)
        line.pack(anchor=tk.W)
        ttk.Label(line, text="让最早的音符从第").pack(side=tk.LEFT)
        self.pos_var = tk.StringVar(
            value="%d" % (int(round(self._anchor)) + 1))
        self.pos_entry = ttk.Entry(line, textvariable=self.pos_var, width=7,
                                   justify=tk.CENTER)
        self.pos_entry.pack(side=tk.LEFT, padx=4)
        ttk.Label(line, text="个 16 分音符开始（从 1 数起）").pack(side=tk.LEFT)

        # 把序号换算成小节 / 拍，省得用户自己数
        self.target_hint_var = tk.StringVar()
        ttk.Label(pos, textvariable=self.target_hint_var,
                  foreground="#777").pack(anchor=tk.W, pady=(4, 0))

        self.move_blocks_var = tk.BooleanVar(value=True)
        self.blocks_chk = ttk.Checkbutton(
            body, text="同时平移效果器块（让低通 / 混响继续盖住这些音符）",
            variable=self.move_blocks_var, command=self._refresh)
        self.blocks_chk.pack(anchor=tk.W, pady=(10, 0))
        if self.scope != 'all' or not self.app.effect_blocks:
            self.move_blocks_var.set(False)
            self.blocks_chk.configure(state=tk.DISABLED)

        self.preview_var = tk.StringVar()
        ttk.Label(body, textvariable=self.preview_var, wraplength=430,
                  justify=tk.LEFT, foreground="#444").pack(anchor=tk.W, pady=(10, 0))

        btns = ttk.Frame(body)
        btns.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(btns, text="取消", command=self.destroy).pack(side=tk.RIGHT)
        self.apply_btn = ttk.Button(btns, text="应用", command=self._apply)
        self.apply_btn.pack(side=tk.RIGHT, padx=(0, 6))

        self.pos_entry.bind("<KeyRelease>", self._refresh)
        self.bind("<Return>", lambda e: self._apply())
        self.pos_entry.select_range(0, tk.END)
        self.pos_entry.focus_set()

        self._refresh()

    def _refresh(self, *_):
        off_grid = abs(self._anchor - round(self._anchor)) > 1e-6
        self.now_var.set("最早音符：%s\n最晚结束：%s\n整段长度：%.2f 个单位（平移前后不变）%s"
                         % (self._pos_text(self._anchor),
                            self._pos_text(self._latest_end),
                            self._latest_end - self._anchor,
                            "\n最早音符没落在网格上，填整数会顺带把它对齐" if off_grid else ""))

        delta, err = self._plan()
        if err:
            self.target_hint_var.set("")
            self.preview_var.set(err)
            self.apply_btn.configure(state=tk.DISABLED)
            return

        target = self._anchor + delta
        self.target_hint_var.set("= " + self._bar_text(target) if self._bar_text(target)
                                 else "= 未对齐网格")

        if abs(delta) < 1e-9:
            self.preview_var.set("位置不变：最早音符已经在第 %d 个 16 分音符"
                                 % (int(round(self._anchor)) + 1))
        else:
            # 起点不在网格上时位移会带小数，这时不再写“+0 个 16 分音符”误导人
            whole = int(round(delta))
            suffix = ("（%+d 个 16 分音符）" % whole
                      if abs(delta - whole) < 1e-6 else "")
            self.preview_var.set(
                "平移 %+.2f 个单位%s\n最早音符 %s → %s"
                % (delta, suffix,
                   self._pos_text(self._anchor, short=True),
                   self._pos_text(self._anchor + delta, short=True) + " 16 分音符"))
        self.apply_btn.configure(state=tk.NORMAL)

    # ------------------------------------------------------------ 应用
    def _apply(self):
        delta, err = self._plan()
        if err:
            messagebox.showwarning("同步移动音符", err, parent=self)
            return

        app = self.app
        old_anchor = self._anchor
        app.save_state()          # 快照取“改之前”
        for note in self.targets:
            note['start'] = float(note.get('start', 0.0)) + delta

        moved_blocks = 0
        if self.move_blocks_var.get():
            for blk in app.effect_blocks:
                # 左边界夹到 0：整体左移时色块不会跑到画布外，尾巴跟着一起挪
                blk['start'] = max(0.0, float(blk['start']) + delta)
                blk['end'] = float(blk['end']) + delta
                moved_blocks += 1

        app.draw_notes()
        if moved_blocks:
            app.draw_effect_blocks()
        app.status_var.set(
            "宏操作：同步移动 %d 个音符 %+.2f 个单位，最早音符 第 %d → 第 %d 个 16 分音符%s"
            % (len(self.targets), delta, int(round(old_anchor)) + 1,
               int(round(old_anchor + delta)) + 1,
               "（含 %d 个效果器块）" % moved_blocks if moved_blocks else ""))
        self.destroy()

    def _center(self, master):
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        try:
            x = master.winfo_rootx() + (master.winfo_width() - w) // 2
            y = master.winfo_rooty() + (master.winfo_height() - h) // 3
        except tk.TclError:
            x = y = 200
        self.geometry("+%d+%d" % (max(0, x), max(0, y)))


class SegmentCopyDialog(tk.Toplevel):
    """编辑：按时间段选中音符，并把这批复刻粘贴到新的起点

    三个输入框统一用「第 n 个 16 分音符」，从 1 数起：
    第 1 个就是工程起点 0，一个 16 分音符占 1 个单位。
    改完起止序号立刻在画布上选中这一段（紫色边框），
    「复制并粘贴」把这批音符深拷贝一份整体平移，使复制品里
    最早的那个音符正好落在「粘贴到第 n 个」，组内间距与时长不变。
    小节 / 拍的换算走当前拍号（4/4、3/4、6/8 各不相同）。
    """

    def __init__(self, app, parent=None):
        super().__init__(parent or app.root)
        self.app = app
        self.seg_notes = []
        self.title("按时间段选中 / 复制粘贴")
        self.resizable(False, False)
        self.transient(parent or app.root)

        # 默认值：有选区就照选区的起止，否则从光标处往后 16 个
        sel = app._selection_notes()
        if sel:
            first = min(float(n.get('start', 0.0)) for n in sel)
            last = max(float(n.get('start', 0.0)) + float(n.get('duration', 0.0))
                       for n in sel)
            a = int(math.floor(first + 1e-9)) + 1
            b = max(a, int(math.ceil(last - 1e-9)))
        else:
            a = int(math.floor(app.cursor_unit + 1e-9)) + 1
            b = a + 15
        self.start_var = tk.StringVar(value=str(a))
        self.end_var = tk.StringVar(value=str(b))
        self.dest_var = tk.StringVar(value=str(b + 1))

        self._build()
        self._center(parent or app.root)
        try:
            self.grab_set()
        except tk.TclError:
            pass
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ------------------------------------------------------------ 输入解析
    def _int_value(self, var, name):
        """把输入框读成正整数序号，第二个返回值是错误信息"""
        try:
            raw = float(str(var.get()).strip())
        except ValueError:
            return None, "「%s」请填一个整数（第几个 16 分音符）" % name
        if abs(raw - round(raw)) > 1e-6:
            return None, "「%s」16 分音符的序号是整数，请填 1、2、3 …" % name
        if raw < 1:
            return None, "「%s」序号从 1 数起，不能小于 1" % name
        return int(round(raw)), ""

    def _span(self):
        """返回 (起点序号, 终点序号, 错误信息)"""
        a, err = self._int_value(self.start_var, "起点")
        if err:
            return None, None, err
        b, err = self._int_value(self.end_var, "终点")
        if err:
            return None, None, err
        if b < a:
            return None, None, "终点不能早于起点"
        return a, b, ""

    def _range_notes(self, a, b):
        """起点落在第 a 个到第 b 个 16 分音符里的音符（含两端）"""
        lo, hi = float(a - 1), float(b)
        return [n for n in self.app.sequence
                if lo - 1e-6 <= float(n.get('start', 0.0)) < hi - 1e-6]

    def _bar_text(self, units):
        """「第 b 小节 第 p 拍 第 s 个」，按当前拍号换算；不在网格上返回空串"""
        u = float(units)
        if abs(u - round(u)) > 1e-6:
            return ""
        bar_len, beat_len = meter_units(self.app)
        bar = int(u // bar_len) + 1
        beat = int((u % bar_len) // beat_len) + 1
        sub = int(u % beat_len) + 1
        return "第 %d 小节 第 %d 拍 第 %d 个" % (bar, beat, sub)

    def _pos_text(self, units, short=False):
        """把单位数写成「第 n 个 16 分音符（第 b 小节 …）」"""
        u = float(units)
        if abs(u - round(u)) < 1e-6:
            n = int(round(u)) + 1
            if short:
                return "第 %d 个" % n
            bar_len, beat_len = meter_units(self.app)
            bar = int(u // bar_len) + 1
            beat = int((u % bar_len) // beat_len) + 1
            sub = int(u % beat_len) + 1
            return "第 %d 个 16 分音符（第 %d 小节 第 %d 拍 第 %d 个）" % (n, bar, beat, sub)
        return ("第 %.2f 个（未对齐网格）" % (u + 1.0) if short
                else "第 %.2f 个 16 分音符（未对齐网格）" % (u + 1.0))

    # ------------------------------------------------------------ 界面
    def _build(self):
        body = ttk.Frame(self, padding=(16, 14, 16, 10))
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text="按时间段选中音符，并复制到新的起点",
                  font=("", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(body, foreground="#666", justify=tk.LEFT, wraplength=440,
                  text="序号都从 1 数起，第 1 个 16 分音符就是工程起点。"
                       "改完上面两个数字，画布上会立刻出现紫色框选。"
                  ).pack(anchor=tk.W, pady=(4, 0))

        seg = ttk.LabelFrame(body, text="时间段", padding=8)
        seg.pack(fill=tk.X, pady=(10, 0))
        line1 = ttk.Frame(seg)
        line1.pack(anchor=tk.W)
        ttk.Label(line1, text="从第").pack(side=tk.LEFT)
        self.start_entry = ttk.Entry(line1, textvariable=self.start_var, width=7,
                                     justify=tk.CENTER)
        self.start_entry.pack(side=tk.LEFT, padx=4)
        ttk.Label(line1, text="个 到第").pack(side=tk.LEFT)
        self.end_entry = ttk.Entry(line1, textvariable=self.end_var, width=7,
                                   justify=tk.CENTER)
        self.end_entry.pack(side=tk.LEFT, padx=4)
        ttk.Label(line1, text="个 16 分音符（含两端）").pack(side=tk.LEFT)

        self.seg_hint_var = tk.StringVar()
        ttk.Label(seg, textvariable=self.seg_hint_var, foreground="#777",
                  justify=tk.LEFT, wraplength=440).pack(anchor=tk.W, pady=(4, 0))

        dest = ttk.LabelFrame(body, text="粘贴到起点", padding=8)
        dest.pack(fill=tk.X, pady=(10, 0))
        line2 = ttk.Frame(dest)
        line2.pack(anchor=tk.W)
        ttk.Label(line2, text="粘贴到第").pack(side=tk.LEFT)
        self.dest_entry = ttk.Entry(line2, textvariable=self.dest_var, width=7,
                                    justify=tk.CENTER)
        self.dest_entry.pack(side=tk.LEFT, padx=4)
        ttk.Label(line2, text="个 16 分音符（最早的音符落在这里）").pack(side=tk.LEFT)

        self.dest_hint_var = tk.StringVar()
        ttk.Label(dest, textvariable=self.dest_hint_var, foreground="#777").pack(
            anchor=tk.W, pady=(4, 0))

        self.preview_var = tk.StringVar()
        ttk.Label(body, textvariable=self.preview_var, wraplength=440,
                  justify=tk.LEFT, foreground="#444").pack(anchor=tk.W, pady=(10, 0))

        btns = ttk.Frame(body)
        btns.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(btns, text="取消", command=self.destroy).pack(side=tk.RIGHT)
        self.paste_btn = ttk.Button(btns, text="复制并粘贴", command=self.on_paste)
        self.paste_btn.pack(side=tk.RIGHT, padx=(0, 6))
        self.select_btn = ttk.Button(btns, text="仅选中", command=self.on_select)
        self.select_btn.pack(side=tk.RIGHT, padx=(0, 6))

        for entry in (self.start_entry, self.end_entry, self.dest_entry):
            entry.bind("<KeyRelease>", self._refresh)
        self.bind("<Return>", lambda _e: self.on_paste())
        self.start_entry.select_range(0, tk.END)
        self.start_entry.focus_set()

        self._refresh()

    # ------------------------------------------------------------ 实时预览
    def _refresh(self, *_):
        a, b, err = self._span()
        if err:
            self.seg_notes = []
            self.seg_hint_var.set(err)
            self.select_btn.configure(state=tk.DISABLED)
            self.paste_btn.configure(state=tk.DISABLED)
            self.dest_hint_var.set("")
            self.preview_var.set("")
            return

        notes = self._range_notes(a, b)
        self.seg_notes = notes
        if notes:
            pitches = [int(n.get('pitch', 0)) for n in notes]
            kinds = []
            for n in notes:
                wt = n.get('wavetype')
                if wt not in kinds:
                    kinds.append(wt)
            self.seg_hint_var.set(
                "这一段共 %d 个音符，音高 %s–%s，涉及 %d 种音色：%s"
                % (len(notes),
                   self.app.get_note_from_pitch(min(pitches)),
                   self.app.get_note_from_pitch(max(pitches)),
                   len(kinds), " / ".join(str(k) for k in kinds)))
            # 自动选中：改完序号立刻在画布上看到紫色框
            self.app.multi_selected_notes = notes
            self.app.selected_note = None
            self.app._apply_highlight()
        else:
            self.seg_hint_var.set("这一段里没有音符（序号范围 %d–%d）" % (a, b))
            self.app.multi_selected_notes = []
            self.app.selected_note = None
            self.app._apply_highlight()

        self.select_btn.configure(state=tk.NORMAL if notes else tk.DISABLED)

        c, derr = self._int_value(self.dest_var, "粘贴到")
        if derr:
            self.dest_hint_var.set(derr)
            self.paste_btn.configure(state=tk.DISABLED)
            self.preview_var.set("")
            return
        self.dest_hint_var.set("= " + self._bar_text(float(c - 1))
                               if self._bar_text(float(c - 1)) else "= 未对齐网格")
        if not notes:
            self.paste_btn.configure(state=tk.DISABLED)
            self.preview_var.set("")
            return

        base = min(float(n.get('start', 0.0)) for n in notes)
        end = max(float(n.get('start', 0.0)) + float(n.get('duration', 0.0))
                  for n in notes)
        target = float(c - 1)
        span = end - base
        self.preview_var.set(
            "将复制 %d 个音符：%s → %s\n整段长度 %.2f 个单位（复制前后不变）"
            % (len(notes), self._pos_text(base, short=True),
               self._pos_text(target, short=True), span))
        self.paste_btn.configure(state=tk.NORMAL)

    # ------------------------------------------------------------ 应用
    def on_select(self):
        """只做选中，不改音符（方便选完再用别的菜单操作）"""
        a, b, err = self._span()
        if err:
            messagebox.showwarning("按时间段选中", err, parent=self)
            return
        notes = self._range_notes(a, b)
        if not notes:
            messagebox.showinfo("按时间段选中",
                                "第 %d–%d 个 16 分音符之间没有音符" % (a, b), parent=self)
            return
        self.app.multi_selected_notes = notes
        self.app.selected_note = None
        self.app._apply_highlight()
        self.app.status_var.set(
            "已选中第 %d–%d 个 16 分音符内的 %d 个音符" % (a, b, len(notes)))

    def on_paste(self):
        """复制这一段并整体平移，使最早的音符落在指定序号"""
        a, b, err = self._span()
        if err:
            messagebox.showwarning("复制并粘贴", err, parent=self)
            return
        c, err = self._int_value(self.dest_var, "粘贴到")
        if err:
            messagebox.showwarning("复制并粘贴", err, parent=self)
            return
        notes = self._range_notes(a, b)
        if not notes:
            messagebox.showinfo("复制并粘贴",
                                "第 %d–%d 个 16 分音符之间没有音符" % (a, b), parent=self)
            return

        app = self.app
        base = min(float(n.get('start', 0.0)) for n in notes)
        target = float(c - 1)
        delta = target - base

        app.save_state()          # 快照取“复制之前”
        pasted = []
        for src in notes:
            item = copy.deepcopy(src)
            item.pop('rect_id', None)
            item.pop('text_id', None)
            item['start'] = float(item.get('start', 0.0)) + delta
            app.sequence.append(item)     # 音轨 / 音色 / 力度原样保留
            pasted.append(item)

        app.multi_selected_notes = pasted
        app.selected_note = None
        app.draw_notes()
        app.status_var.set(
            "已把第 %d–%d 个 16 分音符内的 %d 个音符复制到第 %d 个（%+.2f 个单位）"
            % (a, b, len(pasted), int(round(target)) + 1, delta))
        self.destroy()

    def _center(self, master):
        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        try:
            x = master.winfo_rootx() + (master.winfo_width() - w) // 2
            y = master.winfo_rooty() + (master.winfo_height() - h) // 3
        except tk.TclError:
            x = y = 200
        self.geometry("+%d+%d" % (max(0, x), max(0, y)))


if __name__ == "__main__":
    root = tk.Tk()
    app = PianoRollEditor(root)
    
    # 安全退出处理
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()