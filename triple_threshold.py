import numpy as np
import librosa
import soundfile as sf
import wave
import matplotlib.pyplot as plt
from pydub import AudioSegment
from matplotlib import font_manager
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter


def double_threshold(improved,class_name,filename,num):
    wlen = 512
    inc = 256
    improved = improved   #1是改进
    f = wave.open(filename, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    wave_data = np.frombuffer(str_data, dtype=np.short)
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))  #归一化
    signal_length = len(wave_data)  # 信号总长度
    if signal_length <= wlen:  # 若信号长度小于一个帧的长度，则帧数定义为1
        nf = 1
    else:  # 否则，计算帧的总数量
        nf = int(np.ceil((1.0 * signal_length - wlen + inc) / inc))

    pad_length = int((nf - 1) * inc + wlen)  # 所有帧加起来总的铺平后的长度
    zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于sFFT中的扩充数组操作
    pad_signal = np.concatenate((wave_data, zeros))  # 填补后的信号记为pad_signal
    indices = np.tile(np.arange(0, wlen), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc),
                                                             (wlen, 1)).T  # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
    frames = pad_signal[indices]  # 得到帧信号
    windown = np.hanning(wlen)
    # 计算每一帧的短时能量
    ste = np.zeros(nf)  # 存储每一帧的短时能量
    for i in range(0, nf):
        a = frames[i:i + 1]
        b = a[0] * windown
        c = np.square(b)
        ste[i] = np.sum(c)  # 求和得到能量
    min_ste = np.min(ste)
    average_ste = np.mean(ste)

    # 计算每一帧的短时过零率
    stz = np.zeros(nf)
    for i in range(nf):
        a = frames[i:i + 1]
        b = windown * a[0]
        for j in range(wlen - 1):
            if b[j] * b[j + 1] < 0:
                stz[i] = stz[i] + 1

    min_stz = np.min(stz)
    average_stz = np.mean(stz)

    # 设置能量阈值和过零率阈值
    a = 0.10
    b = 0.10
    c = 0.5
    d = 0.5
    # ste_low = min_ste + 0.08*(average_ste - min_ste)
    ste_low = a * average_ste
    # stz_low = min_stz + 0.08*(average_stz - min_stz)
    stz_low = b * average_stz
    # ste_high = min_ste + 0.5*(average_ste - min_ste)
    ste_high = c * average_ste
    # stz_high = min_stz + 0.5*(average_stz - min_stz)
    stz_high = d * average_stz
    # print(ste_low,stz_low)
    # print(ste_high,stz_high)
    print(a,b,c,d)
    print(ste_low, ste_high, stz_low, stz_high)
    # 遍历每个帧，判断能量是否超过阈值，截取声音事件
    segment_start = None
    segments = []
    flag = 0  # 判断是否有起始点

    previous_segment_end = 0  # 上一个声音事件的结束帧
    # 双阈值法
    for i in range(0, nf):
        # print(ste[i],stz[i])
        if ste[i] > ste_low and stz[i] > stz_low:  # 判断低阈值
            # print(1)
            if flag == 0:
                strat_ready = i
                k = 0  # 判断后续帧是否大于高阈值
                if i + 5 < nf:
                    for j in range(i, i + 5):  # 如果后续帧都高于低阈值
                        if ste[j] > ste_low and stz[j] > stz_low:
                            k += 1
                    for x in range(i, i + 5):  # 且有一帧高于高阈值，则记录起始点
                        if k == 5:
                            if ste[x] > ste_high and stz[x] > stz_high:
                                if segment_start is None:
                                    segment_start = strat_ready
                                    flag = 1
                                    break
        elif segment_start is not None:
            # print(ste[i],stz[i])
            segment_end = i
            if segment_end - segment_start > 5 and segment_end - segment_start < 20:  # 判断持续时间
                if improved == 1:
                    if len(segments) >= 1:
                        if segment_start - previous_segment_end < 50:  # 判断本个声音片段的开始与上个声音片段的结尾间隔:多个声音组合成一个完整声
                            segments[-1] = (segments[-1][0], segment_end)
                        else:
                            segments.append((segment_start, segment_end))
                    else:
                        segments.append((segment_start, segment_end))
                    previous_segment_end = segment_end
                else:
                    segments.append((segment_start, segment_end))
            segment_start = None
            flag = 0

        # 显示截取声音区间
        # 读取声音文件（可以根据你的需求选择不同的音频加载工具）
    audio_data, sample_rate = sf.read(filename)

    title_font = font_manager.FontProperties(family='serif', style='normal', weight='bold', size=30)
    ticks_font = font_manager.FontProperties(family='serif', style='normal', weight='bold', size=30)
    # 获取声音信号的时间轴
    time = np.arange(0, len(audio_data)) / sample_rate
    # 绘制声音事件截取段
    plt.figure(figsize=(12, 9))
    plt.plot(time, audio_data, c=(70 / 255, 130 / 255, 190 / 255), lw=1)
    plt.xlabel("Time/s", fontproperties=title_font)
    plt.ylabel("Amplitude", fontproperties=title_font)
    plt.xticks(fontproperties=ticks_font)
    plt.yticks(fontproperties=ticks_font)
    # 获取当前轴对象
    ax = plt.gca()
    # 修改图框的线条粗细
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    plt.title("Sound waveform", fontproperties=title_font)

    # 定义格式化函数
    def format_func(value, tick_number):
        return f"{value:.1f}"

    # 应用格式化函数到 x 轴
    ax.xaxis.set_major_formatter(FuncFormatter(format_func))
    ax.yaxis.set_major_formatter(FuncFormatter(format_func))
    # plt.title(f"{a},{b},{c},{d}")
    for segment in segments:
        plt.axvline(x=segment[0] * (inc * 1.0 / framerate) / 2, color="r", linestyle="--",linewidth=3)
        plt.axvline(x=segment[1] * (inc * 1.0 / framerate) / 2, color="g", linestyle="--",linewidth=3)
        # print(segment[0],segment[1])
    plt.grid(False)
    plt.title("Sound waveform",fontproperties=title_font)
    if improved == 0:
        plt.savefig(f"sample\双阈值三阈值法对比结果\双阈值\{class_name}_double.jpg",dpi = 400)
    else:
        plt.savefig(f"sample\双阈值三阈值法对比结果\三阈值\{class_name}_triple.jpg",dpi = 400)

    plt.show()

    # 输出声音事件截取时间段
    for segment in segments:
        start = (segment[0] * (inc * 1.0 / framerate)) / 2
        end = (segment[1] * (inc * 1.0 / framerate)) / 2
        print("声音事件截取段：{:.2f}s - {:.2f}s,时长为:{:.2f}s, {} - {}".format(start, end, end-start,segment[0], segment[1]))
    audio_file = AudioSegment.from_wav(filename)

    # 声音事件截取段的时间范围，以毫秒为单位
    # output_audio = AudioSegment.empty()
    # 遍历声音事件截取段，将每个段截取并添加到输出音频中
    # for i, segment in enumerate(segments):
    #     start_time, end_time = segment
    #     start_time = ((segment[0] * (inc * 1.0 / framerate)) /2) *1000
    #     end_time = ((segment[1] * (inc * 1.0 / framerate)) /2) *1000
    #     segment_audio = audio_file[start_time:end_time]
    #     output_filename = f"chick_sound/audio_predict/predict{num}/sound/{class_name}/{class_name}_{i+1}.wav"
    #     segment_audio.export(output_filename, format="wav")
    # print(len(segments))
    return segments


#截取预测样本
for i in range(1):
    class_name = "snore"
    improved = 1
    filename = f"sample/双阈值三阈值法对比结果/{class_name}.wav"   #拿这个样本来得到截取区间
    double_threshold(improved,class_name=class_name,filename=filename,num=i+1)