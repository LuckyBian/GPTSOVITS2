import os
import pandas as pd
import soundfile as sf
import librosa
from pesq import pesq

def process_directory(true_dir, gen_dir):
    scores = []

    # 获取 true 文件夹中所有音频文件名
    audio_files = sorted(os.listdir(true_dir))
    
    for audio_file in audio_files:
        if audio_file.endswith('.wav'):
            ref_path = os.path.join(true_dir, audio_file)
            deg_path = os.path.join(gen_dir, audio_file)

            # 读取参考音频和退化音频
            ref_audio, ref_rate = sf.read(ref_path)
            deg_audio, deg_rate = sf.read(deg_path)

            # 确保音频为单通道
            if ref_audio.ndim > 1:
                ref_audio = ref_audio[:, 0]  # 选择第一个通道
            if deg_audio.ndim > 1:
                deg_audio = deg_audio[:, 0]  # 选择第一个通道

            # 统一采样率到 16000 Hz
            target_rate = 16000
            if ref_rate != target_rate:
                ref_audio = librosa.resample(ref_audio.astype('float32'), orig_sr=ref_rate, target_sr=target_rate)
            if deg_rate != target_rate:
                deg_audio = librosa.resample(deg_audio.astype('float32'), orig_sr=deg_rate, target_sr=target_rate)

            # 计算 PESQ 分数
            score = pesq(target_rate, ref_audio, deg_audio, 'wb')
            scores.append((audio_file[:-4], score))  # 去除 .wav 后缀并保存分数

    # 创建 DataFrame 并计算平均分数
    df = pd.DataFrame(scores, columns=['Index', 'Score'])
    avg_score = df['Score'].mean()
    df.loc['Average'] = pd.Series({'Score': avg_score})  # 在 DataFrame 末尾添加平均分数行

    return df

# 文件夹路径
true_dir = '/Users/bianweizhenbian/Documents/GPTSOVITS2/myoutput/true'
gen_dir = '/Users/bianweizhenbian/Documents/GPTSOVITS2/myoutput/gen'

# 处理目录并打印结果
results_df = process_directory(true_dir, gen_dir)
print(results_df)
