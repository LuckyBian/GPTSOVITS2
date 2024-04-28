import os
import librosa
import argparse
import numpy as np

def calculate_sisdr(reference_signal, estimated_signal):
    # 确保两个信号长度一致
    min_len = min(len(reference_signal), len(estimated_signal))
    reference_signal = reference_signal[:min_len]
    estimated_signal = estimated_signal[:min_len]

    # SI-SDR计算
    alpha = np.dot(estimated_signal, reference_signal) / np.dot(reference_signal, reference_signal)
    target_signal = alpha * reference_signal
    noise = estimated_signal - target_signal
    sisdr = 10 * np.log10(np.dot(target_signal, target_signal) / np.dot(noise, noise))
    return sisdr

def calculate_directory_sisdr(true_dir, gen_dir):
    scores = []
    # 获取 true 文件夹中所有音频文件名
    audio_files = sorted(os.listdir(true_dir))

    for audio_file in audio_files:
        if audio_file.endswith('.wav'):
            ref_path = os.path.join(true_dir, audio_file)
            est_path = os.path.join(gen_dir, audio_file)

            # 加载音频文件
            reference_signal, _ = librosa.load(ref_path, sr=None)
            estimated_signal, _ = librosa.load(est_path, sr=None)

            # 计算SI-SDR
            sisdr_value = calculate_sisdr(reference_signal, estimated_signal)
            scores.append((audio_file, sisdr_value))

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量计算两个目录中音频文件的SI-SDR")
    parser.add_argument("--true_dir", type=str, default="/Users/bianweizhenbian/Documents/GPTSOVITS2/myoutput/true", help="包含参考音频文件的目录路径")
    parser.add_argument("--gen_dir", type=str, default="/Users/bianweizhenbian/Documents/GPTSOVITS2/myoutput/gen", help="包含生成音频文件的目录路径")

    args = parser.parse_args()

    # 处理目录并计算SI-SDR
    results = calculate_directory_sisdr(args.true_dir, args.gen_dir)

    # 打印结果
    for file_name, sisdr_value in results:
        print(f"{file_name}: SI-SDR = {sisdr_value} dB")
