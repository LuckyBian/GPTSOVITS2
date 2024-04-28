import os
import argparse
import librosa
from pystoi import stoi

def calculate_stoi(clean, denoised, target_sr=16000):
    # 确保两个音频文件长度相同
    min_len = min(len(clean), len(denoised))
    clean = clean[:min_len]
    denoised = denoised[:min_len]
    
    # 计算STOI分数
    score = stoi(clean, denoised, target_sr, extended=False)
    return score

def calculate_directory_stoi(clean_dir, denoised_dir, target_sr=16000):
    scores = {}
    # 获取清晰音频文件夹中所有音频文件名
    audio_files = sorted(os.listdir(clean_dir))

    for audio_file in audio_files:
        if audio_file.endswith('.wav'):
            clean_path = os.path.join(clean_dir, audio_file)
            denoised_path = os.path.join(denoised_dir, audio_file)

            # 加载音频文件
            clean_audio, _ = librosa.load(clean_path, sr=target_sr)
            denoised_audio, _ = librosa.load(denoised_path, sr=target_sr)

            # 计算STOI分数
            score = calculate_stoi(clean_audio, denoised_audio, target_sr)
            scores[audio_file] = score

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量计算STOI分数以评估音频文件夹中音频的质量")
    parser.add_argument("--clean_dir", type=str, default="/Users/bianweizhenbian/Documents/GPTSOVITS2/myoutput/true", help="包含原始清晰音频文件的目录路径")
    parser.add_argument("--denoised_dir", type=str, default="/Users/bianweizhenbian/Documents/GPTSOVITS2/myoutput/gen", help="包含合成/去噪后音频文件的目录路径")
    parser.add_argument("--sr", type=int, default=16000, help="目标音频的采样率（默认：16000Hz）")
    
    args = parser.parse_args()
    
    scores = calculate_directory_stoi(args.clean_dir, args.denoised_dir, args.sr)
    for file_name, score in scores.items():
        print(f"{file_name}: STOI分数 = {score}")
