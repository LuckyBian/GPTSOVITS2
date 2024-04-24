import librosa
import numpy as np
from scipy.spatial.distance import cosine

def get_mfcc(audio_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    averaged_mfcc = np.mean(mfcc, axis=1)
    return averaged_mfcc

def compare_audio(ref_audio_paths, test_audio_path):
    test_mfcc = get_mfcc(test_audio_path)
    similarities = {}
    for role, path in ref_audio_paths.items():
        ref_mfcc = get_mfcc(path)
        similarity = 1 - cosine(ref_mfcc, test_mfcc)
        similarities[role] = similarity
    return max(similarities, key=similarities.get)

# 示例参考音频路径
ref_audio_paths = {
    '1': '/home/weizhenbian/mycode/asr2/ref/1.wav',
    '2': '/home/weizhenbian/mycode/asr2/ref/2.wav',
    '3': '/home/weizhenbian/mycode/asr2/ref/3.wav',
    '4': '/home/weizhenbian/mycode/asr2/ref/4.wav',
    '5': '/home/weizhenbian/mycode/asr2/ref/5.wav'
}
