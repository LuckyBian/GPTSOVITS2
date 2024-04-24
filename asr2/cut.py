from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import librosa
import numpy as np
from scipy.spatial.distance import cosine
from pydub.silence import split_on_silence, detect_nonsilent

ref_audio_paths = {
    '1': '/home/weizhenbian/mycode/asr2/ref/1.wav',
    '2': '/home/weizhenbian/mycode/asr2/ref/2.wav',
    '3': '/home/weizhenbian/mycode/asr2/ref/3.wav',
    '4': '/home/weizhenbian/mycode/asr2/ref/4.wav',
    '5': '/home/weizhenbian/mycode/asr2/ref/5.wav'
}

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


def find_and_save_vocal_chunks(audio_path, output_dir, min_duration=0.5):
    sound = AudioSegment.from_wav(audio_path)
    nonsilent_parts = detect_nonsilent(
        sound,
        min_silence_len=100,
        silence_thresh=-40,
        seek_step=1
    )
    
    output_lines = []
    for i, (start_ms, end_ms) in enumerate(nonsilent_parts):
        duration = (end_ms - start_ms) / 1000.0  # Convert from ms to s
        if duration >= min_duration:
            chunk = sound[start_ms:end_ms]
            output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_path))[0] + f"_chunk_{i+1}.wav")
            chunk.export(output_file, format="wav")
            role = compare_audio(ref_audio_paths, output_file)
            output_lines.append(f"{output_file}|{start_ms / 1000.0:.2f}|{duration:.2f}|{role}")

    output_txt_file = os.path.join(output_dir, os.path.splitext(os.path.basename(audio_path))[0] + "_timestamps_roles.txt")
    with open(output_txt_file, "w") as f:
        f.write("\n".join(output_lines))

    return output_txt_file

# Example usage
output_dir = '/home/weizhenbian/mycode/asr2/cut'
audio_path = '/home/weizhenbian/mycode/asr2/in/mix.wav'
txt_file_path = find_and_save_vocal_chunks(audio_path, output_dir)
print(f"Timestamps saved to: {txt_file_path}")