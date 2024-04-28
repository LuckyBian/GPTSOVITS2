from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
import matplotlib.pyplot as plt

def split_audio(file_path, segment_length=10):
    audio, sr = librosa.load(file_path, sr=None)
    segment_length_samples = segment_length * sr
    segments = [audio[i:i + segment_length_samples] for i in range(0, len(audio), segment_length_samples)]
    return segments, sr

def process_directory(directory):
    data_dir = Path(directory)
    wav_fpaths = list(data_dir.glob("**/*.wav"))
    segments = []
    wav_fpaths_expanded = []

    print(f"Splitting and preprocessing {len(wav_fpaths)} WAV files.")
    for wav_fpath in tqdm(wav_fpaths, unit="files"):
        audio_segments, _ = split_audio(wav_fpath)
        for segment in audio_segments:
            processed_segment = preprocess_wav(segment)
            segments.append(processed_segment)
            wav_fpaths_expanded.append(wav_fpath)

    encoder = VoiceEncoder()
    print("Loaded the voice encoder model.")
    embeds = np.array([encoder.embed_utterance(wav) for wav in segments])
    speakers = np.array([fpath.parent.name for fpath in wav_fpaths_expanded])
    names = np.array([f"{fpath.stem}_{i}" for fpath, i in zip(wav_fpaths_expanded, range(len(segments)))])

    return embeds, speakers, names

def analyze_speakers(embeds, speakers, names, real_speaker_dir_name="true"):
    num_real_speakers = np.sum(speakers == real_speaker_dir_name)
    num_samples = min(num_real_speakers, 6)  # Choose up to 6 samples, or fewer if not enough available

    if num_samples > 0:
        gt_indices = np.random.choice(np.where(speakers == real_speaker_dir_name)[0], num_samples, replace=False)
        mask = np.zeros(len(embeds), dtype=bool)
        mask[gt_indices] = True
        gt_embeds = embeds[mask]
        gt_names = names[mask]
        gt_speakers = speakers[mask]
        embeds, speakers, names = embeds[~mask], speakers[~mask], names[~mask]

        scores = (gt_embeds @ embeds.T).mean(axis=0)
        sort = np.argsort(scores)[::-1]
        scores, names, speakers = scores[sort], names[sort], speakers[sort]

        # Plotting
        fig, _ = plt.subplots(figsize=(10, 6))
        indices = np.arange(len(scores))
        plt.axhline(0.75, ls="dashed", c="black")
        plt.bar(indices, scores, color=np.where(speakers == "gen", "red", "green"))
        plt.xticks(indices, names, rotation="vertical", fontsize=8)
        plt.xlabel("Names")
        plt.ylim(0, 1)
        plt.ylabel("Similarity to ground truth")
        fig.subplots_adjust(bottom=0.25)
        plt.show()
    else:
        print("Not enough 'real' speakers to perform analysis.")

# Example usage:
embeds, speakers, names = process_directory("/Users/bianweizhenbian/Documents/GPTSOVITS2/myoutput")
analyze_speakers(embeds, speakers, names)
