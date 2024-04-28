from resemblyzer import preprocess_wav, VoiceEncoder
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
from pathlib import Path
from tqdm import tqdm

# 更新音频输入路径
true_voice_path = Path("/Users/bianweizhenbian/Documents/GPTSOVITS2/myoutput/gen")
demo_voice_path = Path("/Users/bianweizhenbian/Documents/GPTSOVITS2/myoutput/true")

encoder = VoiceEncoder("cpu")

# 获取两个文件夹中的所有音频文件路径
true_wav_fpaths = sorted(list(true_voice_path.glob("**/*.wav")))  # 确保路径按文件名排序
demo_wav_fpaths = sorted(list(demo_voice_path.glob("**/*.wav")))  # 确保路径按文件名排序

# 为每个说话者处理音频文件
true_speaker_wavs = {f"True_{i}": preprocess_wav(wav_fpath) for i, wav_fpath in enumerate(tqdm(true_wav_fpaths, "Preprocessing True Wavs", len(true_wav_fpaths), unit="wavs"))}
demo_speaker_wavs = {f"Gen_{i}": preprocess_wav(wav_fpath) for i, wav_fpath in enumerate(tqdm(demo_wav_fpaths, "Preprocessing Demo Wavs", len(demo_wav_fpaths), unit="wavs"))}

# 计算嵌入
embeds_a = np.array([encoder.embed_utterance(wav) for wav in true_speaker_wavs.values()])
embeds_b = np.array([encoder.embed_utterance(wav) for wav in demo_speaker_wavs.values()])

print("Shape of embeddings: %s" % str(embeds_a.shape))

# 计算两组嵌入之间的相似度
utt_sim_matrix = np.inner(embeds_a, embeds_b)

# 绘制相似性矩阵图
fig, ax = plt.subplots(figsize=(8, 8))
labels_a = list(true_speaker_wavs.keys())
labels_b = list(demo_speaker_wavs.keys())

def plot_similarity_matrix(matrix, labels_a, labels_b, ax, title=""):
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(labels_b)))
    ax.set_yticks(np.arange(len(labels_a)))
    ax.set_xticklabels(labels_b)
    ax.set_yticklabels(labels_a)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)

plot_similarity_matrix(utt_sim_matrix, labels_a, labels_b, ax,
                       "Cross-similarity between utterances")
plt.tight_layout()
plt.show()
