import os

inp_text = os.environ.get("inp_text")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("_CUDA_VISIBLE_DEVICES")
opt_dir = os.environ.get("opt_dir")
pretrained_s2G = os.environ.get("pretrained_s2G")
s2config_path = os.environ.get("s2config_path")
is_half = eval(os.environ.get("is_half", "True"))
import math, traceback
import multiprocessing
import sys, pdb

now_dir = os.getcwd()
sys.path.append(now_dir)
from random import shuffle
import torch.multiprocessing as mp
from glob import glob
from tqdm import tqdm
import logging, librosa, utils, torch
from module.models import SynthesizerTrn

logging.getLogger("numba").setLevel(logging.WARNING)

# 确定路径，输入是上一步的音频特征编码
hubert_dir = "/home/weizhenbian/mycode/features/guo/4-cnhubert"

# 输出路径
semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)

# 查看目标文件是否存在，和之前一样，一个gpu线程负责一部分文件的处理
if not os.path.exists(semantic_path):
    os.makedirs(opt_dir, exist_ok=True)

    # 确定device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 拿到模型的参数
    hps = utils.get_hparams_from_file(s2config_path)

    # 创建模型
    # 参数太多，通过json文件导入
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1, # 2048 // 2 + 1 = 1025
        hps.train.segment_size // hps.data.hop_length, # 20480 / 640 = 32
        n_speakers=hps.data.n_speakers, # 300
        **hps.model
    ).to(device)

    # 确定精度
    if is_half:
        vq_model = vq_model.half()

    # 确定为评估不是训练，不更新参数
    vq_model.eval()

    vq_model.load_state_dict(torch.load(pretrained_s2G, map_location="cpu")["weight"], strict=False)

    def process_wav_file(wav_name, lines):

        # 拿到hubert的路径，并把文件的名字也加上
        hubert_path = os.path.join(hubert_dir, f"{wav_name}.pt")

        # 如果没找到则直接返回
        if not os.path.exists(hubert_path):
            return
        # 将音频编码信息导入device
        ssl_content = torch.load(hubert_path, map_location="cpu").to(device)

        # 查看精度，不重要
        if is_half:
            ssl_content = ssl_content.half()

        # 提取隐藏层
        codes = vq_model.extract_latent(ssl_content)

        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append(f"{wav_name}\t{semantic}")

    processed_lines = []

    # 读取0c中的文件，将文本读取进来
    with open(inp_text, "r", encoding="utf8") as f:
        lines = [line.strip() for line in f]
    
    # gpu多线程进行
    for line in lines[int(i_part)::int(all_parts)]:

        try:
            # 只拿音频文件的名字
            wav_name, _, _, _ = line.split("|")

            wav_name = os.path.basename(wav_name)

            # 输入为音频的名字和一个空的list
            process_wav_file(wav_name, processed_lines)
            
        except Exception as e:
            print(f"Error processing line: {line}\n{traceback.format_exc()}")
    
    # 将编码的语意文本进行输出
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(processed_lines))