import argparse
import os
from pathlib import Path
import subprocess

def extract_text(inp_text, inp_wav_dir, exp_name, gpu_numbers, bert_pretrained_dir, exp_root):

    # 创建路径
    opt_dir = os.path.join(exp_root, exp_name)
    os.makedirs(opt_dir, exist_ok=True)
    gpu_names = gpu_numbers.split("-")
    all_parts = len(gpu_names)

# 一个gpu处理一个任务，最后再进行汇总
    for i_part, gpu_name in enumerate(gpu_names):
        env = os.environ.copy()
        env.update({
            "inp_text": inp_text, # asr
            "inp_wav_dir": inp_wav_dir, # 音频的wav
            "exp_name": exp_name, #模型名
            "opt_dir": opt_dir,
            "bert_pretrained_dir": bert_pretrained_dir, # 模型路径
            "i_part": str(i_part),
            "all_parts": str(all_parts),
            "_CUDA_VISIBLE_DEVICES": gpu_name,
            "is_half": str(True)
        })
        cmd = f'python get_text.py'
        print(f"Executing: {cmd}")
        process = subprocess.Popen(cmd, shell=True, env=env)
        process.wait()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract text from audio files.")
    parser.add_argument("--inp_text", default="/home/weizhenbian/mycode/data/asr/cut.list")
    parser.add_argument("--inp_wav_dir", default="/home/weizhenbian/mycode/data/cut")
    parser.add_argument("--exp_name", default="guo")
    parser.add_argument("--gpu_numbers", default="0")
    parser.add_argument("--bert_pretrained_dir", default="/home/weizhenbian/mycode/get_text/GPT-SoVITS/chinese-roberta-wwm-ext-large")
    parser.add_argument("--exp_root", default="/home/weizhenbian/mycode/features")
    
    args = parser.parse_args()
    
    extract_text(args.inp_text, args.inp_wav_dir, args.exp_name, args.gpu_numbers, args.bert_pretrained_dir, args.exp_root)
