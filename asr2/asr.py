import os
import sys
import argparse
from subprocess import Popen
from tools2 import my_utils
from tools2.asr.config import asr_dict


def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang):

    # 路径清理，整理一下路径的格式，防止系统不同路径格式不同。没啥用
    asr_inp_dir = my_utils.clean_path(asr_inp_dir)
    asr_opt_dir = my_utils.clean_path(asr_opt_dir)


    # 构建命令，根据选择的不同的模型，使用不同的方法
    # 这里使用的是中文，自动运行 funasr_asr.py
    cmd = f'python tools2/asr/{asr_dict[asr_model]["path"]}'
    cmd += f' -i "{asr_inp_dir}"' #输入
    cmd += f' -o "{asr_opt_dir}"' # 输出
    cmd += f' -s {asr_model_size}' # 使用模型的大小
    cmd += f' -l {asr_lang}' # 语言选择
    cmd += " -p float32"

    print(f"Starting ASR task with command: {cmd}")
    p_asr = Popen(cmd, shell=True)
    p_asr.wait()
    print("ASR task completed.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Batch offline ASR task")
    parser.add_argument("--asr_inp_dir", type=str,default="/home/weizhenbian/mycode/asr2/combineaudio")
    parser.add_argument("--asr_opt_dir", type=str,default="/home/weizhenbian/mycode/asr2/text")
    parser.add_argument("--asr_model", type=str, default="Faster Whisper (多语种)")
    parser.add_argument("--asr_model_size", type=str, default="large")
    parser.add_argument("--asr_lang", type=str, default="zh")
    args = parser.parse_args()

    open_asr(args.asr_inp_dir, args.asr_opt_dir, args.asr_model, args.asr_model_size, args.asr_lang)
