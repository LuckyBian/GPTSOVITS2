import sys
import os
import subprocess
import argparse
import torch
import psutil
from multiprocessing import cpu_count

n_cpu=cpu_count()  
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if_gpu_ok = False

if torch.cuda.is_available() or ngpu != 0:
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if any(value in gpu_name.upper()for value in ["10","16","20","30","40","A2","A3","A4","P4","A50","500","A60","70","80","90","M4","T4","TITAN","L4","4060"]):
            if_gpu_ok = True
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(int(torch.cuda.get_device_properties(i).total_memory/ 1024/ 1024/ 1024+ 0.4))

if if_gpu_ok and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = ("%s\t%s" % ("0", "CPU"))
    gpu_infos.append("%s\t%s" % ("0", "CPU"))
    default_batch_size = psutil.virtual_memory().total/ 1024 / 1024 / 1024 / 2
gpus = "-".join([i[0] for i in gpu_infos])

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def extract_ssl_features(args):
    gpu_names = args.gpu_numbers.split("-")
    all_parts = len(gpu_names)

    for i_part, gpu_name in enumerate(gpu_names):
        env = os.environ.copy()
        env["inp_text"] = args.inp_text
        env["inp_wav_dir"] = args.inp_wav_dir
        env["exp_name"] = args.exp_name
        env["i_part"] = str(i_part)
        env["all_parts"] = str(all_parts)
        env["_CUDA_VISIBLE_DEVICES"] = gpu_name
        env["is_half"] = str(args.is_half)
        env["ssl_pretrained_dir"] = args.ssl_pretrained_dir
        env["exp_root"] = args.exp_root

        subprocess.call(['python', 'get_hubert.py'], env=env)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Extract SSL features from audio files.")
    parser.add_argument("--inp_text", default="/home/weizhenbian/mycode/data/asr/cut.list")
    parser.add_argument("--inp_wav_dir", default="/home/weizhenbian/mycode/data/cut")
    parser.add_argument("--exp_name", default="guo")
    parser.add_argument("--gpu_numbers", default=gpus)
    parser.add_argument("--ssl_pretrained_dir", default="/home/weizhenbian/mycode/pretrain/GPT-SoVITS/chinese-hubert-base")
    parser.add_argument("--exp_root", default="/home/weizhenbian/mycode/features/guo")
    parser.add_argument("--is_half", default="True")

    args = parser.parse_args()
    extract_ssl_features(args)
