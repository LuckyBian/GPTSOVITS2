import argparse
import yaml
import os
import subprocess
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

def run_gpt_training(args):
    # 打开文件，导入参数
    with open(args.config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Adjust configurations based on input arguments
    config["train"]["batch_size"] = args.batch_size
    config["train"]["epochs"] = args.total_epoch
    config["train"]["if_dpo"] = args.if_dpo
    config["train"]["if_save_latest"] = args.if_save_latest
    config["train"]["if_save_every_weights"] = args.if_save_every_weights
    config["train"]["save_every_n_epoch"] = args.save_every_epoch
    config["train"]["gpu_numbers"] = args.gpu_numbers.replace("-", ",")
    config["pretrained_s1"] = args.pretrained_s1
    config["train"]["exp_name"] = args.exp_name

    # Specify directories based on the experiment name
    s1_dir = f"{args.exp_root}/{args.exp_name}"
    config["train"]["exp_dir"] = s1_dir
    config["train"]["half_weights_save_dir"] = "/home/weizhenbian/mycode/model"

    config["train_semantic_path"] = "/home/weizhenbian/mycode/features/guo/6-name2semantic.tsv"
    config["train_phoneme_path"] = "/home/weizhenbian/mycode/features/guo/2-name2text-0.txt"
    config["output_dir"] = f"{s1_dir}/logs_s1"

    # Write the modified configuration to a temporary file
    tmp_config_path = "/home/weizhenbian/mycode/gpt/tmp/tmp_s1.yaml"

    # 将新的参数写入 yaml
    with open(tmp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # 设置环境
    # Set CUDA_VISIBLE_DEVICES environment variable
    os.environ["_CUDA_VISIBLE_DEVICES"] = args.gpu_numbers.replace("-", ",")
    
    # 运行核心代码
    cmd = f'{args.python_exec} s1_train.py --config_file "{tmp_config_path}"'
    print(cmd)
    subprocess.run(cmd, shell=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run GPT training.")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--total_epoch", type=int, default=10)

    parser.add_argument("--if_dpo", action="store_true",default=False)
    parser.add_argument("--if_save_latest", default=True)
    parser.add_argument("--if_save_every_weights", default=True)

    parser.add_argument("--save_every_epoch", type=int, default=5)
    parser.add_argument("--gpu_numbers", type=str, default="0")

    parser.add_argument("--pretrained_s1", type=str, default="/home/weizhenbian/mycode/pretrain/GPT-SoVITS/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")
    parser.add_argument("--exp_name", type=str, default="guo")
    
    parser.add_argument("--exp_root", type=str, default="/home/weizhenbian/mycode/features")
    parser.add_argument("--config_path", type=str, default="/home/weizhenbian/mycode/gpt/s1longer.yaml")

    parser.add_argument("--tmp", type=str, default="./tmp")
    parser.add_argument("--python_exec", type=str, default="python")

    args = parser.parse_args()
    run_gpt_training(args)
