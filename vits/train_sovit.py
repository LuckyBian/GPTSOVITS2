import argparse
import json
import os
import subprocess
import torch
import psutil
from multiprocessing import cpu_count

# 确定 GPU设备的数量，设置线程数量
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

######################################################################################################################


def run_sovits_training(batch_size, 
                        total_epoch, 
                        exp_name, 
                        text_low_lr_rate, 
                        if_save_latest, 
                        if_save_every_weights, 
                        save_every_epoch, 
                        gpu_numbers, 
                        pretrained_s2G, 
                        pretrained_s2D, 
                        is_half, 
                        exp_root, 
                        tmp):
    # Load and modify the training configuration
    with open("s2.json") as f:
        data = json.load(f)

    # Prepare directories
    s2_dir = f"{exp_root}/{exp_name}"
    os.makedirs(f"{s2_dir}/logs_s2", exist_ok=True)

    # Adjust settings based on precision
    if not is_half:
        data["train"]["fp16_run"] = False
        batch_size = max(1, batch_size // 2)

    # Update the configuration
    data["train"].update({
        "batch_size": batch_size,
        "epochs": total_epoch,
        "text_low_lr_rate": text_low_lr_rate,
        "pretrained_s2G": pretrained_s2G,
        "pretrained_s2D": pretrained_s2D,
        "if_save_latest": if_save_latest,
        "if_save_every_weights": if_save_every_weights,
        "save_every_epoch": save_every_epoch,
        "gpu_numbers": gpu_numbers
    })

    data["data"].update({
        "exp_dir": s2_dir,
    })

    data.update({
        "save_weight_dir": "/home/weizhenbian/mycode/model",
        "name": exp_name,
        "s2_ckpt_dir": s2_dir
    })

    # Save the updated configuration
    tmp_config_path = f"{tmp}/tmp_s2.json"
    with open(tmp_config_path, "w") as f:
        json.dump(data, f, indent=4)

    # Prepare the environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_numbers  # Assuming gpu_numbers is a string like '0,1,2'
    env["is_half"] = "1" if is_half else "0"

    # Execute the training script
    cmd = [
        "python", "s2_train.py",
        "--config", tmp_config_path
    ]

    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SoVITS Training Script")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--total_epoch", type=int, default=25)
    parser.add_argument("--text_low_lr_rate", type=float, default=0.4)
    parser.add_argument("--if_save_latest", type=bool, default=True)
    parser.add_argument("--if_save_every_weights", type=bool, default=True)
    parser.add_argument("--save_every_epoch", type=int, default=4)
    parser.add_argument("--gpu_numbers", default='0')


    parser.add_argument("--exp_name", default="guo")
    parser.add_argument("--pretrained_s2G", default="/home/weizhenbian/mycode/pretrain/GPT-SoVITS/s2G488k.pth")
    parser.add_argument("--pretrained_s2D", default="/home/weizhenbian/mycode/pretrain/GPT-SoVITS/s2D488k.pth")
    
    parser.add_argument("--exp_root", default="/home/weizhenbian/mycode/features")

    parser.add_argument("--sovits_weight_root", default="/home/weizhenbian/mycode/model")
    parser.add_argument("--tmp", default="/home/weizhenbian/mycode/vits/temp")
    parser.add_argument("--is_half", type=bool, default=True)
    parser.add_argument("--python_exec", default="python")

    args = parser.parse_args()
    run_sovits_training(args.batch_size, 
                        args.total_epoch, 
                        args.exp_name, 
                        args.text_low_lr_rate, 
                        args.if_save_latest, 
                        args.if_save_every_weights, 
                        args.save_every_epoch, 
                        args.gpu_numbers, 
                        args.pretrained_s2G, 
                        args.pretrained_s2D, 
                        args.is_half, 
                        args.exp_root, 
                        args.tmp)
