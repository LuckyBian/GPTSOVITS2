import argparse
import os
import subprocess
from pathlib import Path
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

def run_semantic_extraction(inp_text, exp_name, gpu_numbers, pretrained_s2G_path, s2config_path, is_half, exp_root):
    opt_dir = f"{exp_root}/{exp_name}"
    gpu_names = gpu_numbers.split("-")
    all_parts = len(gpu_names)
    
    for i_part, gpu_name in enumerate(gpu_names):
        env = os.environ.copy()
        env.update({
            "inp_text": inp_text,
            "exp_name": exp_name,
            "opt_dir": opt_dir,
            "pretrained_s2G": pretrained_s2G_path,
            "s2config_path": s2config_path,
            "is_half": str(is_half),
            "i_part": str(i_part),
            "all_parts": str(all_parts),
            "_CUDA_VISIBLE_DEVICES": gpu_name,
        })

        cmd = ["python", "get_semantic.py"]
        subprocess.run(cmd, env=env)

    semantic_files = [Path(opt_dir) / f"6-name2semantic-{i}.tsv" for i in range(all_parts)]
    combined_semantic_path = Path(opt_dir) / "6-name2semantic.tsv"
    
    with combined_semantic_path.open("w", encoding="utf8") as outfile:
        outfile.write("item_name\tsemantic_audio\n")
        for semantic_file in semantic_files:
            with semantic_file.open("r", encoding="utf8") as infile:
                outfile.write(infile.read())
                outfile.write("\n")
            semantic_file.unlink()  # Optionally delete the part files

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run semantic token extraction.")
    parser.add_argument("--inp_text", default="/home/weizhenbian/mycode/data/asr/cut.list")
    parser.add_argument("--exp_name", default="guo")

    parser.add_argument("--gpu_numbers", default=gpus)

    parser.add_argument("--pretrained_s2G", default="/home/weizhenbian/mycode/pretrain/GPT-SoVITS/s2G488k.pth")
    
    parser.add_argument("--s2config_path", default="/home/weizhenbian/mycode/get_emo/s2.json")
    parser.add_argument("--is_half", type=bool, default=True)
    parser.add_argument("--exp_root", default="/home/weizhenbian/mycode/features")
    
    args = parser.parse_args()
    run_semantic_extraction(args.inp_text, args.exp_name, args.gpu_numbers, args.pretrained_s2G, args.s2config_path, args.is_half, args.exp_root)
