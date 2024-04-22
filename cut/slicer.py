# 对音频进行分割
import os
import sys
from subprocess import Popen
import argparse

# 导入工具函数
from tools import my_utils

# 定义音频切割函数
def open_slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, n_parts):
    # 清理和验证输入路径
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)

    if not os.path.exists(inp):
        print("输入路径不存在")
        return
    if os.path.isfile(inp):
        n_parts = 1
    elif os.path.isdir(inp):
        pass
    else:
        print("输入路径存在但既不是文件也不是文件夹")
        return

    ps_slice = []
    for i_part in range(n_parts):
        cmd = f'"{sys.executable}" tools/slice_audio.py "{inp}" "{opt_root}" {threshold} {min_length} {min_interval} {hop_size} {max_sil_kept} {_max} {alpha} {i_part} {n_parts}'
        print(f"Executing command: {cmd}")
        p = Popen(cmd, shell=True)
        ps_slice.append(p)

    print("切割执行中")
    for p in ps_slice:
        p.wait()
    print("切割结束")

# 添加命令行参数处理
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动切割音频文件")
    parser.add_argument("--inp", type=str,default="/home/weizhenbian/mycode/data/denoise")
    parser.add_argument("--opt_root", type=str, default="/home/weizhenbian/mycode/data/cut",help="切分后的子音频的输出根目录")
    parser.add_argument("--threshold", type=str, default="-34",help="音量小于这个值视作静音的备选切割点")
    parser.add_argument("--min_length", type=str,default="4000", help="每段最小多长，如果第一段太短一直和后面段连起来直到超过这个值")
    parser.add_argument("--min_interval", type=str,default="300", help="最短切割间隔")
    parser.add_argument("--hop_size", type=str,default="10", help="怎么算音量曲线，越小精度越大计算量越高")
    parser.add_argument("--max_sil_kept", type=str,default="500", help="切完后静音最多留多长")
    parser.add_argument("--_max", type=float,default=0.9, help="归一化后最大值多少")
    parser.add_argument("--alpha", type=float,default=0.25, help="混多少比例归一化后音频进来")
    parser.add_argument("--n_parts", type=int, default=4, help="切割使用的进程数")

    args = parser.parse_args()

    # 调用函数执行音频切割
    open_slice(args.inp, args.opt_root, args.threshold, args.min_length, args.min_interval, args.hop_size, args.max_sil_kept, args._max, args.alpha, args.n_parts)
