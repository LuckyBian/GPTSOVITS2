import os,sys

inp_text = os.getenv("inp_text")
inp_wav_dir = os.getenv("inp_wav_dir")
exp_name = os.getenv("exp_name")
i_part = os.getenv("i_part")
all_parts = os.getenv("all_parts")
cuda_visible_devices = os.getenv("_CUDA_VISIBLE_DEVICES")
is_half = os.getenv("is_half") == "True"
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

from feature_extractor import cnhubert
opt_dir= os.environ.get("exp_root")
cnhubert.cnhubert_base_path= os.environ.get("ssl_pretrained_dir")
is_half=eval(os.environ.get("is_half","False"))

import pdb,traceback,numpy as np,logging
from scipy.io import wavfile
import librosa,torch
now_dir = os.getcwd()
sys.path.append(now_dir)
from my_utils import load_audio

from time import time as ttime
import shutil

def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s%s.pth"%(ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

# 设置保存路径
hubert_dir="%s/4-cnhubert"%(opt_dir)
wav32dir="%s/5-wav32k"%(opt_dir)
os.makedirs(opt_dir,exist_ok=True)
os.makedirs(hubert_dir,exist_ok=True)
os.makedirs(wav32dir,exist_ok=True)

# 设置参数
maxx=0.95
alpha=0.5

# 选择设备
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

# 导入模型
model=cnhubert.get_model()

#根据精度选择处理方式
if(is_half==True):
    model=model.half().to(device)
else:
    model = model.to(device)

nan_fails=[]
def name2go(wav_name,wav_path):

    # 确定输出路径，文件名为音频名。pt
    hubert_path="%s/%s.pt"%(hubert_dir,wav_name)

    # 如果不存在，直接return
    if(os.path.exists(hubert_path)):return

    # 使用ffepeg导入模音频
    tmp_audio = load_audio(wav_path, 32000)

    # 拿到音频的最大幅度
    tmp_max = np.abs(tmp_audio).max()

    # 如果太大，则直接返回
    if tmp_max > 2.2:
        print("%s-filtered,%s" % (wav_name, tmp_max))
        return
    
    # 调整音频幅度
    # 最终的结果是一个经过特定比例混合和缩放的音频信号
    # 32768是16位PCM音频的最大幅度值（由于PCM值范围是[-32768, 32767]）
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio

    # 1145.14代表一个特定的幅度调整目标
    # 将归一化音频与原始音频按照一定比例混合
    # 对音频数据进行预处理，以适应后续的音频分析或处理任务
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio

    tmp_audio = librosa.resample(
        tmp_audio32b, orig_sr=32000, target_sr=16000
    )

    # 转换成张量
    tensor_wav16 = torch.from_numpy(tmp_audio)

    # 根据精度放入gpu设备
    if (is_half == True):
        tensor_wav16=tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)

    # torch.Size([1, 768, 215])
    # 拿到音频特征的隐藏层
    ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()

    # 看一下是否存在
    if np.isnan(ssl.detach().numpy()).sum()!= 0:
        nan_fails.append(wav_name)
        print("nan filtered:%s"%wav_name)
        return
    
    # 保存重新处理后的音频
    wavfile.write(
        "%s/%s"%(wav32dir,wav_name),
        32000,
        tmp_audio32.astype("int16"),
    )

    # 保存音频特征(HuBert的隐藏层)
    my_save(ssl,hubert_path )

# 还是打开0c的输出，拿到音频的文本。
with open(inp_text,"r",encoding="utf8")as f:
    lines=f.read().strip("\n").split("\n")

# 和之前一样，多线程运行
for line in lines[int(i_part)::int(all_parts)]:

    try:
        # 拆解0c的输出文本
        wav_name, spk_name, language, text = line.split("|")

        if (inp_wav_dir != "" and inp_wav_dir != None):
            # 拿到音频名和音频路径
            wav_name = os.path.basename(wav_name)
            wav_path = "%s/%s"%(inp_wav_dir, wav_name)

        else:
            wav_path=wav_name
            wav_name = os.path.basename(wav_name)
            
        name2go(wav_name,wav_path)

    except:
        print(line,traceback.format_exc())

if(len(nan_fails)>0 and is_half==True):
    is_half=False
    model=model.float()
    for wav_name in nan_fails:
        try:
            name2go(wav_name)
        except:
            print(wav_name,traceback.format_exc())
