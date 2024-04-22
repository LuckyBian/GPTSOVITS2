# 导包并接收参数
import os
inp_text = os.environ.get("inp_text")
inp_wav_dir = os.environ.get("inp_wav_dir")
exp_name = os.environ.get("exp_name")
i_part = os.environ.get("i_part")
all_parts = os.environ.get("all_parts")
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("_CUDA_VISIBLE_DEVICES")
opt_dir = os.environ.get("opt_dir")
bert_pretrained_dir = os.environ.get("bert_pretrained_dir")
is_half = eval(os.environ.get("is_half", "True"))
import sys, numpy as np, traceback, pdb
import os.path
from glob import glob
from tqdm import tqdm
from text.cleaner import clean_text
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
from time import time as ttime
import shutil


def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s%s.pth"%(ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

# 得到当前任务的保存路径
txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)

if os.path.exists(txt_path) == False:
    # 如果输出路径不存在，创建输出文件和文件夹
    bert_dir = "%s/3-bert" % (opt_dir)
    os.makedirs(opt_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)

    # 选择设备
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    # 导入预训练模型
    tokenizer = AutoTokenizer.from_pretrained(bert_pretrained_dir)

    bert_model = AutoModelForMaskedLM.from_pretrained(bert_pretrained_dir)

    # 根据训练精度将模型导入设备
    if is_half == True:
        bert_model = bert_model.half().to(device)
    else:
        bert_model = bert_model.to(device)

    def get_bert_feature(text, word2ph):

        with torch.no_grad():
            # 对文本进行编码
            inputs = tokenizer(text, return_tensors="pt")

            for i in inputs:
                # 将文本移动到设备上，加速
                inputs[i] = inputs[i].to(device)

            # 获取隐藏层
            res = bert_model(**inputs, output_hidden_states=True)
            
            # 只保留文本相关的内容
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
            
        # 确保长度一致
        assert len(word2ph) == len(text)

        phone_level_feature = []

        # 遍历每个字
        for i in range(len(word2ph)):
            # 调整维度，比如分为声母和韵母，维度为2，复制特征两层
            repeat_feature = res[i].repeat(word2ph[i], 1)

            phone_level_feature.append(repeat_feature)
        #将张量拼接在一起
        phone_level_feature = torch.cat(phone_level_feature, dim=0)

        return phone_level_feature.T

    def process(data, res):
        # 处理每一个line
        for name, text, lan in data:

            try:
                # 获得音频文件的名字
                name = os.path.basename(name)

                # 返回 拼音， mask, 文本
                phones, word2ph, norm_text = clean_text(
                    text.replace("%", "-").replace("￥", ","), lan
                )
                path_bert = "%s/%s.pt" % (bert_dir, name)

                if os.path.exists(path_bert) == False and lan == "zh":

                    # 拿到文字的音素级别的编码
                    bert_feature = get_bert_feature(norm_text, word2ph)

                    #对汉字进行编码，有多少个声母韵母就复制几层，和音频变量长度相互对应
                    assert bert_feature.shape[-1] == len(phones)

                    # torch.save(bert_feature, path_bert)
                    my_save(bert_feature, path_bert)

                phones = " ".join(phones)

                # 添加文本信息
                res.append([name, phones, word2ph, norm_text])
            except:
                print(name, text, traceback.format_exc())

    todo = []
    res = []

    # 对音频信息进行拆解
    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    # 格式化语言，支持中英日
    language_v1_to_language_v2 = {
        "ZH": "zh",
        "zh": "zh",
        "JP": "ja",
        "jp": "ja",
        "JA": "ja",
        "ja": "ja",
        "EN": "en",
        "en": "en",
        "En": "en",
        "YUE": "yue"
    }

    # 每个gpu设备处理几个line，多线程处理
    for line in lines[int(i_part) :: int(all_parts)]:
        try:
            # 得到信息
            wav_name, spk_name, language, text = line.split("|")
            
            # 放到todo里面
            todo.append(
                [wav_name, text, language_v1_to_language_v2.get(language, language)]
            )
        except:
            print(line, traceback.format_exc())

    process(todo, res)

    # 保存文本信息
    opt = []
    for name, phones, word2ph, norm_text in res:
        opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
