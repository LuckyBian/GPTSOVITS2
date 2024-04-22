import os
import pdb
import re

import cn2an
from pypinyin import lazy_pinyin, Style

from text.symbols import punctuation
from text.tone_sandhi import ToneSandhi
from text.zh_normalization.text_normlization import TextNormalizer

normalizer = lambda x: cn2an.transform(x, "an2cn")

current_file_path = os.path.dirname(__file__)
pinyin_to_symbol_map = {
    line.split("\t")[0]: line.strip().split("\t")[1]
    for line in open(os.path.join(current_file_path, "opencpop-strict.txt")).readlines()
}

import jieba_fast.posseg as psg


rep_map = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "/": ",",
    "—": "-",
    "~": "…",
    "～":"…",
}

tone_modifier = ToneSandhi()


def replace_punctuation(text):

    text = text.replace("嗯", "恩").replace("呣", "母")

    # 对标点进行统一处理，替换成相同的
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    replaced_text = pattern.sub(lambda x: rep_map[x.group()], text)

    # 删除所有不是中文的字符
    replaced_text = re.sub(
        r"[^\u4e00-\u9fa5" + "".join(punctuation) + r"]+", "", replaced_text
    )

    return replaced_text


def g2p(text):

    # 对文本进行再次处理，处理标点
    pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
    sentences = [i for i in re.split(pattern, text) if i.strip() != ""]

    # 得到音素信息（拼音）和标记
    phones, word2ph = _g2p(sentences)
    return phones, word2ph


def _get_initials_finals(word):

    initials = []
    finals = []

    orig_initials = lazy_pinyin(word, neutral_tone_with_five=True, style=Style.INITIALS)
    orig_finals = lazy_pinyin(
        word, neutral_tone_with_five=True, style=Style.FINALS_TONE3
    )
    for c, v in zip(orig_initials, orig_finals):
        initials.append(c)
        finals.append(v)
    return initials, finals


def _g2p(segments):

    phones_list = []
    word2ph = []

    # 直接处理一句话，因为在list里面，只循环一次
    for seg in segments:

        pinyins = []

        # 将里面所有的英语都去除
        seg = re.sub("[a-zA-Z]+", "", seg)

        # 对词性进行标注
        seg_cut = psg.lcut(seg)

        initials = []
        finals = []

        # 调整标记的格式
        seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)


        for word, pos in seg_cut:
            # 防止有英语
            if pos == "eng":
                continue

            # 标记对应词组的开头拼音和结尾拼音，处理连读变音问题
            sub_initials, sub_finals = _get_initials_finals(word)
            
            # 处理连读变音
            sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
            finals.append(sub_finals)
            initials.append(sub_initials)
        # 将列表展开
        initials = sum(initials, [])
        finals = sum(finals, [])
        #
        for c, v in zip(initials, finals):
            raw_pinyin = c + v
            
            # 处理标点
            if c == v:
                assert c in punctuation
                phone = [c]
                word2ph.append(1)
            else:
                # 将拼音与声调分离
                v_without_tone = v[:-1]
                tone = v[-1]

                pinyin = c + v_without_tone

                assert tone in "12345"

                if c:
                    # 多音节
                    v_rep_map = {
                        "uei": "ui",
                        "iou": "iu",
                        "uen": "un",
                    }
                    if v_without_tone in v_rep_map.keys():
                        pinyin = c + v_rep_map[v_without_tone]
                else:
                    # 单音节
                    pinyin_rep_map = {
                        "ing": "ying",
                        "i": "yi",
                        "in": "yin",
                        "u": "wu",
                    }
                    if pinyin in pinyin_rep_map.keys():
                        pinyin = pinyin_rep_map[pinyin]
                    else:
                        single_rep_map = {
                            "v": "yu",
                            "e": "e",
                            "i": "y",
                            "u": "w",
                        }
                        if pinyin[0] in single_rep_map.keys():
                            pinyin = single_rep_map[pinyin[0]] + pinyin[1:]

                assert pinyin in pinyin_to_symbol_map.keys(), (pinyin, seg, raw_pinyin)
                new_c, new_v = pinyin_to_symbol_map[pinyin].split(" ")
                new_v = new_v + tone
                phone = [new_c, new_v]
                word2ph.append(len(phone))

            phones_list += phone
    return phones_list, word2ph


def text_normalize(text):
    
    tx = TextNormalizer()

    sentences = tx.normalize(text)

    dest_text = ""

    for sentence in sentences:
        dest_text += replace_punctuation(sentence)

    # 返回标准化的中文文本
    return dest_text


if __name__ == "__main__":
    text = "啊——但是《原神》是由,米哈\游自主，研发的一款全.新开放世界.冒险游戏"
    text = "呣呣呣～就是…大人的鼹鼠党吧？"
    text = "你好"
    text = text_normalize(text)
    print(g2p(text))


# # 示例用法
# text = "这是一个示例文本：,你好！这是一个测试..."
# print(g2p_paddle(text))  # 输出: 这是一个示例文本你好这是一个测试
