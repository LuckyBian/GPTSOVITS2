import jieba_fast.posseg as psg
from text.tone_sandhi import ToneSandhi
import re
from text.symbols import punctuation
from pypinyin import lazy_pinyin, Style

text = "我爱北京天安门."

pattern = r"(?<=[{0}])\s*".format("".join(punctuation))
sentences = [i for i in re.split(pattern, text) if i.strip() != ""]

print(sentences)

for seg in sentences:
    print(seg)

seg_cut = psg.lcut(text)
print(seg_cut)

tone_modifier = ToneSandhi()
seg_cut = tone_modifier.pre_merge_for_modify(seg_cut)
print(seg_cut)

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


for word, pos in seg_cut:       
    sub_initials, sub_finals = _get_initials_finals(word)
    print(sub_finals)
    sub_finals = tone_modifier.modified_tone(word, pos, sub_finals)
    

