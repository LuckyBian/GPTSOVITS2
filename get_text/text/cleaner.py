from text import chinese, japanese, cleaned_text_to_sequence, symbols, english

language_module_map = {"zh": chinese, "ja": japanese, "en": english}
special = [
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
]


def clean_text(text, language):
    # 查看语言类型
    if(language not in language_module_map):
        language="en"
        text=" "

    for special_s, special_l, target_symbol in special:
        # 如果为中文并且有特殊语言符号，则直接return
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
        

    # 得到对应的语言，不同的语言，标准化方式不一样
    language_module = language_module_map[language]

    # 得到标准化的文本
    norm_text = language_module.text_normalize(text)

    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None

    for ph in phones:
        assert ph in symbols
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):

    # 将特殊符号替换为逗号
    text = text.replace(special_s, ",")
  
    # 选择对应的语言
    language_module = language_module_map[language]

    # 对中文文本进行标准化处理，得到中文的List
    norm_text = language_module.text_normalize(text)

    # 得到中文的音素信息（拼音）
    phones = language_module.g2p(norm_text)

    new_ph = []

    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text, language):
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))
