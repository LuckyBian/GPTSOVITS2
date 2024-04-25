import argparse
from operator import itemgetter
from googletrans import Translator, LANGUAGES

def parse_line(line):
    # 解析行数据
    path, _, _, text = line.strip().split('|')
    filename = path.split('/')[-1]
    start_time, duration, character_id = filename.split('_')[1:4]
    start_time = start_time[:-1]  # 去掉多余的点
    duration = duration[:-1]  # 去掉多余的点
    character_id = character_id.split('.')[0]  # 去掉.wav
    return int(character_id), float(start_time), float(duration), text

def translate_text(text, dest_lang):
    translator = Translator()
    if dest_lang not in LANGUAGES and dest_lang not in LANGUAGES.values():
        raise ValueError(f"Unsupported language code: {dest_lang}")
    return translator.translate(text, dest=dest_lang).text

def main(input_file, output_file, target_lang='en'):
    # 读取文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析数据
    entries = [parse_line(line) for line in lines]

    # 按开始时间排序
    sorted_entries = sorted(entries, key=itemgetter(1))

    # 翻译文本并写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for character_id, start_time, duration, text in sorted_entries:
            translated_text = translate_text(text, target_lang)
            f.write(f"{character_id}|{start_time}|{duration}|{translated_text}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process audio metadata and translate text.")
    parser.add_argument('--input_file', default="/home/weizhenbian/mycode/asr2/text/combineaudio.list")
    parser.add_argument('--output_file', default="/home/weizhenbian/mycode/inf2/taici/taici.txt")
    parser.add_argument('--lang', default='en', help="Target language for translation (en or ja)")
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.lang)
