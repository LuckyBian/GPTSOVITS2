from pydub import AudioSegment
import os

def merge_audio_segments(input_file, output_dir='output'):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    current_audio = None
    current_start = 0.0
    current_duration = 0.0
    current_role = None
    merged_data = []

    with open(input_file, 'r') as f:
        for line in f:
            path, start, duration, role = line.strip().split('|')
            start = float(start)
            duration = float(duration)
            role = int(role)

            # 加载音频片段
            audio = AudioSegment.from_file(path)

            if current_role == role:
                # 追加当前音频片段到已有的音频片段
                current_audio += audio
                current_duration += duration
            else:
                if current_audio:
                    # 输出合并后的音频文件
                    output_path = os.path.join(output_dir, f"merged_{current_start:.2f}_{current_duration:.2f}_{current_role}.wav")
                    current_audio.export(output_path, format="wav")
                    merged_data.append(f"{output_path}|{current_start:.2f}|{current_duration:.2f}|{current_role}")

                # 重置当前音频片段
                current_audio = audio
                current_start = start
                current_duration = duration
                current_role = role

    # 输出最后一个合并的音频
    if current_audio:
        output_path = os.path.join(output_dir, f"merged_{current_start:.2f}_{current_duration:.2f}_{current_role}.wav")
        current_audio.export(output_path, format="wav")
        merged_data.append(f"{output_path}|{current_start:.2f}|{current_duration:.2f}|{current_role}")

    # 更新 TXT 文件
    with open(os.path.join(output_dir, 'merged_output.txt'), 'w') as f:
        for item in merged_data:
            f.write(item + '\n')

# 使用示例
input_file = '/home/weizhenbian/mycode/asr2/cut/mix_timestamps_roles.txt'  # 替换为实际输入文件的路径
output_dir = '/home/weizhenbian/mycode/asr2/combineaudio'  # 指定自定义输出目录
merge_audio_segments(input_file, output_dir)
