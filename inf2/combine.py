from pydub import AudioSegment
import os

# 指定文件夹路径
folder_path = '/home/weizhenbian/mycode/inf2/out'

# 初始化一个空的音频段用于最终合并
final_track = AudioSegment.silent(duration=0)

# 遍历指定文件夹中的所有文件
for filename in sorted(os.listdir(folder_path)):
    if filename.endswith('.wav'):
        # 解析文件名获取索引、开始时间和目标时长
        index, start_s, duration_s = filename.rstrip('.wav').split('_')
        start_s = float(start_s) * 1000  # 转换为毫秒
        
        # 加载音频文件
        sound = AudioSegment.from_wav(os.path.join(folder_path, filename))
        
        # 确保最终音轨的长度足以进行叠加
        end_time = start_s + sound.duration_seconds * 1000
        if len(final_track) < end_time:
            final_track += AudioSegment.silent(duration=(end_time - len(final_track)))
        
        # 将当前处理的音频叠加到正确的位置
        final_track = final_track.overlay(sound, position=start_s)

# 导出合并后的音频文件
final_track.export("/home/weizhenbian/mycode/inf2/final/final_merged_output.wav", format="wav")

print("音频文件处理完成，并已导出。")
