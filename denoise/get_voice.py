import os
import sys
import argparse
import traceback
import librosa
import ffmpeg
import soundfile as sf
import torch
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho
import numpy as np

# 设置权重根目录和设备
weight_uvr5_root = "/home/weizhenbian/TTS/my_code/0a/uvr5_weights"
device = "cuda" if torch.cuda.is_available() else "cpu"

def uvr(model_name, inp_root, save_root_vocal, save_root_ins, agg, format0, is_half):
    infos = []
    pre_fun = None
    print(f"Starting audio processing with model: {model_name}")
    
    try:
        print(f"Input root directory: {inp_root}")
        print(f"Save root for vocals: {save_root_vocal}, Save root for instruments: {save_root_ins}")
        print(f"Aggressiveness: {agg}, Output format: {format0}, Half precision: {is_half}")

        #模型选择
        is_hp3 = "HP3" in model_name

        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
            print("Initialized MDXNetDereverb")

        # 两种模型
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )
            print(f"Initialized {'AudioPre' if 'DeEcho' not in model_name else 'AudioPreDeEcho'}")

        paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root) if os.path.isfile(os.path.join(inp_root, name))]
        print(f"Found {len(paths)} audio files to process.")
        
        for inp_path in paths:
            #看下有多少的要处理的音频
            print(f"Processing file: {inp_path}")

            need_reformat = True
            try:
                # 拿到音频信息，需要是双声道，采样率为44100才能进行分离
                info = ffmpeg.probe(inp_path, cmd="ffprobe")

                print(f"Audio channels: {info['streams'][0]['channels']}, Sample rate: {info['streams'][0]['sample_rate']}")
                
                if info["streams"][0]["channels"] == 2 and info["streams"][0]["sample_rate"] == "44100":

                    need_reformat = False
                    print("Format is correct, proceeding without reformatting.")

                    #没有处理，直接进function
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0, is_hp3)
                
            except Exception as e:
                print(f"Error during format checking: {e}")
                traceback.print_exc()
                need_reformat = True

            if need_reformat:
                #创建临时文件
                tmp_path = f"/tmp/{os.path.basename(inp_path)}.reformatted.wav"
                #对音频格式进行修改
                os.system(f"ffmpeg -i {inp_path} -vn -acodec pcm_s16le -ac 2 -ar 44100 {tmp_path} -y")
                print(f"Reformatted audio saved to: {tmp_path}")
                #更改输入路径
                inp_path = tmp_path

                data, samplerate = sf.read(inp_path)
                # 检查数据中是否有无穷大或NaN值，并进行清理
                if np.any(np.isinf(data)) or np.any(np.isnan(data)):
                    print("Audio data contains infinite or NaN values. Cleaning up...")
                    data[np.isinf(data)] = 0  # 将无穷大值设置为0
                    data[np.isnan(data)] = 0  # 将NaN值设置为0
                    # 保存修正后的音频数据到文件
                    sf.write(inp_path, data, samplerate)
                try:
                    pre_fun._path_audio_(inp_path, save_root_ins, save_root_vocal, format0, is_hp3)
                    print(f"Processed and saved: {inp_path}")
                except Exception as e:
                    print(f"Error during audio processing: {e}")
                    traceback.print_exc()

            infos.append(f"{os.path.basename(inp_path)}->Success")

    except Exception as e:
        print(f"General error in processing: {e}")
        infos.append(traceback.format_exc())
    
    # 对刚刚定义的模型进行清理
    finally:
        if pre_fun is not None:
            try:
                del pre_fun
                print("Cleaned up model.")

            except Exception as e:
                print(f"Error during cleanup: {e}")

                traceback.print_exc()

        print("Clean empty cache")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return "\n".join(infos)


def main():
    parser = argparse.ArgumentParser(description="Audio processing with UVR model")
    parser.add_argument("--model_name", type=str,default="DeEchoAggressive")
    parser.add_argument("--inp_root", type=str,default="/home/weizhenbian/TTS/my_code/0a/output")
    parser.add_argument("--save_root_vocal", type=str, default="/home/weizhenbian/TTS/my_code/0a/output")
    parser.add_argument("--save_root_ins", type=str, default="/home/weizhenbian/TTS/my_code/0a/useless")
    parser.add_argument("--agg", type=int, default=10)
    parser.add_argument("--format0", type=str, default="wav")
    parser.add_argument("--is_half", type=eval, choices=[True, False], default=False)

    args = parser.parse_args()

    infos = uvr(args.model_name, args.inp_root, args.save_root_vocal, args.save_root_ins, args.agg, args.format0, args.is_half)

if __name__ == "__main__":
    main()