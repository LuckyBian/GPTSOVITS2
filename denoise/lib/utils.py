import json

import numpy as np
import torch
from tqdm import tqdm


def load_data(file_name: str = "./lib/name_params.json") -> dict:
    with open(file_name, "r") as f:
        data = json.load(f)

    return data


def make_padding(width, cropsize, offset):

    left = offset

    roi_size = cropsize - left * 2

    if roi_size == 0:
        roi_size = cropsize

    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


def inference(X_spec, device, model, aggressiveness, data):

    # 分别为输入频谱，设备，模型，干扰参数，其他参数

    def _execute(
        X_mag_pad, 
        roi_size, 
        n_window, 
        device, 
        model, 
        aggressiveness, 
        is_half=True
    ):
        model.eval()
        with torch.no_grad():
            preds = []

            iterations = [n_window]

            total_iterations = sum(iterations)
            for i in tqdm(range(n_window)):
                start = i * roi_size
                X_mag_window = X_mag_pad[
                    None, :, :, start : start + data["window_size"]
                ]
                X_mag_window = torch.from_numpy(X_mag_window)
                if is_half:
                    X_mag_window = X_mag_window.half()
                X_mag_window = X_mag_window.to(device)

                pred = model.predict(X_mag_window, aggressiveness)

                pred = pred.detach().cpu().numpy()
                preds.append(pred[0])

            pred = np.concatenate(preds, axis=2)
        return pred

    def preprocess(X_spec):
        #得到频谱的幅度和相位
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    # 得到频谱的幅度和相位
    X_mag, X_phase = preprocess(X_spec)

    # 对频谱进行标准化，通常为3个维度：频率数，频道数，时间帧数
    coef = X_mag.max()
    #让每个值处以频谱幅度的最大值
    X_mag_pre = X_mag / coef

    #获得时间帧数
    n_frame = X_mag_pre.shape[2]

    # data_window = 512，保证信息采集完整
    pad_l, pad_r, roi_size = make_padding(n_frame, data["window_size"], model.offset)

    # 看看有多少个窗
    n_window = int(np.ceil(n_frame / roi_size))

    #进行填充，让窗的大小与频谱相互适应
    X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

    if list(model.state_dict().values())[0].dtype == torch.float16:
        is_half = True
    else:
        is_half = False

    # 调用模型进行预测
    pred = _execute(
        X_mag_pad, 
        roi_size, 
        n_window, 
        device, 
        model, 
        aggressiveness, 
        is_half
    )

    pred = pred[:, :, :n_frame]

    if data["tta"]:
        pad_l += roi_size // 2
        pad_r += roi_size // 2
        n_window += 1

        X_mag_pad = np.pad(X_mag_pre, ((0, 0), (0, 0), (pad_l, pad_r)), mode="constant")

        pred_tta = _execute(
            X_mag_pad, roi_size, n_window, device, model, aggressiveness, is_half
        )
        pred_tta = pred_tta[:, :, roi_size // 2 :]
        pred_tta = pred_tta[:, :, :n_frame]

        return (pred + pred_tta) * 0.5 * coef, X_mag, np.exp(1.0j * X_phase)
    
    else:

        # 返回的分别为预测图谱， 图谱的幅度， 图谱的复数图谱
        # 后两者可以重构为原图谱
        return pred * coef,  X_mag,  np.exp(1.0j * X_phase)


def _get_name_params(model_path, model_hash):
    data = load_data()
    flag = False
    ModelName = model_path
    for type in list(data):
        for model in list(data[type][0]):
            for i in range(len(data[type][0][model])):
                if str(data[type][0][model][i]["hash_name"]) == model_hash:
                    flag = True
                elif str(data[type][0][model][i]["hash_name"]) in ModelName:
                    flag = True

                if flag:
                    model_params_auto = data[type][0][model][i]["model_params"]
                    param_name_auto = data[type][0][model][i]["param_name"]
                    if type == "equivalent":
                        return param_name_auto, model_params_auto
                    else:
                        flag = False
    return param_name_auto, model_params_auto
