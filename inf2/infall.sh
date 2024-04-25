#!/bin/bash

# 定义 Python 脚本的路径
SCRIPT_PATH="/home/weizhenbian/mycode/inf2/inf.py"

# 定义角色 ID 到名字的映射
declare -A id_to_name=(
    [1]="liuxing"
    [2]="xiaoyu"
    [3]="xiaoxue"
    [4]="xiadonghai"
    [5]="liumei"
)

# 定义角色 ID 到 prompt_text 的映射
declare -A id_to_prompt=(
    [1]="你放心，我是不会欺负你的，这一点我可以向你保证。"
    [2]="自由嘛，反正晚上不洗脚绝对绝对不能上床的"
    [3]="哦对，还有那个叫刘星的孩子，他欺负过你"
    [4]="诶，这不是你今天第一天来嘛？这可是妈妈为了迎接你专门做的。"
    [5]="啊，小雪有很多习惯，以后咱们大家都慢慢习惯。"
)

# 初始化行数计数器
line_count=0

# 读取 txt 文件每一行
while IFS= read -r line; do
    # 更新行数
    ((line_count++))

    # 解析行以获取需要的字段
    IFS='|' read -ra ADDR <<< "$line"
    idx="${ADDR[0]}"
    start_time="${ADDR[1]}"
    duration="${ADDR[2]}"
    text="${ADDR[3]}"

    # 获取角色名和对应的 prompt_text
    character_name="${id_to_name[$idx]}"
    prompt_text="${id_to_prompt[$idx]}"

    # 构造参考音频的路径
    ref_wav_path="/home/weizhenbian/mycode/inf2/ref/${character_name}.wav"

    # 构造模型路径
    gpt_path="/home/weizhenbian/mycode/model/gpt/${character_name}.ckpt"
    sovits_path="/home/weizhenbian/mycode/model/vits/${character_name}.pth"

    # 构造输出路径
    output_path="/home/weizhenbian/out/${line_count}_${start_time}_${duration}"
    #output_file="${output_path}/${character_name}_output_${idx}.wav"

    # 确保输出目录存在
    mkdir -p "$output_path"

    # 打印将要执行的 Python 命令
    echo "Running Python script with the following command:"
    echo "python $SCRIPT_PATH --prompt_text \"$prompt_text\" --text \"$text\" --output_path \"$output_path\" --ref_wav_path \"$ref_wav_path\" --gpt_path \"$gpt_path\" --sovits_path \"$sovits_path\" --text_language \"英文\" --prompt_language \"中文\""

    # 根据解析的数据设置参数并运行Python脚本
    python $SCRIPT_PATH \
        --prompt_text "$prompt_text" \
        --text "$text" \
        --output_path "$output_path" \
        --ref_wav_path "$ref_wav_path" \
        --gpt_path "$gpt_path" \
        --sovits_path "$sovits_path" \
        --text_language "英文" \
        --prompt_language "中文"
    
    # 检查文件是否生成并且Python脚本运行成功
    if [ $? -eq 0 ] && [ -f "$output_file" ]; then
        # 移动和重命名输出文件
        mv "$output_file" "/home/weizhenbian/out/${line_count}_${start_time}_${duration}.wav"
        # 删除原始输出目录
        rm -r "$output_path"
    else
        echo "Error occurred: Python script failed or output file not generated."
    fi

done < "/home/weizhenbian/mycode/inf2/taici/taici.txt"
