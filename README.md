# GPTSOVITS2

## Create the Env:
```
conda create --name gptsovits python=3.10
conda activate gptsovits
```

## Install Package

```
conda install -y -c pytorch -c nvidia cudatoolkit
conda install -y -c conda-forge gcc gxx ffmpeg cmake -c pytorch -c nvidia
pip install -r requirements.txt
```

## Train your own Model: Data Preprocessing

### Step 1: Denoise

Put about 5 minutes of training .wav audio into data/ori. The purpose of this step is noise reduction. Note that you need to prepare a Linux system, other systems are not supported for the time being. If your training audio is of high quality, you can skip this step and go directly to step 2.

```
pip install resemble-enhance --upgrade --pre
resemble-enhance data/ori data/denoise
```
### Step 2: Cut

The purpose of this step is to slice the audio to facilitate subsequent training. Note that the input and output paths need to be adjusted.

```
cd cut
python slicer.py
cd ..
```

### Step 3: ASR

The purpose of this step is to convert the audio into text. The default is Chinese. If you need to adjust it to English, you need to change the parameters manually. Note that the input and output paths need to be adjusted. You can choose 'zh', 'en', 'ja', 'auto'

Note that the first run will take some time to download the pre-trained model to recognize text.

```
cd asr
python asr.py
cd ..
```