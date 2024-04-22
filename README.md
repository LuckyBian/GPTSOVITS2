# GPTSOVITS2

## SECTION 1 : PROJECT TITLE

### A Project Name

### Demo

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT

Embarking on a PhD is a rigorous academic endeavor. Students are required to deeply engage in specialized fields, collaborating and exchanging insights with global researchers. To excel academically, they continuously delve into, peruse, and dissect numerous scholarly articles. Along this journey, challenges arise, including determining which papers to study, pinpointing their research trajectory, and keeping abreast of groundbreaking studies. While Google Scholar stands as a predominant academic search tool, offering vast scholarly resources, it is not without its limitations.


## SECTION 3 : CREDITS / PROJECT CONTRIBUTION


| Official Full Name  | Student ID (MTech Applicable)  | Work Items  | Email (Optional) |
| :------------ |:---------------:| :-----| :-----|
| Bian Weizhen  | A0285814W | ALL| E1221626@nus.edu.sg |
| Liu Siyan     | A0285814W | ALL| E1221626@nus.edu.sg |


## SECTION 4 : BUSINESS VIDEO 

## SECTION 5 : USER GUIDE


### Create the Env:
```
conda create --name gptsovits python=3.10
conda activate gptsovits
```

### Install Package

```
conda install -y -c pytorch -c nvidia cudatoolkit
conda install -y -c conda-forge gcc gxx ffmpeg cmake -c pytorch -c nvidia
pip install -r requirements.txt
```

### Train your own Model: Data Preprocessing

#### Step 1: Denoise

Put about 5 minutes of training .wav audio into data/ori. The purpose of this step is noise reduction. Note that you need to prepare a Linux system, other systems are not supported for the time being. If your training audio is of high quality, you can skip this step and go directly to step 2.

```
pip install resemble-enhance --upgrade --pre
resemble-enhance data/ori data/denoise
```
#### Step 2: Cut

The purpose of this step is to slice the audio to facilitate subsequent training. Note that the input and output paths need to be adjusted.

```
cd cut
python slicer.py
```

#### Step 3: ASR

The purpose of this step is to convert the audio into text. The default is Chinese. If you need to adjust it to English, you need to change the parameters manually. Note that the input and output paths need to be adjusted. You can choose 'zh', 'en', 'ja', 'auto'

Note that the first run will take some time to download the pre-trained model to recognize text.

```
cd asr
python asr.py
```

### Train your own Model: Get the Features

#### Step 1: Extract pronunciation and text encoding

The purpose of this step is to extract the text encoding and phonetic encoding of the data. The input data is the list and Cut sliced audio extracted by ASR in the previous step. In addition, you need to name the model and use this name in subsequent work.


Download the pre-trained model from [here](https://drive.google.com/file/d/1wTg0rchyW_WhWCrbVSKargXFf2GsllIk/view?usp=drive_link) and unzip it and put it in the pretrain folder.

```
cd get_text
python sst.py
```

#### Step 2: Extract audio encoding

The purpose of this step is to reconstruct the audio and extract the audio encoding. Note that the parameters entered need to be the same as those in the previous step.

```
cd get_audio
python get_feature.py
```

#### Step 3: Extract emotional encoding

The purpose of this step is to obtain the emotional encoding of the audio. Note that you need to modify some paths in 'get_semantic.py' and 'lan.py' based on the path to generate the feature previously.

```
cd get_emo
python lan.py
```

### Train your own Model: Fine-tune the VITS and GPT Model

#### Step 1: Fine-tune the VITS Model

