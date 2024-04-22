# GPTSOVITS2

## SECTION 1 : Some Demo

### Demo 1:
[![Sudoku AI Solver](img/cover1.png)](https://youtu.be/eDnNtcXmCNE)
### Demo 2:
[![Sudoku AI Solver](img/cover3.png)](https://youtu.be/eDnNtcXmCNE)
### Demo 3:
[![Sudoku AI Solver](img/cover2.png)](https://youtu.be/eDnNtcXmCNE)

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT

### Project Overview:
This project develops an advanced voice cloning and video synchronization technology that enables users to clone any person's voice and apply it to a video, while adjusting the mouth shape of the person in the video to match the newly generated audio. This technology provides unprecedented flexibility and innovation for creating lifelike videos.

### Technical details:
The project utilizes deep learning algorithms to analyze and clone the characteristics of the target voice, including pitch, rhythm and intonation. The Sound Clone module captures subtle articulation characteristics to produce audio that closely resembles the original recording. In addition, our video processing technology uses precise facial tracking and image processing technology to adjust the mouth shape in the video to ensure accurate audio and video synchronization and enhance visual and auditory consistency.

### Application scenarios:
This technology can be widely used in a variety of scenarios, including film post-production, virtual reality, video games, advertising production, and news reporting. Not only can it be used for entertainment and commercial advertising, it can also reproduce the voices and expressions of specific characters in simulated emergencies or training videos, providing a more realistic interactive experience.

### Innovation:
The innovation of this project is that it combines speech recognition, artificial intelligence synthesis and advanced image processing technology to go beyond simply cloning sounds or editing videos, but to create a comprehensive, multi-modal synchronization system that can operate without Recreate and present any content with the presence of the original audio speaker.

### Future outlook:
With the further improvement and optimization of the technology, we expect this project to promote the development of personalized media content and provide users with a richer and more customized audio-visual experience. In addition, the development of this technology will also promote the establishment and improvement of relevant legal, ethical and privacy protection standards.


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

