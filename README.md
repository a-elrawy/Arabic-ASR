
# Egyptian Arabic Speech Recognition

This project prepares and trains an Automatic Speech Recognition (ASR) system using NeMo. 

## Contents
- [Requirements](#requirements)
- [Downloading the Dataset](#downloading-the-dataset)
- [Data Preparation](#data-preparation)
- [Training the ASR Model](#training-the-asr-model)
- [System Architecture and Employed Methodologies](#system-architecture-and-employed-methodologies)
- [Evaluation](#evaluation)
- [References](#references)


## Requirements
If you have issues installing with requirements.txt, you can install the required packages manually using the following commands:
```sh
pip install Cython
pip install nemo_toolkit['asr']
pip install pytorch-lightning==2.2.1
pip install einops
pip install gdown # Optional for downloading the checkpoint
pip install kaggle # Optional for downloading the test set 
```
## Downloading the Dataset

The dataset can be downloaded from the following link: [MTC-AIC2](https://aicgoveg-my.sharepoint.com/:u:/g/personal/n_essam_aic_gov_eg/EWJtic_m6qhBr_2qha55vt0BnL0qqr22G7JIq72Zo_ueGw?e=zyLLC3).

The dataset contains Egyptian Arabic speech data in WAV format and corresponding transcriptions in text files.


After downloading the dataset, extract the contents into the `data` directory.

## Data Preparation
The `data` directory contains scripts for preparing the dataset.

### Scripts

0. **Enter the data directory**:
     ```sh
     cd data
     ```

1. **Creates manifest files for the NeMo ASR pipeline**:
     ```sh
     python create_nemo_manifest.py --mode "train"
     python create_nemo_manifest.py --mode "adapt"
     ```

2. **Training, validation, and test sets**:
     ```sh
     python train_test_split.py
     ```

3. **Exits the data directory**:
      ```sh
      cd ..
      ```

## Training the ASR Model
To train the ASR model, run the following command:

```sh
python examples/asr/asr_ctc/speech_to_text_ctc.py \
  --config-path="../conf/conformer/" \
  --config-name="conformer_ctc_char" \
  model.train_ds.manifest_filepath="data/train_manifest.json" \
  model.validation_ds.manifest_filepath="data/dev_manifest.json" \
  model.test_ds.manifest_filepath="data/test_manifest.json" \
  trainer.accelerator="gpu" \
  trainer.devices=-1 \
  trainer.max_epochs=350
```

This command trains the ASR model using the Conformer architecture with CTC loss. We've trained the model using the `conformer_ctc_char` configuration, which uses a character-level vocabulary. We stopped training after 350 epochs, as the model achieved satisfactory performance.


## System Architecture and Employed Methodologies

### System Architecture
The ASR system is built using the NVIDIA NeMo toolkit, which provides a modular framework for building conversational AI models. The architecture is based on the Conformer model, a convolution-augmented Transformer architecture specifically designed for speech recognition tasks.

- **Encoder**: The Conformer encoder combines convolutional neural networks (CNNs) and self-attention mechanisms to capture both local and global features of the input speech signal. This is achieved through a series of Conformer blocks, each containing a convolution module and a multi-head self-attention module.
- **Decoder**: The system uses a CTC (Connectionist Temporal Classification) decoder, which aligns the input audio features with the output transcription without requiring pre-segmented data.
- **Feature Extractor**: Mel-frequency cepstral coefficients (MFCCs) or spectrograms are used as input features, extracted from the raw audio waveforms.
- **Training Loop**: The model is trained using the Adam optimizer with learning rate scheduling and gradient clipping to stabilize training.

### Employed Methodologies
- **Data Preparation**: The dataset is preprocessed to create manifest files required by NeMo, splitting the data into training, validation, and test sets.
- **Model Training**: The training script uses the `conformer_ctc_char` configuration for the Conformer model with character-level vocabulary. The model is trained for 350 epochs using GPU acceleration.
- **Evaluation**: The trained model is evaluated on a separate test set to assess its performance. Inference is performed using a script that loads the model checkpoint and processes the test data.

### Diarization
Diarization is a critical component of the ASR system, especially for processing telephone recordings with multiple speakers. The diarization process involves several modules, each configurable via the YAML file.

- **Voice Activity Detection (VAD)**: This module detects speech segments within an audio file. The VAD model is configured with parameters such as window length, shift length, and thresholds for speech onset and offset.
- **Speaker Embeddings**: Extracts speaker-specific features from audio segments. The `titanet_large` model is used for this purpose, with multiple scales of window and shift lengths to capture various levels of detail.
- **Clustering**: Groups audio segments into clusters corresponding to different speakers. The clustering module can handle varying numbers of speakers and uses several parameters to enhance speaker counting and clustering accuracy.
- **Multiscale Diarization Decoder (MSDD)**: Uses speaker embeddings to assign speaker labels to each segment. The MSDD model can infer speaker labels in overlapping speech segments and uses a sigmoid threshold to binarize speaker labels.
- **Automatic Speech Recognition (ASR)**: Transcribes the speech segments into text. The ASR model used is `Conformer-CTC-Char`, which is we trained using the mentioned egyptian dataset.



### Technical Details
- **Framework**: NVIDIA NeMo toolkit
- **Model Architecture**: Conformer
- **Loss Function**: Connectionist Temporal Classification (CTC)
- **Optimizer**: Adamw
- **Learning Rate Scheduling**: Noam Annealing with warmup
- **Training Duration**: 350 epochs


## Evaluation
### ASR Model Evaluation
Download the test set from [here](https://www.kaggle.com/competitions/mct-aic-2/data) and extract in the directory.

You can download the checkpoit using the following command:
```sh
gdown https://drive.google.com/drive/folders/1IpHkiMsOndOm8T6UvX6BHPI1xzVZbMdF --folder

```
Or from [here](https://drive.google.com/drive/folders/1IpHkiMsOndOm8T6UvX6BHPI1xzVZbMdF) and place it in the `/checkpoints` directory.

To evaluate the model on the test set, run the following command:
```sh
python inference.py --checkpoint <path_to_checkpoint> --test_dir <path_to_test_dir>
```
### Diarization/Full Evaluation
Download the diarization test set from [here](https://aicgoveg-my.sharepoint.com/:u:/g/personal/n_essam_aic_gov_eg/EdGxtVG3EldPix-hVCIoedcBR2sn-cRiKxfZ6xaLnVie9g?e=th1uWn) and extract in the directory.

To evaluate the diarization model on the test set, run the following command:
```sh
sh diarize.sh wav_dir
# Results will be saved in the `outputs/json` directory
# Make sure the checkpoint is in the `checkpoints` directory with the correct name
```


## References
- NeMo
```@misc{nemo,
  title = {NeMo: a toolkit for Conversational AI and Large Language Models},
  author = {Harper, Eric and Majumdar, Somshubra and Kuchaiev, Oleksii and Li, Jason and Zhang, Yang and Bakhturina, Evelina and Noroozi, Vahid and Subramanian, Sandeep and Koluguri, Nithin and Huang, Jocelyn and Jia, Fei and Balam, Jagadeesh and Yang, Xuesong and Livne, Micha and Dong, Yi and Naren, Sean and Ginsburg, Boris},
  year = {2024},
  url = {https://nvidia.github.io/NeMo/},
  note = {Retrieved from https://github.com/NVIDIA/NeMo}
}
```






