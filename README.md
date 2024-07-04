
# Egyptian Arabic Speech Recognition

This project prepares and trains an Automatic Speech Recognition (ASR) system using NeMo. 

## Requirements
To set up the environment and install necessary packages, run:
```sh
pip install -r requirements.txt
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

### Technical Details
- **Framework**: NVIDIA NeMo toolkit
- **Model Architecture**: Conformer
- **Loss Function**: Connectionist Temporal Classification (CTC)
- **Optimizer**: Adamw
- **Learning Rate Scheduling**: Noam Annealing with warmup
- **Training Duration**: 350 epochs


## Evaluation
Download the test set from [here](https://www.kaggle.com/competitions/mct-aic-2/data) and extract in the directory.

You can download the checkpoit using the following command:
```sh
gdown https://drive.google.com/drive/u/1/folders/1w94yoVpkuAHuFkYbouQzkCsM8t8WZSFS --folder
```
Or from [here](https://drive.google.com/drive/u/1/folders/1w94yoVpkuAHuFkYbouQzkCsM8t8WZSFS) and place it in the `/checkpoints` directory.

To evaluate the model on the test set, run the following command:
```sh
python inference.py --checkpoint <path_to_checkpoint> --test_dir <path_to_test_dir>
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






