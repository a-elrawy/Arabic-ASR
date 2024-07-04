
# Egyptian Arabic Speech Recognition

This project prepares and trains an Automatic Speech Recognition (ASR) system using NeMo. 

## Requirements
```sh
pip install Cython
pip install nemo_toolkit['asr']
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

1. **Creates manifest files for the NeMo ASR pipeline**:
     ```sh
     python data/create_nemo_manifest.py --mode "train"
     python data/create_nemo_manifest.py --mode "adapt"
     ```

2. **Training, validation, and test sets**:
     ```sh
     python data/train_test_split.py
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


## Evaluation
Download the test set from [here](https://www.kaggle.com/competitions/mct-aic-2/data) and extract in the directory.

Download the checkpoint from [here](https://drive.google.com/drive/u/1/folders/1w94yoVpkuAHuFkYbouQzkCsM8t8WZSFS)  and place it in the `/checkpoints` directory.

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