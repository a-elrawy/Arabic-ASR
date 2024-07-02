
# Egyptian Arabic Speech Recognition

This project prepares and trains an Automatic Speech Recognition (ASR) system using NeMo. 

## Requirements
```sh
pip install nemo_toolkit['asr']
pip install gdown # Optional for downloading the checkpoint
```
## Downloading the Dataset

The dataset can be downloaded from the following link: [MTC-AIC2](https://aicgoveg-my.sharepoint.com/:u:/g/personal/n_essam_aic_gov_eg/EWJtic_m6qhBr_2qha55vt0BnL0qqr22G7JIq72Zo_ueGw?e=zyLLC3).

The dataset contains Egyptian Arabic speech data in WAV format and corresponding transcriptions in text files.


## Data Preparation
## Training the ASR Model
Will be released after the competition ends.


## Evaluation
Download the test set from [here](https://www.kaggle.com/competitions/mct-aic-2/data) and extract in the directory.
Download the checkpoint and place it in the `/checkpoints` directory.

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