import os
import csv
import argparse

from nemo.collections.asr.models import EncDecCTCModel

from buckwalter import fromBuckWalter


def main(test_dir, checkpoint):
    asr_model = EncDecCTCModel.load_from_checkpoint(checkpoint)

    files = os.listdir(test_dir)
    audio_files = [os.path.join(test_dir, x) for x in files ]
    audio = [f.split('.')[0] for f in files]

    transcript = asr_model.transcribe(audio_files)
    transcript = [fromBuckWalter(x) for x in transcript]
    
    data = list(zip(audio, transcript))

    filename = "submission.csv"

    with open(filename, mode='w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["audio", "transcript"])  # Writing header
        writer.writerows(data)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint.ckpt', help='Path to the checkpoint')
    parser.add_argument('--test_dir', type=str, default='test', help='Path to the test directory')
    args = parser.parse_args()

    main(args.checkpoint, args.test_dir)