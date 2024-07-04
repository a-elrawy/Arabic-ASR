import json
import argparse
import pandas as pd
import wave

from buckwalter import toBuckWalter

def get_wav_duration(filepath):
    try:
        with wave.open(filepath, 'r') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)
            return round(duration, 2)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def create_manifest(mode):
    df = pd.read_csv(f'{mode}.csv')
    manifest_file = f'{mode}.json'

    with open(manifest_file, 'w') as json_file:
      for _, row in df.iterrows():
          audio_path = f"{mode}/{row['audio']}.wav"
          transcription = toBuckWalter(row['transcript'])
          duration = get_wav_duration(audio_path)

          if duration is not None:
              manifest_data = {
                  "audio_filepath": audio_path,
                  "text": transcription,
                  "duration": duration
              }

          json.dump(manifest_data, json_file)
          json_file.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', help='train, adapt or test')
    args = parser.parse_args()
    create_manifest(args.mode)