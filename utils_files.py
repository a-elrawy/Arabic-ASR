import json
from buckwalter import fromBuckWalter


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def convert_json(input_data):
    output_data = []
    first_speaker = input_data['sentences'][0]["speaker"]
    speaker_map = {}
    next_label = 0
    for entry in input_data['sentences']:
        if entry["speaker"] not in speaker_map:
            if entry["speaker"] == first_speaker:
                speaker_map[entry["speaker"]] = 0
            else:
                speaker_map[entry["speaker"]] = next_label
            next_label += 1

    for sentence in input_data['sentences']:
        output_sentence = {
            "start": float(sentence["start_time"]),
            "end": float(sentence["end_time"]),
            "speaker": speaker_map[sentence['speaker']],
            "text": fromBuckWalter(sentence["text"])
        }
        output_data.append(output_sentence)
    return output_data
