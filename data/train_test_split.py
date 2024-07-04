import os
import json
import random


def write_subset(filename, data):
    with open(filename, 'w') as json_file:
        for entry in data:
            entry['audio_filepath'] = 'data/' + entry['audio_filepath']
            json.dump(entry, json_file)
            json_file.write('\n')


def load_manifest(manifest_file):
    manifest_data = []
    if os.path.isfile(manifest_file):
        with open(manifest_file, 'r') as json_file:
            for line in json_file:
                manifest_data.append(json.loads(line))
    return manifest_data


if __name__ == '__main__':

    train_manifest = 'train.json'
    adapt_manifest = 'adapt.json'

    train_data = load_manifest(train_manifest)
    adapt_data = load_manifest(adapt_manifest)

    manifest_data = train_data + adapt_data
    random.shuffle(manifest_data)

    train_ratio = 0.9  
    dev_ratio = 0.05    
    test_ratio = 0.05  

    total_samples = len(manifest_data)
    train_end = int(total_samples * train_ratio)
    dev_end = train_end + int(total_samples * dev_ratio)

    train_data = manifest_data[:train_end]
    dev_data = manifest_data[train_end:dev_end]
    test_data = manifest_data[dev_end:]


    write_subset('train_manifest.json', train_data)
    write_subset('dev_manifest.json', dev_data)
    write_subset('test_manifest.json', test_data)

    print(f"Data split into train ({len(train_data)} samples), dev ({len(dev_data)} samples), and test ({len(test_data)} samples) sets.")
