import os
import shutil
import argparse

from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.speaker_utils import rttm_to_labels
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR

from utils_diarization import create_config, process_audio_file
from utils_files import load_json, convert_json, write_json

import torch

def diarize_audio(audio_path, output_dir):
    """
    Diarize audio file and save the results in the output_dir
    """
    # Create Temp dir
    temp_path = os.path.join(output_dir, "temp_outputs")
    os.makedirs(temp_path, exist_ok=True)

    # Load and process audio
    mono_file_path = process_audio_file(audio_path, temp_path)
    
    # Create ASR and Diar Pipeline
    with torch.no_grad():
      cfg = create_config(temp_path, mono_file_path)
      asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
      asr_model = asr_decoder_ts.set_asr_model()

      # Transcribe
      word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)
      # Diarize
      asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
      asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset
      diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)

      # Postprocess to Json
      predicted_speaker_label_rttm_path = f"{temp_path}/pred_rttms/{os.path.basename(mono_file_path).replace('.wav', '')}.rttm"
      pred_labels = rttm_to_labels(predicted_speaker_label_rttm_path)
      trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)

      # Convert to the requested json style
      pred_rttms_dir = os.path.join(temp_path, "pred_rttms")
      transcription_path_to_file = os.path.join(pred_rttms_dir, os.path.basename(mono_file_path).replace(".wav", ".json"))

    data = load_json(transcription_path_to_file)
    converted_data = convert_json(data)

    output_transcription_dir = os.path.join(output_dir , 'json')
    os.makedirs(output_transcription_dir, exist_ok=True)

    # Write results in output dir
    output_transcription_path_to_file = os.path.join(output_transcription_dir, os.path.basename(audio_path).replace(".wav", ".json"))
    write_json(output_transcription_path_to_file, converted_data)

    # Remove temp_dir
    shutil.rmtree(temp_path)

    # Free memory
    del asr_decoder_ts
    del asr_model
    del asr_diar_offline
    torch.cuda.empty_cache()



def main(audio_dir):
    ROOT = os.getcwd()
    output_dir = os.path.join(ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    diarize_audio(audio_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diarize Audio Files')
    parser.add_argument('audio_directory', default="input", type=str, help='Directory containing audio files')
    args = parser.parse_args()
    audio_directory = args.audio_directory
    main(audio_directory)