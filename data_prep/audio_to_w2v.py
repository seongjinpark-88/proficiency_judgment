import os
import sys
import numpy as np

import pickle
import torch
import torchaudio
import statistics
import torch.nn.functional as F

import glob

import argparse

from fairseq.models.wav2vec import Wav2VecModel
from fairseq.models.roberta import RobertaModel

'''
from https://github.com/shamanez/BERT-like-is-All-You-Need/blob/master/SPEECH-BERT-TOKENIZATION/convert_aud_to_token.py
'''

# problem_aud = open('problem_aud.txt', 'w')


def print_audio_info(arr):
    print("Min: ", min(arr))
    print("Median: ", statistics.median(arr))
    print("Mean: ", statistics.mean(arr))
    print("Max: ", max(arr))


class EmotionDataPreprocessing():

    def __init__(self, vq_wav2vec):
        # Load vq-wav2vec
        cp = torch.load(vq_wav2vec)
        self.model = Wav2VecModel.build_model(cp['args'], task=None)
        self.model.load_state_dict(cp['model'])
        self.model.eval()

    def preprocess_audio_file(self, idx, filename):
        feats_audio, sr = torchaudio.load(filename, normalization=True)
        print("Audio %s: " % idx, feats_audio.size())
        return feats_audio

    def preprocess_data(self, audio_path, output_path, ext):
        num_items = 1e18
        current_num = 0

        audio_time = []
        if audio_path:
            audio_files = [audio for audio in os.listdir(audio_path) if audio.endswith(ext)]
            print(len(audio_files), "audio_files found")

            for i, audio_file in enumerate(audio_files):
                audio_file_full = os.path.join(audio_path, audio_file)
                audio_features = self.preprocess_audio_file(i, audio_file_full)

                # wav2vec
                z = self.model.feature_extractor(audio_features)
                aggregated_feat = self.model.feature_aggregator(z)
                time = aggregated_feat.size()[2]
                audio_time.append(time)

                output_file = os.path.join(output_path, audio_file.replace(ext, '.pt'))

                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_file, "wb") as output:
                    torch.save(aggregated_feat, output)

                current_num += 1
                if current_num > num_items:
                    break

        print_audio_info(audio_time)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--audio_path', default=None, help='path for raw audio_train files')
    parser.add_argument('-w', '--w2v_path', default=None, help='path for wav2vec model')
    parser.add_argument('-o', '--output', default=None, help='path for output wav2vec data')
    parser.add_argument('-e', '--extension', default='wav', help="extension for sound files")

    args = parser.parse_args()

    audio_path = args.audio_path
    w2v_model = args.w2v_path
    output_path = args.output
    ext = args.extension

    data_processor = EmotionDataPreprocessing(w2v_model)

    data_processor.preprocess_data(audio_path, output_path, ext)

