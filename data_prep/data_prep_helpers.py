# prepare text and audio_train for use in neural network models
import math
import os
import random
import sys
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
from pprint import pprint
from torch import nn
from torch.utils.data import Dataset
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils.class_weight import compute_class_weight
from fairseq.models.wav2vec import Wav2VecModel
from sklearn.preprocessing import MinMaxScaler
import torchaudio

import statistics

# classes
from torch.utils.data.sampler import RandomSampler

class GetFeatures:
    """
    Takes input files and gets segmental and/or suprasegmental features
    Current features extracted: XXXX, YYYY, ZZZZ
    """
    def __init__(self, audio_path, opensmile_path, save_path):
        self.apath = audio_path
        self.smilepath = opensmile_path
        self.savepath = save_path
        self.supra_name = None # todo: delete?
        self.segment_name = None # todo: delete?

    #
    # def copy_files_to_single_directory(self, single_dir_path):
    #     """
    #     Copy all files for different speakers to a single directory
    #     single_dir_path : full path
    #     """
    #     if not os.path.isdir(single_dir_path):
    #         os.system("mkdir {0}".format(single_dir_path))
    #     # for f in os.scandir(self.apath):
    #     #     if f.is_dir() and str(f).startswith("S"):
    #     #         print(f)
    #     os.system("cp -r {0}/S*/wav/* {2}/".format(self.apath, single_dir_path))
    #     self.apath = single_dir_path

    def extract_features(self, supra=False, summary_stats=False):
        """
        Extract the required features in openSMILE
        """
        # for file in directory
        for f in os.listdir(self.apath):
            # get all wav files
            if f.endswith('.wav'):
                wavname = f.split('.')[0]
                # extract features
                # todo: replace config files with the appropriate choice
                if supra is True:
                    os.system("{0}/SMILExtract -C {0}/config/IS10_paraling.conf -I {1}/{2}\
                          -lldcsvoutput {3}/{4}.csv".format(self.smilepath, self.apath, f,
                                                            self.savepath, wavname))
                    # self.supra_name = output_name # todo: delete?
                else:
                    if summary_stats is False:
                        os.system("{0}/SMILExtract -loglevel 0 -C {0}/config/IS09_emotion.conf -I {1}/{2}\
                              -csvoutput {3}/{4}.csv".format(self.smilepath, self.apath, f,
                                                                self.savepath, wavname))
                    else:
                        os.system("{0}/SMILExtract -loglevel 0 -C {0}/config/IS10_paraling.conf -I {1}/{2}\
                              -csvoutput {3}/{4}.csv".format(self.smilepath, self.apath, f,
                                                             self.savepath, wavname))
                    # self.segment_name = output_name # todo: delete?

    def get_features_dict(self, dropped_cols=None):
        """
        Get the set of phonological/phonetic features
        """
        # create a holder for features
        feature_set = {}
        feature_length = {}
        scaler = MinMaxScaler()
        # iterate through csv files created by openSMILE
        for csvfile in os.listdir(self.savepath):
            if csvfile.endswith('.csv'):
                csv_name = csvfile.split(".")[0]
                # get data from these files
                csv_data = pd.read_csv("{0}/{1}".format(self.savepath, csvfile), sep=';')
                time_length = csv_data.shape[0]

                # drop name and time frame, as these aren't useful
                if dropped_cols:
                    csv_data = self.drop_cols(csv_data, dropped_cols)
                else:
                    csv_data = csv_data.drop(['name', 'frameTime'], axis=1).to_numpy().tolist()
                    # csv_data = pd.DataFrame(scaler.fit_transform(csv_data), columns=csv_data.columns).to_numpy().tolist()
                if "nan" in csv_data or "NaN" in csv_data or "inf" in csv_data:
                    pprint.pprint(csv_data)
                    print("Data contains problematic data points")
                    sys.exit(1)

                # add it to the set of features
                # feature_set[csv_name] = csv_data
                feature_set[csv_name] = np.mean(csv_data, axis=0)
                feature_length[csv_name] = 1

                # if time_length <= 686:
                #
                #     target_data = np.zeros((686, 32))
                #     target_data[:time_length, :] = np.array(csv_data)
                #
                #     feature_set[csv_name] = target_data
                #     feature_length[csv_name] = time_length
                # else:
                #     diff = time_length - 686
                #
                #     random_start = np.random.randint(0, diff + 1)
                #     end = time_length - diff + random_start
                #     extracted_data = np.array(csv_data)[random_start:end, :]
                #
                #     feature_set[csv_name] = extracted_data
                #     feature_length[csv_name] = 686

        return feature_set, feature_length

    def drop_cols(self, dataframe, to_drop):
        """
        to drop columns from pandas dataframe
        used in get_features_dict
        """
        return dataframe.drop(to_drop, axis=1).to_numpy().tolist()

    # def get_select_cols(self, cols):
    #     """
    #     If you happen to use a conf file that results in too much data
    #     and want to clean it up, select only the columns you want.
    #     suprafile: the path to a csv file containing results
    #     cols: an array of columns that you want to select
    #     Returns data as an np array
    #     """
    #     suprafile = "{0}/{1}.csv".format(self.apath, self.supra_name)
    #     supras = pd.read_csv(suprafile, sep=',')
    #     try:
    #         return supras[cols]
    #     except:
    #         for col in cols:
    #             if col not in supras.columns:
    #                 cols.remove(col)
    #         return supras[cols].to_numpy()

def make_w2v_dict(audio_path="", wav_names=[], rnn=False):
    # list_wavs = wav_names

    audio_dict = {}
    audio_length = {}

    for wav_name in wav_names:
        torch_file = wav_name + ".pt"

        # print(wav_name)
        filename = os.path.join(audio_path, torch_file)

        aggregated_feat = torch.load(filename)

        mel_time = aggregated_feat.size()[2]

        if rnn:
            if mel_time > 980:
                target_tensor = aggregated_feat[:, :, :980]
                audio_length[wav_name] = 980
            else:
                target_tensor = aggregated_feat
                audio_length[wav_name] = mel_time

        else:
            if mel_time <= 980:
                target_tensor = torch.zeros(1, 512, 980)
                target_tensor[:, :, :mel_time] = aggregated_feat
                audio_length[wav_name] = mel_time
            else:
                # target_tensor = torch.zeros(1, 512, 686)

                diff = mel_time - 980

                random_start = np.random.randint(0, diff + 1)
                end = mel_time - diff + random_start

                target_tensor = aggregated_feat[:, :, random_start:end]

                audio_length[wav_name] = 980

        audio_dict[wav_name] = target_tensor.squeeze(0)

    return audio_dict, audio_length


def make_acoustic_dict(audio_path, wav_names, rnn):
    # get wav names

    audio_dict = {}

    audio_length = {}
    for wav_name in wav_names:
        audio = wav_name + ".wav"
        if (".wav") in audio:
            audio_name = audio.replace(".wav", "")
            filename = os.path.join(audio_path, audio)

            waveform, sample_rate = torchaudio.load(filename, normalization=True)

            # get mel_spectrogram
            mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate,
                                                                   hop_length=256,
                                                                   n_mels=96,
                                                                   n_fft=256,
                                                                   pad=0)(waveform)
            # get mfcc
            mfcc = torchaudio.transforms.MFCC(sample_rate, n_mfcc=13,
                                              melkwargs={"hop_length": 256, "n_mels": 96, "n_fft": 256})
            mfcc_feature = mfcc.forward(waveform)
            mfcc_delta = torchaudio.transforms.ComputeDeltas().forward(mfcc_feature)
            mfcc_delta_delta = torchaudio.transforms.ComputeDeltas().forward(mfcc_delta)

            ### Choose feature to use (mel_spec = 1, 96, X, mfcc = 1, mfcc, X)
            # concat_feature = torch.cat((mel_spectrogram, mfcc_feature, mfcc_delta, mfcc_delta_delta), dim=1)
            concat_feature = torch.cat((mfcc_feature, mfcc_delta, mfcc_delta_delta), dim=1)
            # concat_feature = mel_spectrogram

            if rnn:
                ### For RNN, clip the audio_train if it's longer than 596
                ### Else, just use it as it is
                mel_time = concat_feature.size()[2]
                feat_size = concat_feature.size()[1]
                if mel_time > 686:
                    diff = mel_time - 686

                    random_start = np.random.randint(0, diff + 1)
                    end = mel_time - diff + random_start
                    target_tensor = concat_feature[:1, :feat_size, random_start:end]
                    audio_length[audio_name] = 686
                else:
                    target_tensor = concat_feature
                    audio_length[audio_name] = mel_time

                audio_dict[audio_name] = target_tensor
            else:
                ### For CNN, clip the audio_train if it's longer than 596
                ### Else, zero-padding
                mel_time = mel_spectrogram.size()[2]
                feat_size = concat_feature.size()[1]

                target_tensor = torch.zeros(1, feat_size, 686)

                if mel_time > 686:

                    diff = mel_time - 686

                    random_start = np.random.randint(0, diff + 1)
                    end = mel_time - diff + random_start
                    target_tensor = concat_feature[:1, :feat_size, random_start:end]
                    audio_length[audio_name] = 686
                else:
                    target_tensor[:, :, :mel_time] = concat_feature
                    audio_length[audio_name] = mel_time

                audio_dict[audio_name] = target_tensor

    return audio_dict, audio_length

def get_phonological_features(setpath):
    """
    Get the phonological features from a csv file
    """
    phon_dict = {}
    with open(setpath, 'r') as phonfile:
        phondata = phonfile.readlines()
        for i in range(1, len(phondata)):
            # print(phondata[i])
            line = phondata[i].rstrip().split(',')
            wav_name = line[0].split('.')[0]
            data = np.array(line[1:], dtype=np.float32)
            phon_dict[wav_name] = data
    return phon_dict

