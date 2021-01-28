import torchaudio, torch
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from fairseq.models.wav2vec import Wav2VecModel
from collections import defaultdict

def normalize_data(data):
    """
    Normalize the input data to try to avoid NaN output + loss
    From: https://machinelearningmastery.com/how-to-improve-neural-network-\
    stability-and-modeling-performance-with-data-scaling/
    """
    scaler = MinMaxScaler()
    # fit and transform in one step
    normalized = scaler.fit_transform(data)
    # inverse transform
    inverse = scaler.inverse_transform(normalized)
    return inverse



def get_phonetic_features(setpath):
    """
    Get the phonological features from a csv file
    """
    phon_dict = {}
    with open(setpath, 'r') as phonfile:
        lines = phonfile.readlines()
        for i in range(1, len(lines)):
            line = lines[i]
            line = line.rstrip().split(',')
            wav_name = line[0].split('.')[0]
            data = [float(item) for item in line[1:]]
            phon_dict[wav_name] = data
    return phon_dict

def get_response_data(data_file):
    with open(data_file, "r") as f:
        data = f.readlines()

    rating_dict = {}
    for i in range(1, len(data)):
        line = data[i].rstrip()
        stim, spk, rating = line.split(",")
        rating_dict[stim] = float(rating)
    return rating_dict

def get_audio_feature(audio_path):
    # get wav names
    wav_names = [wav for wav in os.listdir(audio_path) if wav.endswith("wav")]

    audio_dict = {}
    audio_length = []
    for wav in wav_names:
        wav_name = wav.replace(".wav", "")
        filename = os.path.join(audio_path, wav)

        waveform, sample_rate = torchaudio.load(filename, normalization=True)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate,
                                                               hop_length=256,
                                                               n_mels=96,
                                                               n_fft=256,
                                                               pad=0)(waveform)
        mfcc = torchaudio.transforms.MFCC(sample_rate, n_mfcc=30,
                                          melkwargs={"hop_length": 256, "n_mels": 96, "n_fft": 256})
        mfcc_feature = mfcc.forward(waveform)
        mfcc_delta = torchaudio.transforms.ComputeDeltas().forward(mfcc_feature)
        mfcc_delta_delta = torchaudio.transforms.ComputeDeltas().forward(mfcc_delta)
        # print(mel_spectrogram.size(), mfcc_feature.size(), mfcc_delta.size(), mfcc_delta_delta.size())
        # exit()
        # concat_feature = torch.cat((mel_spectrogram, mfcc_feature, mfcc_delta, mfcc_delta_delta), dim=1)
        # concat_feature = torch.cat((mfcc_feature, mfcc_delta, mfcc_delta_delta), dim=1)
        concat_feature = mel_spectrogram

        mel_time = mel_spectrogram.size()[2]
        if mel_time > 686:
            target_tensor = concat_feature[:1, :96, :686]
            audio_length.append(mel_time)
        else:
            target_tensor = concat_feature
            audio_length.append(mel_time)

        audio_dict[wav_name] = target_tensor
    # print("num. audio: ", len(audio_length))
    # print("min: ", min(audio_length))
    # print("median: ", np.median(audio_length))
    # print("mean: ", np.mean(audio_length))
    # print("max: ", max(audio_length))
    # exit()
    return audio_dict

def get_w2v_feature(audio_path, w2v_model):

    cp = torch.load(w2v_model)

    model = Wav2VecModel.build_model(cp['args'], task=None)
    model.load_state_dict(cp['model'])

    wav_names = [wav for wav in os.listdir(audio_path) if wav.endswith("wav")]

    audio_dict = {}
    audio_length = []
    audio_length = []
    for wav in wav_names:
        wav_name = wav.replace(".wav", "")
        filename = os.path.join(audio_path, wav)

        waveform, sample_rate = torchaudio.load(filename, normalization=True)
        z = model.feature_extractor(waveform)
        aggregated_feat = model.feature_aggregator(z)

        mel_time = aggregated_feat.size()[2]
        if mel_time > 686:
            target_tensor = aggregated_feat[:, :, :686]
            audio_length.append(686)
        else:
            target_tensor = aggregated_feat
            audio_length.append(mel_time)

        audio_dict[wav_name] = target_tensor
    
    return audio_dict

def load_tensor(w2v_name, w2v_dir):
    return torch.load(os.path.join(w2v_dir, w2v_name))


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
                              -lldcsvoutput {3}/{4}.csv".format(self.smilepath, self.apath, f,
                                                                self.savepath, wavname))
                    else:
                        os.system("{0}/SMILExtract -loglevel 0 -C {0}/config/IS09_emotion.conf -I {1}/{2}\
                              -csvoutput {3}/{4}.csv".format(self.smilepath, self.apath, f,
                                                             self.savepath, wavname))
                    # self.segment_name = output_name # todo: delete?

    def get_features_dict(self, dropped_cols=None):
        """
        Get the set of phonological/phonetic features
        """
        # create a holder for features
        feature_set = {}
        # audio_length = []
        # iterate through csv files created by openSMILE
        for csvfile in os.listdir(self.savepath):
            if csvfile.endswith('.csv'):
                csv_name = csvfile.split(".")[0]
                # get data from these files
                csv_data = pd.read_csv("{0}/{1}".format(self.savepath, csvfile), sep=';')
                # drop name and time frame, as these aren't useful
                if dropped_cols:
                    csv_data = self.drop_cols(csv_data, dropped_cols)
                else:
                    csv_data = csv_data.drop(['name', 'frameTime'], axis=1).to_numpy().tolist()
                if "nan" in csv_data or "NaN" in csv_data or "inf" in csv_data:
                    pprint.pprint(csv_data)
                    print("Data contains problematic data points")
                    sys.exit(1)

                # add it to the set of features
                # audio_length.append(np.shape(csv_data)[0])
                # print(np.shape(csv_data))
                if np.shape(csv_data)[0] > 400:
                    final_feat = csv_data[:400]
                else:
                    final_feat = csv_data
                # print(np.shape(final_feat))
                feature_set[csv_name] = final_feat
        # print("num. audio: ", len(audio_length))
        # print("min: ", min(audio_length))
        # print("median: ", np.median(audio_length))
        # print("mean: ", np.mean(audio_length))
        # print("max: ", max(audio_length))
        # exit()
        return feature_set

    def drop_cols(self, dataframe, to_drop):
        """
        to drop columns from pandas dataframe
        used in get_features_dict
        """
        return dataframe.drop(to_drop, axis=1).to_numpy().tolist()