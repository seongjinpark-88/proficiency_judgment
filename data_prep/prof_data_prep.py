# prepare proficiency input for use

import os
import sys

import numpy as np
import pandas as pd

import torch
from torch import nn
import pickle
from torch.utils.data import Dataset

from data_prep.data_prep_helpers import (
    make_w2v_dict, make_acoustic_dict, GetFeatures, get_phonological_features)

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict, defaultdict

from torch.nn.utils.rnn import pad_sequence


class PrepData:
    def __init__(self, w2v_path, audio_data_path, acoustic_data_path, rhythm_file, response_data_path, rnn=False):
        self.w2v_path = w2v_path
        self.audio_path = audio_data_path
        self.acoustic_path = acoustic_data_path
        self.rhythm = rhythm_file

        resp_files = ["accented_avgs.csv", "fluency_avgs.csv", "comp_avgs.csv"]

        self.label_info = defaultdict(dict)

        for file in resp_files:
            with open(os.path.join(response_data_path, file), "r") as f:
                data = f.readlines()

                for i in range(1, len(data)):
                    # rating, stim_name, stim_file = data[i].rstrip().split(",")
                    # print(data[i])
                    stim_name, spk, rating = data[i].rstrip().split(",")
                    # spk, _ = stim_name.split("_")
                    if "accented" in file:
                        self.label_info[stim_name]['acc'] = float(rating)
                    elif "fluency" in file:
                        self.label_info[stim_name]['flu'] = float(rating)
                    else:
                        self.label_info[stim_name]['comp'] = float(rating)

                    self.label_info[stim_name]['spk'] = spk

        self.wav_names = []

        self.wav_names = [name for name in list(self.label_info.keys())]

        # self.audio_dict, self.audio_length = make_w2v_dict(self.w2v_path, self.wav_names, rnn=rnn)
        self.audio_dict, self.audio_length = make_acoustic_dict(self.audio_path, self.wav_names, rnn=rnn)
        acoustics = GetFeatures(self.audio_path, "~/opensmile-2.3.0", self.acoustic_path)
        # acoustics.extract_features(supra=True, summary_stats=False)
        self.acoustic_feats, self.acoustic_length = acoustics.get_features_dict(dropped_cols=["name", "frameTime"])
        # self.acoustic_feats = self.acoustic_feats.get_features_dict(dropped_cols=['name', 'frameTime'])

        self.rhythm_feats = get_phonological_features(self.rhythm)

    def create_cv(self, data_dir):
        cv1 = ["S19", "S03", "S02"]
        cv2 = ["S23", "S05", "S04"]
        cv3 = ["S24", "S07", "S21"]
        cv4 = ["S25", "S08", "S22"]
        cv5 = ["S28", "S09", "S26"]

        cv = [cv1, cv2, cv3, cv4, cv5]

        for i in range(0, 5):
            train_dict = defaultdict(dict)
            test_dict = defaultdict(dict)
            test_spk_list = cv[i]
            cv_idx = i + 1

            for key in self.label_info.keys():
                acc_rating = self.label_info[key]['acc']
                flu_rating = self.label_info[key]['flu']
                comp_rating = self.label_info[key]['comp']
                w2v = self.audio_dict[key]
                w2v_len = self.audio_length[key]
                acoustic = np.array(self.acoustic_feats[key], dtype=np.float32)
                acoustic_len = int(self.acoustic_length[key])
                rhythm = self.rhythm_feats[key]

                if self.label_info[key]['spk'] in test_spk_list:
                    test_dict[key] ={
                        'acc': acc_rating,
                        'flu': flu_rating,
                        'comp': comp_rating,
                        'audio': w2v,
                        'audio_length': w2v_len,
                        'acoustic': acoustic,
                        'acoustic_length': acoustic_len,
                        'rhythm': rhythm
                    }
                else:
                    train_dict[key] = {
                        'acc': acc_rating,
                        'flu': flu_rating,
                        'comp': comp_rating,
                        'audio': w2v,
                        'audio_length': w2v_len,
                        'acoustic': acoustic,
                        'acoustic_length': acoustic_len,
                        'rhythm': rhythm
                    }

            train_file_name = "CV_%s_train.pk" % cv_idx
            test_file_name = "CV_%s_test.pk" % cv_idx

            with open(os.path.join(data_dir, train_file_name), 'wb') as outfile:
                pickle.dump(train_dict, outfile)

            with open(os.path.join(data_dir, test_file_name), 'wb') as outfile:
                pickle.dump(test_dict, outfile)

    def create_all_data(self, data_dir):
        data_dict = defaultdict(dict)

        for key in self.label_info.keys():
            acc_rating = self.label_info[key]['acc']
            flu_rating = self.label_info[key]['flu']
            comp_rating = self.label_info[key]['comp']
            w2v = self.audio_dict[key]
            w2v_len = self.audio_length[key]
            acoustic = np.array(self.acoustic_feats[key], dtype=np.float32)
            acoustic_len = int(self.acoustic_length[key])
            rhythm = self.rhythm_feats[key]

            data_dict[key] = {
                'acc': acc_rating,
                'flu': flu_rating,
                'comp': comp_rating,
                'audio': w2v,
                'audio_length': w2v_len,
                'acoustic': acoustic,
                'acoustic_length': acoustic_len,
                'rhythm': rhythm
            }

        data_file_name = "all_data.pk"

        with open(os.path.join(data_dir, data_file_name), 'wb') as outfile:
            pickle.dump(data_dict, outfile)


class ProfPrepData(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.data = data_dict
        self.wav_names = list(self.data.keys())

    def __len__(self):
        return len(self.wav_names)

    def __getitem__(self, idx):
        try:
            # print(self.wav_names[idx])
            acc_rating = self.data[self.wav_names[idx]]['acc']
            flu_rating = self.data[self.wav_names[idx]]['flu']
            comp_rating = self.data[self.wav_names[idx]]['comp']
            w2v = self.data[self.wav_names[idx]]['audio']
            w2v_len = self.data[self.wav_names[idx]]['audio_length']
            # print("w2v: ", np.shape(w2v))
            # print("w2v len: ", w2v_len)
            acoustic = self.data[self.wav_names[idx]]['acoustic']
            acoustic_length = self.data[self.wav_names[idx]]['acoustic_length']
            rhythm = self.data[self.wav_names[idx]]['rhythm']
            # print("acoustic: ", np.shape(acoustic))
            # print("acoustic_len: ", acoustic_length)
            # print("rhythm ", np.shape(rhythm))
            item = {
                        'acc': acc_rating,
                        'flu': flu_rating,
                        'comp': comp_rating,
                        'audio': w2v,
                        'length': w2v_len,
                        'acoustic': acoustic,
                        'acoustic_length': acoustic_length,
                        'rhythm': rhythm
                    }
            return item

        except KeyError:
            next


class DataPrep:
    """
    A class to prepare meld for input into a generic Dataset
    """

    def __init__(
            self,
            data_path,
            acoustic_path,
            rhythm_file,
            rnn=False
    ):
        self.w2v_path = os.path.join(data_path, "w2v_large")
        self.wav_path = os.path.join(data_path, "wavs")
        self.acoustic_path = acoustic_path
        self.response_path = os.path.join(data_path, "perception_results")
        self.rhythm = os.path.join(self.response_path, rhythm_file)
        self.dict_path = os.path.join(data_path, "cv_data")
        self.cv_path = os.path.join(data_path, "prof_data")
        self.rnn = rnn

    def generate_cv_data(self):
        if "CV_1_train.pk" not in os.listdir(self.dict_path):
            # w2v_path, audio_data_path, acoustic_data_path, rhythm_file, response_data_path, rnn=False):
            data = PrepData(self.w2v_path,
                            self.wav_path,
                            self.acoustic_path,
                            self.rhythm,
                            self.response_path,
                            rnn=self.rnn)
            data.create_cv(self.dict_path)

        if "train_cv_list.pt" not in os.listdir(self.cv_path):

            self.train_cv = []
            self.test_cv = []

            for cv in range(0, 5):
                cv_idx = cv + 1
                train_name = "CV_%s_train.pk" % cv_idx
                test_name = "CV_%s_test.pk" % cv_idx

                train_dict = pickle.load(open(os.path.join(self.dict_path, train_name), "rb"))
                test_dict = pickle.load(open(os.path.join(self.dict_path, test_name), "rb"))

                train_dataset = ProfPrepData(train_dict)
                test_dataset = ProfPrepData(test_dict)

                self.train_cv.append(train_dataset)
                self.test_cv.append(test_dataset)

            print("CREATING DATASET")
            train_name = "train_cv_list.pt"
            test_name = "test_cv_list.pt"

            with open(os.path.join(self.cv_path, train_name), "wb") as data_file:
                torch.save(self.train_cv, data_file)

            with open(os.path.join(self.cv_path, test_name), "wb") as data_file:
                torch.save(self.test_cv, data_file)

        else:
            print("LOADING DATASET")
            self.train_cv = torch.load(os.path.join(self.cv_path, "train_cv_list.pt"))
            self.test_cv = torch.load(os.path.join(self.cv_path, "test_cv_list.pt"))

    def generate_all_data(self):
        if "all_data.pk" not in os.listdir(self.dict_path):
            data = PrepData(self.w2v_path,
                            self.wav_path,
                            self.acoustic_path,
                            self.rhythm,
                            self.response_path,
                            rnn=self.rnn)
            data.create_cv(self.dict_path)

        if "data.pt" not in os.listdir(self.cv_path):
            data_dict = pickle.load(open(os.path.join(self.dict_path, "all_data.pk"), "rb"))
            self.dataset = ProfPrepData(data_dict)
            with open(os.path.join(self.cv_path, "data.pt"), "wb") as data_file:
                torch.save(self.dataset, data_file)

        else:
            print("LOADING DATASET")
            self.all_data = torch.load(os.path.join(self.cv_path, "data.pt"))

    def get_train(self):
        return self.train_cv

    def get_test(self):
        return self.test_cv

    def get_all(self):
        return self.all_data
