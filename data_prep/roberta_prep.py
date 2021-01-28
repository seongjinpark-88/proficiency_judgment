### Generate dataset for AudioOnly Roberta

import sys

sys.path.append("/work/seongjinpark/tomcat-speech")

import torch
import os
from collections import defaultdict
import pickle
import numpy as np


class RobertaDataPrep(torch.utils.data.Dataset):
    def __init__(self, audio_token_path, response_data_path):
        self.audio_token = audio_token_path
        # with open(length_data, "rb") as length_dict:
        #     self.length_data = pickle.load(length_dict)
        # print(self.length_data)

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
        for wav in os.listdir(audio_token_path):
            name = wav.replace(".txt", "")
            if name in list(self.label_info.keys()):
                self.wav_names.append(name)

        # self.wav_names = [name.replace(".txt", "") for name in os.listdir(audio_token_path) if name in self.label_info.keys()]
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
                audio_token_name = os.path.join(self.audio_token, 
                    key + ".txt")
                at = torch.as_tensor(np.genfromtxt(audio_token_name, 
                    delimiter="\t", dtype=np.int32)[:-1])
        

                if self.label_info[key]['spk'] in test_spk_list:
                    test_dict[key] ={
                        'acc': acc_rating,
                        'flu': flu_rating,
                        'comp': comp_rating,
                        'w2v': at
                    }
                else:
                    train_dict[key] = {
                        'acc': acc_rating,
                        'flu': flu_rating,
                        'comp': comp_rating,
                        'w2v': at
                    }

            train_file_name = "CV_%s_w2v_train.pk" % cv_idx
            test_file_name = "CV_%s_w2v_test.pk" % cv_idx

            with open(os.path.join(data_dir, train_file_name), 'wb') as outfile:
                pickle.dump(train_dict, outfile)

            with open(os.path.join(data_dir, test_file_name), 'wb') as outfile:
                pickle.dump(test_dict, outfile)

class RobertaPrepData(torch.utils.data.Dataset):
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
            w2v = self.data[self.wav_names[idx]]['w2v']
            item = {
                        'acc': acc_rating,
                        'flu': flu_rating,
                        'comp': comp_rating,
                        'w2v': w2v
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
            rhythm_file,
            rnn=False
    ):
        self.w2v_path = os.path.join(data_path, "w2v_roberta")
        self.response_path = os.path.join(data_path, "perception_results")
        self.dict_path = os.path.join(data_path, "cv_data")
        self.cv_path = os.path.join(data_path, "prof_data")

        if "CV_1_w2v_train.pk" not in os.listdir(self.dict_path):
            # w2v_path, audio_data_path, acoustic_data_path, rhythm_file, response_data_path, rnn=False):
            data = RobertaDataPrep(self.w2v_path,
                                   self.response_path)
            data.create_cv(self.dict_path)

        if "train_w2v_cv_list.pt" not in os.listdir(self.cv_path):

            self.train_cv = []
            self.test_cv = []

            for cv in range(0, 5):
                cv_idx = cv + 1
                train_name = "CV_%s_w2v_train.pk" % cv_idx
                test_name = "CV_%s_w2v_test.pk" % cv_idx

                train_dict = pickle.load(open(os.path.join(self.dict_path, train_name), "rb"))
                test_dict = pickle.load(open(os.path.join(self.dict_path, test_name), "rb"))

                train_dataset = RobertaPrepData(train_dict)
                test_dataset = RobertaPrepData(test_dict)

                self.train_cv.append(train_dataset)
                self.test_cv.append(test_dataset)

            print("CREATING DATASET")
            train_name = "train_w2v_cv_list.pt"
            test_name = "test_w2v_cv_list.pt"

            with open(os.path.join(self.cv_path, train_name), "wb") as data_file:
                torch.save(self.train_cv, data_file)

            with open(os.path.join(self.cv_path, test_name), "wb") as data_file:
                torch.save(self.test_cv, data_file)

        else:
            print("LOADING DATASET")
            self.train_cv = torch.load(os.path.join(self.cv_path, "train_w2v_cv_list.pt"))
            self.test_cv = torch.load(os.path.join(self.cv_path, "test_w2v_cv_list.pt"))

    def get_train(self):
        return self.train_cv

    def get_test(self):
        return self.test_cv