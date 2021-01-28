import os
import data_prep.data_prep as dp
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


class DataPrep:
    """
    A class to prepare proficiency data as an input to the model
    """

    def __init__(self, dataloader_path="./data", data_path="../keras_asa/", phonetic_feature="IS09_featureset", w2v_model=""):

        check_file = "/train.pth"
        if os.path.exists(dataloader_path + check_file):
            print("LOADING TRAIN SET")
            self.train_data_loader = torch.load(dataloader_path + "train.pth")
            print("LOADING TEST SET")
            self.test_data_loader = torch.load(dataloader_path + "test.pth")

        else:
            print("LOADING DATA")
            self.wav_dir = os.path.join(data_path, "audio/mono")
            self.phonetic_feat_dir = os.path.join(data_path, "audio", phonetic_feature)
            self.perception_dir = os.path.join(data_path, "data/perception_results")
            # print("LOAD AUDIO FEATURE")
            # self.audio_feature = dp.get_audio_feature(self.wav_dir)
            # self.audio_feature = dp.get_w2v_feature(self.wav_dir, w2v_model)
            print("LOAD RHYTHM FEATURE")
            self.phonetic_feature = dp.get_phonetic_features(os.path.join(data_path, "data/rhythm.csv"))
            print("LOAD OPENSMILE FEATURE")
            self.phono_feature = dp.GetFeatures(self.wav_dir, "~/opensmile-2.3.0", self.phonetic_feat_dir)
            # summary_test.extract_features(summary_stats=True)
            # self.acoustic_features = self.phono_feature.get_features_dict(dropped_cols=['name', 'frameTime',
            #                                                                   "pcm_RMSenergy_sma_de",
            #                                                                   "pcm_fftMag_mfcc_sma_de[1]",
            #                                                                   "pcm_fftMag_mfcc_sma_de[2]",
            #                                                                   "pcm_fftMag_mfcc_sma_de[3]",
            #                                                                   "pcm_fftMag_mfcc_sma_de[4]",
            #                                                                   "pcm_fftMag_mfcc_sma_de[5]",
            #                                                                   "pcm_fftMag_mfcc_sma_de[6]",
            #                                                                   "pcm_fftMag_mfcc_sma_de[7]",
            #                                                                   "pcm_fftMag_mfcc_sma_de[8]",
            #                                                                   "pcm_fftMag_mfcc_sma_de[9]",
            #                                                                   "pcm_fftMag_mfcc_sma_de[10]",
            #                                                                   "pcm_fftMag_mfcc_sma_de[11]",
            #                                                                   "pcm_fftMag_mfcc_sma_de[12]",
            #                                                                   "pcm_zcr_sma_de", "voiceProb_sma_de",
            #                                                                   "F0_sma_de"])
            self.acoustic_features = self.phono_feature.get_features_dict(dropped_cols=['name', 'frameTime'])

            self.speaker_list = ["S02", "S03", "S04", "S05", "S07", "S08", "S09", "S19", "S21", "S22", "S23", "S24", "S25",
                            "S26", "S28"]

            self.acc_data = dp.get_response_data(os.path.join(self.perception_dir, "accented_avgs.csv"))
            self.flu_data = dp.get_response_data(os.path.join(self.perception_dir, "fluency_avgs.csv"))
            self.com_data = dp.get_response_data(os.path.join(self.perception_dir, "comp_avgs.csv"))

            raw_feat = []
            raw_feat_len = []
            phono_feat = []
            phono_feat_len = []
            phonetic_feat = []

            ys_acc = []
            ys_flu = []
            ys_comp = []

            print("CREATING DATALOADERS")
            for key in self.acc_data.keys():

                # create acoustic info
                raw_info = dp.load_tensor(key + ".pt", "data/w2v/")
                # raw_info = self.audio_feature[key]
                raw_feat_len.append(raw_info.size()[2])
                raw_info = raw_info.squeeze(0).transpose(0, 1)
                raw_feat.append(raw_info)

                # create phonetic info
                # print(np.shape(self.acoustic_features[key]))
                phono_info = dp.normalize_data(self.acoustic_features[key])
                phono_feat_len.append(np.shape(phono_info)[0])
                phono_feat.append(torch.tensor(phono_info))
                # print(np.shape(phono_info))
                # print(phono_info)

                # create phonological info
                phonetic_info = self.phonetic_feature[key]
                phonetic_feat.append(phonetic_info)

                ys_acc.append(self.acc_data[key])
                ys_flu.append(self.flu_data[key])
                ys_comp.append(self.com_data[key])

            raw_feat = torch.nn.utils.rnn.pad_sequence(raw_feat, batch_first=True)
            raw_feat_len = torch.tensor(raw_feat_len)
            phono_feat = torch.nn.utils.rnn.pad_sequence(phono_feat, batch_first=True)
            phono_feat_len = torch.tensor(phono_feat_len)
            phonetic_feat = torch.tensor(phonetic_feat)
            # print(np.shape(ys_acc))
            # print(ys_acc)
            ys_acc = torch.tensor(ys_acc)
            ys_flu = torch.tensor(ys_flu)
            ys_comp = torch.tensor(ys_comp)

            self.data = TensorDataset(raw_feat, raw_feat_len, phono_feat, phono_feat_len, phonetic_feat, ys_acc, ys_flu, ys_comp)

            self.train_dataset, self.test_dataset = \
                train_test_split(self.data, test_size=0.33, random_state=8008)

            self.train_data_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
            self.test_data_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=True)

            print("SAVE DATALOADERS")
            torch.save(self.train_data_loader, os.path.join(dataloader_path, "train.pth"))
            torch.save(self.test_data_loader, os.path.join(dataloader_path, "test.pth"))

    def get_train(self):
        return self.train_data_loader

    def get_test(self):
        return self.test_data_loader
