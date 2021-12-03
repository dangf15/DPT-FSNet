import torch
from torch.utils.data import Dataset
from preprocess import SignalToFrames, ToTensor
import numpy as np
import random
import h5py


class TrainingDataset(Dataset):
    r"""Training dataset."""

    def __init__(self, file_path, frame_size=512, frame_shift=256, nsamples=64000):

        with open(file_path, 'r') as train_file_list:
            self.file_list = [line.strip() for line in train_file_list.readlines()]

        self.nsamples = nsamples
        self.get_frames = SignalToFrames(frame_size=frame_size,
                                         frame_shift=frame_shift)
        self.to_tensor = ToTensor()

    def __len__(self):
        #print(len(self.file_list))
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')
        feature = reader['noisy_raw'][:]
        label = reader['clean_raw'][:]
        reader.close()


        size = feature.shape[0]
        start = random.randint(0, max(0, size  - self.nsamples))
        feature = feature[start:start + self.nsamples]
        label = label[start:start + self.nsamples]

        #print(feature.shape)
        feature = np.reshape(feature, [1, -1])  # [1, sig_len]
        #print(feature.shape)
        label = np.reshape(label, [1, -1])  # [1, sig_len]

       # feature = self.get_frames(feture.shape)ature)  # [1, num_frames, sig_len]      
        feature = self.to_tensor(feature)  # [1, sig_len]
        label = self.to_tensor(label)  # [1, sig_len]


        sig_len = feature.shape[-1]
        feature_ = torch.zeros((1, self.nsamples))
        label_ = torch.zeros((1, self.nsamples))
        feature_[:,:sig_len] = feature
        label_[:,:sig_len] = label
        #print(feature.shape)

        #return feature, label
        return feature_, label_, sig_len


class EvalDataset(Dataset):
    r"""Evaluation dataset."""

    def __init__(self, file_path, frame_size=512, frame_shift=256):

        #self.filename = filename
        with open(file_path, 'r') as validation_file_list:
            self.file_list = [line.strip() for line in validation_file_list.readlines()]

        self.get_frames = SignalToFrames(frame_size=frame_size,
                                         frame_shift=frame_shift)
        self.to_tensor = ToTensor()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        filename = self.file_list[index]
        reader = h5py.File(filename, 'r')

        feature = reader['noisy_raw'][:]
        label = reader['clean_raw'][:]

        feature = np.reshape(feature, [1, -1])  # [1, 1, sig_len]

#        feature = self.get_frames(feature)  # [1, 1, num_frames, frame_size]
    #    print(feature.shape)     
        feature = self.to_tensor(feature)  # [1, 1, num_frames, frame_size]
        label = self.to_tensor(label)  # [sig_len, ]

        return feature, label