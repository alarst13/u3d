import os
from sklearn.model_selection import train_test_split
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class VideoClipsDataset(Dataset):
    def __init__(self, videos_dir, clip_len=16):
        self.videos_dir = videos_dir
        self.clip_len = clip_len

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(videos_dir)):
            for fname in os.listdir(os.path.join(videos_dir, label)):
                self.fnames.append(os.path.join(videos_dir, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        print('Number of random videos: {:d}'.format(len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index,
                            label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        self.label_array = np.array(
            [self.label2index[label] for label in labels], dtype=int)

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.temporal_crop(buffer)
        labels = np.array(self.label_array[index])

        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img)
                        for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, 112, 112, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float32)
            buffer[i] = frame

        return buffer

    def temporal_crop(self, buffer):
        time_index = np.random.randint(buffer.shape[0] - self.clip_len)
        buffer = buffer[time_index:time_index + self.clip_len, :, :, :]
        return buffer
