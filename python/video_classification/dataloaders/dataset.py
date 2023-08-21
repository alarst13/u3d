import os
from sklearn.model_selection import train_test_split
import argparse
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, output_dir, root_dir=None, dataset='ucf101', split='train', clip_len=16, preprocess=False, verbose=True):
        self.root_dir, self.output_dir = root_dir, output_dir
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 171
        self.crop_size = 112

        if (not self.check_preprocess()) or preprocess:
            if not self.check_integrity():
                raise RuntimeError('Dataset not found or corrupted.' +
                                   ' You need to download it from the official website.')
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(
                dataset))
            self.preprocess()

        # A workaround to skip the rest of the code when `split` is empty
        if split == '':
            return

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)
        if verbose:
            print('Number of {} videos: {:d}'.format(split, len(self.fnames)))


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
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not self.root_dir:
            raise ValueError("Root directory is not provided.")
        return os.path.exists(self.root_dir)

    def check_preprocess(self):
        # Ensure output_dir is available
        if not self.output_dir:
            raise ValueError("Output directory is not provided.")

        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                          sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self):
        if not self.root_dir:
            raise ValueError(
                "Cannot preprocess data without a root directory.")
        if not self.output_dir:
            raise ValueError("Output directory is not provided.")

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)
            video_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(
                video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(
                train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        capture = cv2.VideoCapture(os.path.join(
            self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(
                        frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(
                    save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img)
                        for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height,
                          self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                        height_index:height_index + crop_size,
                        width_index:width_index + crop_size, :]

        return buffer


class CombinedDataset(VideoDataset):
    """
    This class represents a combined dataset that merges training, validation, 
    and testing splits from a specified video dataset.

    The class is a subclass of the VideoDataset, which means it inherits 
    functionalities of the VideoDataset but also has the capability to handle 
    multiple dataset splits (train, validation, and test).

    Attributes:
    - datasets: List of VideoDataset objects for each split (train, val, test).
    - fnames: List of filenames accumulated from all dataset splits.
    - label_array: Concatenated array of labels from all dataset splits.
    - clip_len: Length of video clips to be processed.

    Args:
    - data_dir (str): Path to the directory where the dataset splits are stored.
    - dataset (str): Name of the dataset (default is 'ucf101').
    - clip_len (int): Length of video clips (default is 16).

    Example:
    combined_data = CombinedDataset(output_dir="/path/to/output", dataset="ucf101", clip_len=16)
    """

    def __init__(self, data_dir, dataset='ucf101', clip_len=16):
        super(CombinedDataset, self).__init__(output_dir=data_dir,
                                              dataset=dataset, split='', clip_len=clip_len)

        splits = ['train', 'val', 'test']

        # You can potentially extend this to avoid initializing the VideoDataset thrice.
        self.datasets = {split: VideoDataset(
            output_dir=data_dir, dataset=dataset, split=split, clip_len=clip_len, verbose=False) for split in splits}

        self.fnames = []
        self.label_array = np.array([], dtype=int)

        for dataset in self.datasets.values():
            self.fnames.extend(dataset.fnames)
            self.label_array = np.concatenate(
                (self.label_array, dataset.label_array))

        self.clip_len = clip_len


def main(args):
    root_dir = args.root_dir
    output_dir = args.output_dir
    dataset_name = 'ucf101' if args.dataset == 'u' else 'hmdb51'

    # Preprocess the dataset
    VideoDataset(output_dir, root_dir, dataset=dataset_name,
                 split='', clip_len=16, preprocess=True)


if __name__ == "__main__":
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description='VideoDataset Parameters')
    parser.add_argument('--dataset', type=str, choices=[
                        'u', 'h'], default='u', help='Dataset name: "u" for UCF101 or "h" for HMDB51')
    parser.add_argument('--root_dir', type=str, default=None,
                        help='Path to the root directory of the downloaded dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Path to the directory where the preprocessed dataset will be saved')
    args = parser.parse_args()

    main(args)
