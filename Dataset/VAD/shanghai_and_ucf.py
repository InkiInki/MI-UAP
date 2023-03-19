"""
The dataloader for ShanghaiTech and UCF-crime data set.
"""

import torch.utils.data as data
import numpy as np
import torch
import warnings
torch.set_default_tensor_type('torch.FloatTensor')
warnings.filterwarnings("ignore")


class Dataset(data.Dataset):
    def __init__(self, args, transform=None):
        """
        :param  args:     Some args for the dataloader, refer to ./Args/VAD/args_shanghai_and_ucf.py
        """

        self.dataset = args.dataset

        # The type of the images of the input.
        self.modality = args.modality
        # The path of data set.
        if self.dataset == 'shanghai':
            self.rgb_list_file = './Dataset/VAD/shanghai-i3d-train-10crop.list'
        else:
            self.rgb_list_file = './Dataset/VAD/ucf-i3d.list'

        self.tranform = transform
        self.num_frame = 0
        self._parse_list()

    def _parse_list(self):
        """
        Generate the data path and label.
        """
        self.list = list(open(self.rgb_list_file))
        self.label = torch.zeros(len(self.list))
        if self.dataset == "shanghai":
            self.label[:63] = 1
        else:
            self.label[:342] = 1

    def __getitem__(self, index):

        label = self.label[index]
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        # Read images.
        if self.tranform is not None:
            features = self.tranform(features)
        features = features.transpose(1, 0, 2)  # [10, B, T, F]
        divided_features = []
        for feature in features:
            feature = process_feat(feature, 32)
            divided_features.append(feature)
        divided_features = np.array(divided_features, dtype=np.float32)

        return torch.from_numpy(divided_features[4]).unsqueeze(0).float(), label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame


def process_feat(feat, length):
    """
    特征处理
    """
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), length + 1, dtype=np.int)

    for i in range(length):
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat
