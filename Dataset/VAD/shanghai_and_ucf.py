import torch.utils.data as data
import numpy as np
import torch
import warnings
torch.set_default_tensor_type('torch.FloatTensor')
warnings.filterwarnings("ignore")


class Dataset(data.Dataset):
    def __init__(self, args, transform=None, test_mode=False, crop=4):
        # 数据集的特征提取类型类型，例如RGB
        self.modality = args.modality
        # 数据集的类型，例如shanghai
        self.dataset = args.dataset
        # 数据集列表
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = './Dataset/VAD/shanghai-i3d-test-10crop.list'
            else:
                self.rgb_list_file = './Dataset/VAD/shanghai-i3d-train-10crop.list'
        else:
            if test_mode:
                self.rgb_list_file = './Dataset/VAD/ucf-i3d-test.list'
            else:
                self.rgb_list_file = './Dataset/VAD/ucf-i3d.list'

        # 转换器
        self.tranform = transform
        # 是否是测试集
        self.test_mode = test_mode
        self._parse_list()
        # 帧的数量
        self.num_frame = 0
        self.crop = crop

    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if not self.test_mode:
            self.label = torch.zeros(len(self.list))
            if self.dataset == "shanghai":
                self.label[:63] = 1
            else:
                self.label[:342] = 1
        else:
            if self.dataset == "shanghai":
                #
                label = np.load("./Dataset/Vad/gt-sh.npy")
            else:
                label = np.load("./Dataset/VAD/gt-ucf.npy")
            self.label = torch.from_numpy(label).float()

    def __getitem__(self, index):
        """
        获取单个样本
        :param index: 样本索引
        """
        # 获取标签
        label = self.label[index]
        # 每个样本都是经过10次裁剪后的结果
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        # 转换类型
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return torch.from_numpy(features).permute(1, 0, 2)[self.crop].unsqueeze(0).float(), label
        else:
            # 处理10次裁剪帧特征
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            # 划分特征初始化
            divided_features = []
            # 遍历每一次
            for feature in features:
                # 转换每个裁剪部分为32x2048
                feature = process_feat(feature, 32)
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return torch.from_numpy(divided_features[self.crop]).unsqueeze(0).float(), label

    def __len__(self):
        """
        获取数据集的大小
        """
        return len(self.list)

    def get_num_frames(self):
        """
        获取帧的大小
        """
        return self.num_frame


def process_feat(feat, length):
    """
    特征处理
    """
    # 新特征
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    # 生成线性空间
    r = np.linspace(0, len(feat), length + 1, dtype=np.int)

    for i in range(length):
        # 强制将视频特征转换到32行
        if r[i] != r[i + 1]:
            new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
        else:
            new_feat[i, :] = feat[r[i], :]
    return new_feat


if __name__ == '__main__':
    from Args.VAD.args_shanghai_and_ucf import parser
    args = parser.parse_args()
    loader = Dataset(args, test_mode=False)
    a = loader.__getitem__(0)
    print(a[0].shape)
    # path = os.listdir(r"D:\Data\VAD\ucf_train_feature\ucf_train_feature")
    # for i in path:
    #     print(i)
