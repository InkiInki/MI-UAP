"""Pytorch dataset object that loads MNIST dataset as bags."""
import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, data_type="mnist",
                 target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        """
        :param data_type:               The data set type, including mnist, cifar10, and stl10
        :param target_number:           The target number for class
        :param mean_bag_length:         
        :param var_bag_length:          
        :param num_bag:                 
        :param seed:                    
        :param train:                   Load the train or test data set
        """
        self.data_type = data_type
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = {"mnist": 60000, "cifar10": 50000, "stl10": 5000}[self.data_type]
        self.num_in_test = {"mnist": 10000, "cifar10": 10000, "stl10": 1000}[self.data_type]

        self.count_list = []

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        # MNIST, CIFAR10
        loader = None
        if self.data_type == "mnist":
            if self.train:
                loader = data_utils.DataLoader(datasets.MNIST('D:/Data/Image/',
                                                              train=True,
                                                              download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_train,
                                               shuffle=True)
            else:
                loader = data_utils.DataLoader(datasets.MNIST('D:/Data/Image/',
                                                              train=False,
                                                              download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_test,
                                               shuffle=False)
        elif self.data_type == "cifar10":
            if self.train:
                loader = data_utils.DataLoader(datasets.CIFAR10('D:/Data/Image/',
                                                                train=True,
                                                                download=True,
                                                                transform=transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_train,
                                               shuffle=True)
            else:
                loader = data_utils.DataLoader(datasets.CIFAR10('D:/Data/Image/',
                                                                train=False,
                                                                download=True,
                                                                transform=transforms.Compose([
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_test,
                                               shuffle=False)
        elif self.data_type == "stl10":
            if self.train:
                loader = data_utils.DataLoader(datasets.STL10('D:/Data/Image/',
                                                              split="train",
                                                              download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_train,
                                               shuffle=True)
            else:
                loader = data_utils.DataLoader(datasets.STL10('D:/Data/Image/',
                                                              split="test",
                                                              download=True,
                                                              transform=transforms.Compose([
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.5,), (0.5,))])),
                                               batch_size=self.num_in_test,
                                               shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int_(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)
            self.count_list.append(min(1, int(labels_in_bag.float().sum().data.cpu().numpy())))

        return bags_list, labels_list

    def __len__(self):
        """
        获取包的数量
        """
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        """
        :param index:       指定的包的索引
        """
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]
        return bag, label


if __name__ == "__main__":
    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=1000,
                                                   seed=1,
                                                   train=False),
                                         batch_size=1,
                                         shuffle=True)
