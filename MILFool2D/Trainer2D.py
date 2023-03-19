import torch
from torch import nn
from torch import optim
from MILFool2D import NN2D
from sklearn.metrics import accuracy_score, recall_score

# device = torch.device("cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Trainer:
    def __init__(self, net_type="ab", d=50*4*4, num_channel=1, lr=0.0001):
        if net_type == "ab":
            self.net = NN2D.Attention(d=d, num_channel=num_channel)
        elif net_type == "ga":
            self.net = NN2D.GatedAttention(d=d, num_channel=num_channel)
        elif net_type == "la":
            self.net = NN2D.LossAttention(d=d, num_channel=num_channel)
        elif net_type == "ma":
            self.net = NN2D.MAMIL(d=d, num_channel=num_channel)
        elif net_type == "ds":
            i_classifier = NN2D.FCLayer(d, 1)
            b_classifier = NN2D .BClassifier(input_size=d, output_class=1)
            self.net = NN2D.MILNet(d, i_classifier, b_classifier, num_channel=num_channel)

        self.net.to(device)
        self.best_net = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
        self.n_epochs = 50

    def train(self, tr_loader, te_loader):
        best_acc = -1
        best_recall = -1
        self.net.train()
        for epoch in range(self.n_epochs):
            # 总损失
            total_loss = 0.0
            for batch_idx, (bag, label) in enumerate(tr_loader):
                label = label[0].type(torch.LongTensor)
                bag, label = bag.to(device), label.to(device)
                # 梯度清零
                self.optimizer.zero_grad()

                # 前向传递
                y_prob = self.net(bag)[0]
                # 计算损失
                loss = self.criterion(y_prob, label)
                # 损失累加
                total_loss += loss.data.cpu().detach().numpy()
                # 反向传播
                loss.backward()
                # 步进
                self.optimizer.step()
            # print('%d,  loss: %.4f' % (epoch + 1, total_loss / len(tr_loader)))

            # 输出每一轮的精度
            acc, recall = compute_accuracy(self.net, te_loader)
            # print('Acc: %.3lf, recall %.3lf' % (acc, recall))
            # if best_acc < acc and recall > 0.2:
            if best_acc < acc:
                best_acc = acc
                best_recall = recall
                self.best_net = self.net

        # print('Finished Training')
        return best_acc, best_recall

    def train_vad(self, tr_loader, te_loader, image_loader):
        best_acc = 0
        best_recall = 0
        self.net.train()
        for epoch in range(self.n_epochs):
            # 总损失
            total_loss = 0.0
            for batch_idx, bag_path in enumerate(tr_loader):
                try:
                    bag, label = image_loader(bag_path)
                except FileNotFoundError:
                    continue
                bag, label = bag.squeeze(0).to(device), label.type(torch.LongTensor).to(device)
                # 梯度清零
                self.optimizer.zero_grad()

                # 前向传递
                y_prob = self.net(bag)[0]
                # 计算损失
                loss = self.criterion(y_prob, label)
                # 损失累加
                total_loss += loss.data.cpu().detach().numpy()
                # 反向传播
                loss.backward()
                # 步进
                self.optimizer.step()
                # break
            print('%d,  loss: %.4f' % (epoch + 1, total_loss / len(tr_loader)))
            if epoch >= 10:
                torch.save(self.net.state_dict(), "D:\\Data\\weights\\avenue\\%06d.pt" % epoch)
            #
            #     # 输出每一轮的精度
            #     acc, recall = compute_accuracy(self.net, te_loader)
            #     print('Acc: %.3lf, recall %.3lf' % (acc, recall))
            #     if best_acc < acc:
            #         if recall < 0.2:
            #             continue
            #         best_acc = acc
            #         best_recall = recall
            #         self.best_net = self.net

        print('Finished Training')
        return best_acc, best_recall


def compute_accuracy(net, data_loader):
    """
    计算准确率
    :param net:          训练后神经网络
    :param data_loader:  数据集
    :return:             准确率
    """
    net.eval()
    y_list, y_hat_list = [], []
    for batch_idx, (data, label) in enumerate(data_loader):
        bag_label = label[0]
        if torch.cuda.is_available():
            data, bag_label = data.cuda(), bag_label.cuda()
        y_prob, y_hat, A = net(data)
        # 计算准确率
        # print(bag_label.float(), Y_hat)
        y_list.append(int(label[0].float().numpy()[0]))
        y_hat_list.append(int(y_hat.cpu().numpy()[0]))
    acc = accuracy_score(y_list, y_hat_list)
    recall = min(recall_score(y_list, y_hat_list, pos_label=1),
                 recall_score(y_list, y_hat_list, pos_label=0))

    return acc, recall
