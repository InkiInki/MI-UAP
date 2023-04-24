import torch
from torch import nn
from torch import optim
from MILFool import NN
from sklearn.metrics import accuracy_score, recall_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Trainer:
    def __init__(self, d, net_type="ab"):
        if net_type == "ab":
            self.net = NN.Attention(d)
        elif net_type == "ga":
            self.net = NN.GatedAttention(d)
        elif net_type == "la":
            self.net = NN.LossAttention(d)
        elif net_type == "ma":
            self.net = NN.MAMIL(d)
        elif net_type == "ds":
            i_classifier = NN.FCLayer(d, 1)
            b_classifier = NN.BClassifier(input_size=d, output_class=1)
            self.net = NN.MILNet(i_classifier, b_classifier)
        # else: you can add other networks

        self.net.to(device)
        self.best_net = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-5)
        self.n_epochs = 50

    def train(self, tr_loader, te_loader):
        best_acc = -1
        best_recall = -1
        self.net.train()
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for batch_idx, (bag, label) in enumerate(tr_loader):
                label = label.type(torch.LongTensor)
                bag, label = bag.to(device), label.to(device)
                self.optimizer.zero_grad()

                y_prob = self.net(bag)[0]
                loss = self.criterion(y_prob, label)
                total_loss += loss.data.cpu().detach().numpy()
                loss.backward()
                self.optimizer.step()
            print('%d,  loss: %.4f' % (epoch + 1, total_loss / len(tr_loader)))

            acc, recall = compute_accuracy(self.net, te_loader)
            print('Acc: %d %%, recall %d %%' % (100 * acc, 100*recall))
            if best_acc < acc:
                best_acc = acc
                best_recall = recall
                self.best_net = self.net

        print('Finished Training')
        return best_acc, best_recall


def compute_accuracy(net, data_loader):
    net.eval()
    y_list, y_hat_list = [], []
    for batch_idx, (data, label) in enumerate(data_loader):
        bag_label = label[0]
        if torch.cuda.is_available():
            data, bag_label = data.cuda(), bag_label.cuda()
        y_prob, y_hat, A = net(data)
        y_list.append(int(label.numpy()[0]))
        y_hat_list.append(int(y_hat.cpu().numpy()[0]))
    acc = accuracy_score(y_list, y_hat_list)
    recall = min(recall_score(y_list, y_hat_list, pos_label=1),
                 recall_score(y_list, y_hat_list, pos_label=0))

    return acc, recall
