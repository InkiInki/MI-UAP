import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score
from MILFool import BagLoader, Deepfool, MIL, Trainer
from MILFool.utils import get_k_cv_idx

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def project_perturbation(data_point, p, perturbation):
    if p == 2:
        perturbation = perturbation * min(1, data_point / np.linalg.norm(perturbation.flatten(1)))
    elif p == np.inf:
        perturbation = np.sign(perturbation) * np.minimum(abs(perturbation), data_point)
    return perturbation


def get_bag_label(data_loader):
    bags = []
    labels = []
    for bag, label in data_loader:
        bags.append(bag)
        labels.append(label)

    return bags, labels


def generate(tr_set, te_set, net, acc, rec, delta=0.5, max_iter_uni=10, max_iter_df=50, xi=0.5, p=np.inf, num_class=2,
             overshoot=0.01, tr_bag_ratio=0.9, mode="ave"):
    """
    :param tr_set:      训练集
    :param te_set:      测试集
    :param net:         待愚弄神经网络
    :param acc:         未愚弄时的准确率
    :param delta:       愚弄率控制
    :param max_iter_uni:主函数最大迭代次数
    :param max_iter_df: Deepfool最大迭代次数
    :param xi:          扰动控制参数
    :param p:           p范数 (p==2或者p==infinity)
    :param num_class:   数据集的类别
    :param overshoot:   Deepfool的最小扰动控制参数
    :param tr_bag_ratio:最大训练包的数量
    :param mode:
    :return:
    """

    net.to(device)
    # The train bags and their labels
    tr_bag, tr_label = get_bag_label(tr_set)

    # The number of train bags
    max_tr_bag = int(len(tr_bag) * tr_bag_ratio)
    # The index of bags used to generate perturbation
    index_order = np.random.permutation(len(tr_bag))[:max_tr_bag]

    # Initialize the perturbation with the shape (d, ), where d is the dimension of instance in the bag
    v = torch.zeros(tr_bag[0].shape[-1])

    # Record
    fooling_list = [1]
    fooling_recall_list = [1]
    v_list = [v]

    iter = 0
    # The termination condition
    while fooling_list[-1] > delta and iter < max_iter_uni:
        # Shuffle the index
        np.random.shuffle(index_order)
        print("Fooling  ", iter)
        iter += 1

        # Record
        y_hat_list = torch.tensor(np.zeros(0, dtype=np.int64))
        y_per_list = torch.tensor(np.zeros(0, dtype=np.int64))
        y_list = torch.tensor(np.zeros(0, dtype=np.int64))

        i = 0
        # 获取原始标签
        for batch_index, (bag, label) in enumerate(te_set):
            i += 1
            # 获取预测标签
            bag = bag.to(device)
            y_hat = net(bag)[1]
            y_hat_list = torch.cat((y_hat_list, y_hat.cpu()))
            y_list = torch.cat((y_list, label.float()))
        # 清楚缓存
        torch.cuda.empty_cache()
        # 获取扰动后标签
        for batch_index, (bag, label) in enumerate(te_set):
            bag = bag.squeeze(0)
            bag = bag.to(device)
            # 添加扰动
            v = Deepfool.deepfool(bag, net, num_class=num_class, overshoot=overshoot, max_iter=max_iter_df, mode=mode)[0]
            v = project_perturbation(xi, p, v)
            new_bag = bag + torch.as_tensor(v).float().to(device)
            # 计算扰动后标签
            y_per = net(new_bag)[1]
            # 添加扰动标签
            y_per_list = torch.cat((y_per_list, y_per.cpu()))
        # 清空缓存
        torch.cuda.empty_cache()

        # 计算愚弄率
        fooling = accuracy_score(y_list, y_per_list)
        fooling_list.append(fooling)
        v_list.append(v)
        fooling_recall = min(recall_score(y_list, y_per_list, pos_label=1),
                             recall_score(y_list, y_per_list, pos_label=0))
        # print(y_list)
        # print(y_per_list)
        fooling_recall_list.append(fooling_recall)

    v_best_idx = np.argmin(fooling_list)
    fooling = fooling_list[v_best_idx]
    v = v_list[v_best_idx]
    fooling_recall = fooling_recall_list[v_best_idx]

    # std = MinMaxScaler()
    # plot_bag = np.vstack([value.squeeze(0) for value in tr_bag[:10]])
    # print(plot_bag.max(), plot_bag.min())
    # print(v.max(), v.min())
    # plot_bag_v = std.fit_transform(plot_bag + v)
    # plot_bag = std.fit_transform(plot_bag)
    # plot_v = plot_bag_v - plot_bag
    # plt.subplot(3, 1, 1)
    # plt.imshow(plot_bag, cmap="gray")
    # plt.subplot(3, 1, 2)
    # plt.imshow(plot_v, cmap="gray")
    # plt.subplot(3, 1, 3)
    # plt.imshow(plot_bag_v, cmap="gray")
    # plt.show()

    return v, fooling, fooling_recall


def main():
    """"""

    file_name = "C:/Users/zhangyuxuan/Desktop/Data/MIL/Image/elephant.mat"
    mil = MIL.MIL(file_name)
    tr_idxes, te_idxes = get_k_cv_idx(mil.N, k=5)
    fooling_list, acc_list = [], []
    for (tr_idx, te_idx) in zip(tr_idxes, te_idxes):
        tr_loader = BagLoader.BagGenerator(mil.bag_space, mil.bag_lab, tr_idx)
        trainer = Trainer.Trainer(mil.d, "ab")
        te_loader = BagLoader.BagGenerator(mil.bag_space, mil.bag_lab, te_idx)
        acc = trainer.train(tr_loader, te_loader)
        v, fooling, acc = generate(tr_loader, te_loader, trainer.net, acc)
        break


def main_shanghai():
    """"""
    import torch.utils.data as data_utils
    from Args.VAD.args_shanghai_and_ucf import parser
    from Dataset.VAD.shanghai_and_ucf import Dataset
    args = parser.parse_args()
    crop = 4
    net_type = "ds"
    tr_data = Dataset(args, test_mode=False, crop=crop)
    te_data = Dataset(args, test_mode=True, crop=crop)
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    tr_loader = data_utils.DataLoader(tr_data, batch_size=1, **loader_kwargs, shuffle=True)
    te_loader = data_utils.DataLoader(te_data, batch_size=1, **loader_kwargs)
    acc_list, f_acc_list, recall_list, f_recall_list = [], [], [], []
    for i in range(5):
        trainer = Trainer.Trainer(2048, net_type=net_type)
        acc, recall = trainer.train(tr_loader, tr_loader)
        _, f_acc, f_recall = generate(tr_loader, tr_loader, trainer.net, acc, recall, xi=0.01, max_iter_uni=10, mode="ave")
        print(acc, f_acc, recall, f_recall)
        acc_list.append(acc)
        f_acc_list.append(f_acc)
        recall_list.append(recall)
        f_recall_list.append(f_recall)
    print("& $\\pmb{%.3lf, %.3lf}$" % (np.average(acc_list), np.std(acc_list, ddof=1)))
    print("& $\\pmb{%.3lf, %.3lf}$" % (np.average(acc_list) - np.average(f_acc_list), np.std(f_acc_list, ddof=1)))
    print("& $\\pmb{%.3lf, %.3lf}$" % (np.average(recall_list), np.std(recall_list, ddof=1)))
    print("& $\\pmb{%.3lf, %.3lf}$" % (np.average(recall_list) - np.average(f_recall_list), np.std(f_recall_list, ddof=1)))


if __name__ == "__main__":
    # main()
    main_shanghai()
