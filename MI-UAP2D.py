import argparse
import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.metrics import recall_score, accuracy_score
from MILFool2D import Deepfool2D
from MILFool2D import Trainer2D
from MILFool2D.DataLoader2D import MnistBags
from MILFool.utils import project_perturbation, get_bag_label, print_acc_and_recall
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def generate(tr_set, te_set, net, delta=0.5, max_iter_uni=10, max_iter_df=50, p=np.inf, num_class=2,
             overshoot=0.2, tr_bag_ratio=0.9):
    net.to(device)
    tr_bag, tr_label = get_bag_label(tr_set)
    te_bag, te_label = get_bag_label(te_set)

    max_tr_bag = int(len(tr_bag) * tr_bag_ratio)
    index_order = np.random.permutation(len(tr_bag))[:max_tr_bag]

    v = np.zeros((tr_bag[0].shape[-2], tr_bag[0].shape[-1]))

    fooling_list = [1]
    fooling_recall_list = [1]
    v_list = [v]

    iter = 0
    while fooling_list[-1] > delta and iter < max_iter_uni:
        print("Fooling  ", iter)

        for index in index_order:
            bag = tr_bag[index].squeeze(0)
            _, y_hat, _ = net(bag)
            torch.cuda.empty_cache()

            new_bag = bag + torch.as_tensor(v).float()
            y_per = net(new_bag)[1]
            torch.cuda.empty_cache()

            if y_hat == y_per:
                v_delta, iter_k, _, _ = Deepfool2D.deepfool(bag, net, num_class=num_class,
                                                            overshoot=overshoot, max_iter=max_iter_df, mode=mode)
                if iter_k < max_iter_df - 1:
                    v[:, :] += v_delta[0, :, :]
                    v = project_perturbation(xi, p, v)
        iter = iter + 1

        y_hat_list = torch.tensor(np.zeros(0, dtype=np.int64))
        y_per_list = torch.tensor(np.zeros(0, dtype=np.int64))
        y_list = torch.tensor(np.zeros(0, dtype=np.int64))

        i = 0
        for batch_index, (bag, label) in enumerate(zip(te_bag, te_label)):
            i += 1
            bag = bag.to(device)
            y_hat = net(bag)[1]
            y_hat_list = torch.cat((y_hat_list, y_hat.cpu()))
            y_list = torch.cat((y_list, label[0].float()))
        torch.cuda.empty_cache()

        for batch_index, bag in enumerate(te_bag):
            bag = bag.squeeze(0)
            bag = bag.to(device)
            new_bag = bag + torch.as_tensor(v).float()
            y_per = net(new_bag)[1]
            y_per_list = torch.cat((y_per_list, y_per.cpu()))

        torch.cuda.empty_cache()

        fooling = accuracy_score(y_list, y_per_list)
        fooling_list.append(fooling)
        v_list.append(v)

        fooling_recall = min(recall_score(y_list, y_per_list, pos_label=1),
                             recall_score(y_list, y_per_list, pos_label=0))
        fooling_recall_list.append(fooling_recall)

    v_best_idx = np.argmin(fooling_list)
    fooling = fooling_list[v_best_idx]
    v = v_list[v_best_idx]
    fooling_recall = fooling_recall_list[v_best_idx]

    return v, fooling, fooling_recall


def main():
    """"""
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--data_type', type=str, default="mnist", help='the type of databases')
    parser.add_argument('--target_number', type=int, default=9, metavar='T')
    parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML', help='average bag length')
    parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL', help='variance of bag length')
    parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                        help='number of bags in training set')
    parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest', help='number of bags in test set')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    args = parser.parse_args()
    args.data_type = data_type
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    tr_loader = data_utils.DataLoader(MnistBags(
        data_type=args.data_type,
        target_number=args.target_number,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_train,
        seed=args.seed,
        train=True),
        batch_size=1,
        shuffle=True,
        **loader_kwargs)

    te_loader = data_utils.DataLoader(MnistBags(
        data_type=args.data_type,
        target_number=args.target_number,
        mean_bag_length=args.mean_bag_length,
        var_bag_length=args.var_bag_length,
        num_bag=args.num_bags_test,
        seed=args.seed,
        train=False),
        batch_size=1,
        shuffle=False,
        **loader_kwargs)

    acc_list, f_acc_list, recall_list, f_recall_list = [], [], [], []
    trainer = None
    for i in range(5):
        print("Loop %d" % i)
        if args.data_type == "mnist":
            d = 50 * 4 * 4 if net_type != "ma" else 48 * 5 * 5
            trainer = Trainer2D.Trainer(net_type=net_type, d=d, num_channel=1)
        elif args.data_type == "cifar10":
            d = 50 * 5 * 5 if net_type != "ma" else 48 * 6 * 6
            trainer = Trainer2D.Trainer(net_type=net_type, d=d, num_channel=3)
        elif args.data_type == "stl10":
            d = 50 * 21 * 21 if net_type != "ma" else 48 * 22 * 22
            trainer = Trainer2D.Trainer(net_type=net_type, d=d, num_channel=3)
        acc, recall = trainer.train(tr_loader, te_loader)
        _, f_acc, f_recall = generate(tr_loader, te_loader, trainer.best_net, max_iter_uni=10)
        print(acc, f_acc, recall, f_recall)
        acc_list.append(acc)
        f_acc_list.append(f_acc)
        recall_list.append(recall)
        f_recall_list.append(f_recall)
    
    print_acc_and_recall(acc_list, f_acc_list, recall_list, f_recall_list)


if __name__ == "__main__":
    xi = 0.2  # The magnitude of perturbation
    mode = "ave"  # ave or att
    net_type = "ab"  # The attacked network: ab: ABMIL; ga: GAMIL; la: LAMIL; ds: DSMIL; ma: MAMIL
    data_type = "mnist"  # mnist, cifar10, stl10
    main()
