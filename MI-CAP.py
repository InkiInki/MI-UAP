import numpy as np
import torch
from MILFool import Deepfool, Trainer
from MILFool.utils import project_perturbation, get_bag_label, print_acc_and_recall
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def generate(tr_set, te_set, net, acc, delta=0.2, max_iter_uni=10, max_iter_df=50, xi=0.5, p=np.inf, num_class=2,
             overshoot=0.01, tr_bag_ratio=0.9, mode="ave"):
    """
    :param tr_set:      Train set
    :param te_set:      Test set
    :param net:         The trained net that will be attacked
    :param acc:         The acc without attack
    :param delta:       The threshold for fooling rate
    :param max_iter_uni:The maximum iterations of MI-CAP
    :param max_iter_df: The maximum iterations of DeepFool
    :param xi:          The magnitude for perturbation
    :param p:           The l-p norm (p==2 or ==infinity)
    :param num_class:   The number of classes
    :param overshoot:   The threshold for DeepFool
    :param tr_bag_ratio:The ratio for train set
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
    fooling_list = [0]
    acc_list = [acc]
    v_list = [v]

    iter = 0
    # The termination condition
    while fooling_list[-1] < 1 - delta and iter < max_iter_uni:
        # Shuffle the index
        np.random.shuffle(index_order)
        print("Fooling  ", iter)
        iter += 1

        # Record
        y_hat_list = torch.tensor(np.zeros(0, dtype=np.int64))
        y_per_list = torch.tensor(np.zeros(0, dtype=np.int64))
        y_list = torch.tensor(np.zeros(0, dtype=np.int64))

        i = 0
        """Get the true label"""
        for batch_index, (bag, label) in enumerate(te_set):
            i += 1
            bag = bag.to(device)
            y_hat = net(bag)[1]
            y_hat_list = torch.cat((y_hat_list, y_hat.cpu()))
            y_list = torch.cat((y_list, label.float()))
        torch.cuda.empty_cache()

        """Get the predicted label with perturbation"""
        for batch_index, (bag, label) in enumerate(te_set):
            bag = bag.squeeze(0)
            bag = bag.to(device)
            # Generate the perturbation with DeepFool
            v = Deepfool.deepfool(bag, net, num_class=num_class, overshoot=overshoot,
                                  max_iter=max_iter_df, mode=mode)[0]
            # Project the perturbation to control its magnitude
            v = project_perturbation(xi, p, v)
            new_bag = bag + torch.as_tensor(v).float().to(device)
            # Predicted
            y_per = net(new_bag)[1]
            # Record
            y_per_list = torch.cat((y_per_list, y_per.cpu()))
        torch.cuda.empty_cache()

        fooling = float(torch.sum((y_hat_list != y_per_list).float())) / len(y_hat_list)
        fooling_list.append(fooling)
        acc = float((y_per_list == y_list).float().sum()) / len(y_list)
        acc_list.append(acc)
        v_list.append(v)

    v_best_idx = np.argmax(fooling_list)
    fooling = fooling_list[v_best_idx]
    acc = acc_list[v_best_idx]
    v = v_list[v_best_idx]

    return v, fooling, acc


def main_shanghai():
    """"""
    import torch.utils.data as data_utils
    from Args.VAD.args_shanghai_and_ucf import parser
    from Dataset.VAD.shanghai_and_ucf import Dataset
    args = parser.parse_args()
    args.dataset = dataset
    tr_data = Dataset(args)
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    tr_loader = data_utils.DataLoader(tr_data, batch_size=1, **loader_kwargs, shuffle=True)
    acc_list, f_acc_list, recall_list, f_recall_list = [], [], [], []
    for i in range(5):
        print("Loop %d" % i)
        trainer = Trainer.Trainer(2048, net_type=net_type)
        acc, recall = trainer.train(tr_loader, tr_loader)
        _, f_acc, f_recall = generate(tr_loader, tr_loader, trainer.best_net, acc, recall, xi=xi, max_iter_uni=10, mode="att")
        print(acc, f_acc, recall, f_recall)
        acc_list.append(acc)
        f_acc_list.append(f_acc)
        recall_list.append(recall)
        f_recall_list.append(f_recall)
    
    print_acc_and_recall(acc_list, f_acc_list, recall_list, f_recall_list)


if __name__ == "__main__":
    xi = 0.01
    dataset = "shanghai"  # shanghai: shanghaiTech; ucf: ucf-crime
    net_type = "ab"  # The attacked network: ab: ABMIL; ga: GAMIL; la: LAMIL; ds: DSMIL; ma: MAMIL
    main_shanghai()
