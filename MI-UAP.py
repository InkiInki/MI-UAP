import numpy as np
import torch
from sklearn.metrics import recall_score
from MILFool import Deepfool, Trainer
from MILFool.utils import project_perturbation, get_bag_label, print_acc_and_recall
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def generate(tr_set, te_set, net,
             delta=0.5, max_iter_uni=10, max_iter_df=50, xi=0.05, p=np.inf, num_class=2,
             overshoot=0.02, tr_bag_ratio=0.9, mode="att"):
    """
    xi: The magnitude for MI-UAP's perturbation
    others: Please refer to the MI-CAP
    """

    net.to(device)
    tr_bag, tr_label = get_bag_label(tr_set)
    max_tr_bag = int(len(tr_bag) * tr_bag_ratio)
    index_order = np.random.permutation(len(tr_bag))[:max_tr_bag]
    v = torch.zeros(tr_bag[0].shape[-1])

    fooling_list = [1]
    fooling_recall_list = [1]
    v_list = [v]

    iter = 0
    while fooling_list[-1] > delta and iter < max_iter_uni:
        np.random.shuffle(index_order)
        print("Fooling  ", iter)

        # For each train bag
        for index in index_order:
            bag = tr_bag[index].squeeze(0).to(device)
            # The predicted label
            _, y_hat, _ = net(bag)
            torch.cuda.empty_cache()

            # Add perturbation
            new_bag = bag + torch.as_tensor(v).float().to(device)
            # The predicted label of the perturbed bag
            y_per = net(new_bag)[1]
            # Free again
            torch.cuda.empty_cache()

            # Update the perturbation
            if y_hat == y_per:
                # Get the minimum perturbation
                v_delta, iter_k, _, _ = Deepfool.deepfool(new_bag, net, num_class=num_class,
                                                          overshoot=overshoot, max_iter=max_iter_df, mode=mode)
                # Update and project
                if iter_k < max_iter_df - 1:
                    v += v_delta
                    v = project_perturbation(xi, p, v)

        iter = iter + 1

        y_hat_list = torch.tensor(np.zeros(0, dtype=np.int64))
        y_per_list = torch.tensor(np.zeros(0, dtype=np.int64))
        y_list = torch.tensor(np.zeros(0, dtype=np.int64))

        i = 0
        for batch_index, (bag, label) in enumerate(te_set):
            i += 1
            bag = bag.to(device)
            y_hat = net(bag)[1]
            y_hat_list = torch.cat((y_hat_list, y_hat.cpu()))
            y_list = torch.cat((y_list, label.float()))
        torch.cuda.empty_cache()

        for batch_index, (bag, label) in enumerate(te_set):
            bag = bag.squeeze(0)
            bag = bag.to(device)
            new_bag = bag + torch.as_tensor(v).float().to(device)
            y_per = net(new_bag)[1]
            y_per_list = torch.cat((y_per_list, y_per.cpu()))
        torch.cuda.empty_cache()

        fooling = float(torch.sum((y_list == y_per_list).float())) / len(y_hat_list)
        fooling_list.append(fooling)
        v_list.append(v)
        fooling_recall = min(recall_score(y_list, y_per_list, pos_label=1),
                             recall_score(y_list, y_per_list, pos_label=0))
        fooling_recall_list.append(fooling_recall)

    # Find the perturbation with the best fooling rate
    v_best_idx = np.argmin(fooling_list)
    fooling = fooling_list[v_best_idx]
    v = v_list[v_best_idx].numpy()
    fooling_recall = fooling_recall_list[v_best_idx]

    return v, fooling, fooling_recall


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
        _, f_acc, f_recall = generate(tr_loader, tr_loader, trainer.net, xi=0.01, max_iter_uni=10, mode="ave")
        print(acc, f_acc, recall, f_recall)
        acc_list.append(acc)
        f_acc_list.append(f_acc)
        recall_list.append(recall)
        f_recall_list.append(f_recall)
    print_acc_and_recall(acc_list, f_acc_list, recall_list, f_recall_list)


if __name__ == "__main__":
    dataset = "shanghai"  # shanghai: shanghaiTech; ucf: ucf-crime
    net_type = "ab"  # The attacked network: ab: ABMIL; ga: GAMIL; la: LAMIL; ds: DSMIL; ma: MAMIL
    main_shanghai()
