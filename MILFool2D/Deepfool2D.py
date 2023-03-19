import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
import collections


def deepfool(bag, net, num_class=2, overshoot=0.02, max_iter=50, mode="att"):

    """
       :param bag:           包
       :param net:           待愚弄的神经网络
       :param num_class:     愚弄的最大类别数，默认与数据集的类别相等
       :param overshoot:     防止扰动的更新消失.
       :param max_iter:      愚弄的最大迭代次数
       :param mode:          梯度计算方式
       :return:              最小扰动、愚弄所需的最小迭代次数、愚弄后的标签、扰动图像
    """

    is_cuda = torch.cuda.is_available()
    bag = bag.squeeze(0)
    if is_cuda:
        bag = bag.cuda()
        net = net.cuda()

    # 分别获取包的预测概率、包标签，以及实例的预测概率
    y_bag, _, y_ins = net.forward(Variable(bag, requires_grad=True))
    y_bag = y_bag.data.cpu().numpy().flatten()
    y_ins = y_ins.data.cpu().numpy().flatten()

    # 获取包预测概率的降序索引，用于判断具体哪一类别用于反向传递
    bag_idx = y_bag.argsort()[::-1]
    # 获取实例预测概率降序后的索引，用于判断扰动哪一张图像
    ins_idx = y_ins.argsort()[::-1]
    # 获取每个实例的形状
    input_shape = bag[0].cpu().numpy().shape
    # 深度拷贝包
    new_bag = copy.deepcopy(bag)
    # 初始化权重
    w = np.zeros(input_shape)
    # 初始化最小扰动
    r_tot = np.zeros(input_shape)

    # 再次计算概率
    new_bag = Variable(new_bag, requires_grad=True)
    y_bag, y_hat, y_ins = net.forward(new_bag)
    y_hat = y_hat.cpu().data.numpy()[0]
    y_per = y_hat

    # 更新扰动
    loop_i = 0
    while y_per == y_hat and loop_i < max_iter:
        pert = np.inf
        y_bag[0, y_hat].backward(retain_graph=True)
        if mode == "att":
            grad_orig = new_bag.grad.data.cpu().numpy().copy()[ins_idx[0]]
        else:
            grad_orig = new_bag.grad.data.cpu().numpy().copy().mean(0)

        # 愚弄每一个类别
        for k in range(1, num_class):
            # 梯度清零
            zero_gradients(new_bag)
            # 再次反向传播
            y_bag[0, bag_idx[k]].backward(retain_graph=True)
            # 获取当前梯度
            if mode == "ave":
                cur_grad = new_bag.grad.data.cpu().numpy().copy()[ins_idx[0]]
            else:
                cur_grad = new_bag.grad.data.cpu().numpy().copy().mean(0)

            # 设置新的w_k和f_k
            w_k = cur_grad - grad_orig
            f_k = (y_bag[0, bag_idx[k]] - y_bag[0, y_hat]).data.cpu().numpy()

            # 计算扰动
            pert_k = abs(f_k)/(np.linalg.norm(w_k.flatten()) + 1e-6)
            # 更新扰动
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # 计算扰动r_i和r_tot
        # 增加1e-4以增加数值稳定性
        r_i = (pert+1e-4) * w / (np.linalg.norm(w) + 1e-6)
        r_tot = np.float32(r_tot + r_i)

        # 更新扰动包
        # new_bag.data[ins_idx[0]] = pert_ins
        if is_cuda:
            new_bag.data = bag + (1 + overshoot) * torch.from_numpy(r_tot).cuda()
        else:
            new_bag.data = bag + (1 + overshoot) * torch.from_numpy(r_tot)

        # 更新扰动包
        # 更新预测
        y_bag, _, y_ins = net.forward(new_bag)
        # 更新预测标签
        y_per = np.argmax(y_bag.data.cpu().numpy().flatten())
        # 更新循环次数
        loop_i += 1

    return (1+overshoot)*r_tot, loop_i, y_hat, y_per


def zero_gradients(x):
    if isinstance(x, torch.Tensor):
        if x.grad is not None:
            x.grad.detach_()
            x.grad.zero_()
    elif isinstance(x, collections.abc.Iterable):
        for elem in x:
            zero_gradients(elem)


def main():
    from MILFool2D.NN2D import Attention
    net = Attention()
    from MILFool2D.DataLoader2D import MnistBags
    import torch.utils.data as data_utils
    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=200,
                                                   seed=1,
                                                   train=False),
                                         batch_size=1,
                                         shuffle=True)
    for bag, y in train_loader:
        # break
        deepfool(bag, net, 2, 0.02, 30)


if __name__ == '__main__':
    main()
