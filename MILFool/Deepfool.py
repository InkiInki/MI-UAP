import numpy as np
import torch as torch
import copy
import collections
from torch.autograd import Variable


def deepfool(bag, net, num_class=2, overshoot=0.02, max_iter=50, mode="att"):
    """
       :param bag:           The bag in MIL
       :param net:           The trained net
       :param num_class:     The number of classes
       :param overshoot:     The param used to guarantee perturbation updates
       :param max_iter:      The maximum iteration of DeepFool
       :param mode:          The computation mode of gradient, ave or att
    """

    is_cuda = torch.cuda.is_available()
    bag = bag.squeeze(0)
    if is_cuda:
        bag = bag.cuda()
        net = net.cuda()

    y_bag, _, y_ins = net.forward(Variable(bag, requires_grad=True))
    y_bag = y_bag.data.cpu().numpy().flatten()
    y_ins = y_ins.data.cpu().numpy().flatten()

    bag_idx = y_bag.argsort()[::-1]
    ins_idx = y_ins.argsort()[::-1]
    input_shape = bag[0].cpu().numpy().shape
    new_bag = copy.deepcopy(bag)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    new_bag = Variable(new_bag, requires_grad=True)
    y_bag, y_hat, _ = net.forward(new_bag)
    y_hat = y_hat.cpu().data.numpy()[0]
    y_per = y_hat

    """Update perturbation"""
    loop_i = 0
    while y_per == y_hat and loop_i < max_iter:
        pert = np.inf
        y_bag[0, y_hat].backward(retain_graph=True)
        if mode == "att":
            grad_orig = new_bag.grad.data.cpu().numpy().copy()[ins_idx[0]]
        else:
            grad_orig = new_bag.grad.data.cpu().numpy().copy().mean(0)

        # For each category
        for k in range(1, num_class):
            zero_gradients(new_bag)
            y_bag[0, bag_idx[k]].backward(retain_graph=True)
            if mode == "ave":
                cur_grad = new_bag.grad.data.cpu().numpy().copy()[ins_idx[0]]
            else:
                cur_grad = new_bag.grad.data.cpu().numpy().copy().mean(0)

            w_k = cur_grad - grad_orig
            f_k = (y_bag[0, bag_idx[k]] - y_bag[0, y_hat]).data.cpu().numpy()

            pert_k = abs(f_k)/(np.linalg.norm(w_k.flatten()) + 1e-6)
            if pert_k < pert:
                pert = pert_k
                w = w_k

        r_i = (pert+1e-4) * w / (np.linalg.norm(w) + 1e-6)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            new_bag.data = bag + (1 + overshoot) * torch.as_tensor(r_tot).cuda()
        else:
            new_bag.data = bag + (1 + overshoot) * torch.as_tensor(r_tot)

        y_bag, _, y_ins = net.forward(new_bag)
        y_per = np.argmax(y_bag.data.cpu().numpy().flatten())
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
