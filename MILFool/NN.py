import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d, C=2):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(d, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, C),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part2(x.float())  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A.reshape(-1, 1), 1, 0)
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H.reshape(-1, self.L))  # KxL

        Y_prob = self.classifier(M)
        Y_hat = Y_prob.max(1)[1]

        return Y_prob, Y_hat, A


class GatedAttention(nn.Module):
    def __init__(self, d, C=2):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(d, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, C),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        H = self.feature_extractor_part2(x)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication # NxK
        A = torch.transpose(A.reshape(-1, 1), 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H.reshape(-1, self.L))  # KxL

        Y_prob = self.classifier(M)
        Y_hat = Y_prob.max(1)[1]

        return Y_prob, Y_hat, A


class AttentionLayer(nn.Module):
    def __init__(self, dim=512):
        super(AttentionLayer, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, 1)

    def forward(self, features, W_1, b_1, flag):
        if flag == 1:
            out_c = F.linear(features, W_1, b_1)
            out = out_c - out_c.max()
            out = out.exp()
            # print(out.shape)
            out = out.sum(1, keepdim=True)
            alpha = out / out.sum(0)
            alpha01 = features.size(0) * alpha.expand_as(features)
            context = torch.mul(features, alpha01)
        else:
            context = features
            alpha = torch.zeros(features.size(0), 1)

        return context, out_c, torch.squeeze(alpha)


class LossAttention(nn.Module):
    def __init__(self, ins_len, n_class=2):
        super(LossAttention, self).__init__()
        self.linear_1 = nn.Linear(ins_len, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.linear_3 = nn.Linear(128, 64)
        self.drop = nn.Dropout()
        self.linear = nn.Linear(64, n_class)
        self.att_layer = AttentionLayer(ins_len)

    def forward(self, bag, flag=1):
        bag = bag.float().reshape(-1, bag.shape[-1])
        bag_1 = self.drop(F.relu(self.linear_1(bag)))
        bag_2 = self.drop(F.relu(self.linear_2(bag_1)))
        bag_3 = self.drop(F.relu(self.linear_3(bag_2)))
        out, out_c, alpha = self.att_layer(bag_3, self.linear.weight, self.linear.bias, flag)
        out = out.mean(0, keepdim=True)

        Y_prob = self.linear(out)
        Y_hat = Y_prob.max(1)[1]
        return Y_prob, Y_hat, alpha


class MAMIL(nn.Module):
    def __init__(self, d, C=2, n_templates: int = 10, bottleneck_width: int = 4):
        """Initializes 2D MAMIL model.

        Args:
          n_templates: Number of templates.
          bottleneck_width: Bottleneck spatial width (and height), depends on input patches size.
        """
        super().__init__()
        self.C = C
        self.L = 128
        self.L2 = self.L * 2
        self.D = 128
        self.K = 1
        self.n_templates = n_templates
        self.bottleneck_width = bottleneck_width

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(d, self.L),
            nn.ReLU(),
        )

        self.neighbours_attention = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.Tanh()
        )

        self.templates = nn.Parameter(torch.randn(self.n_templates, self.L2,
                                                  requires_grad=True))
        self.proto_attention = nn.Sequential(
            nn.Linear(self.L2, self.L2),
            nn.Tanh()
        )
        self.global_attention = nn.Sequential(
            nn.Linear(self.L2, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L2 * self.K, self.C),
            nn.Sigmoid()
        )

    def forward(self, x):
        """"""
        # (n_i, L)
        x = x.reshape(-1, x.shape[-1])
        H = self.feature_extractor_part2(x)  # NxL

        neighbourhood_embeddings = []
        for i in range(H.shape[0]):
            if i == 0:
                if H.shape[0] == 1:
                    cur_neighbours = torch.tile(H[0], (2, 1))
                else:
                    cur_neighbours = torch.tile(H[i + 1], (2, 1))
            elif i == H.shape[0] - 1:
                cur_neighbours = torch.tile(H[i - 1], (2, 1))
            else:
                cur_neighbours = torch.vstack([H[i - 1], H[i + 1]])
            # 当前实例(1, L)
            cur_instance_embedding = H[i].reshape(1, -1)
            # 当前实例的邻居(2, L)
            # 与邻居的关系
            cur_alphas = torch.mm(
                self.neighbours_attention(cur_neighbours),
                cur_instance_embedding.T
            )  # 2x1
            cur_alphas = torch.transpose(cur_alphas, 1, 0)  # 1x2
            cur_alphas = F.softmax(cur_alphas, dim=1)  # 1x2
            cur_neighbourhood_emb = torch.mm(cur_alphas, cur_neighbours)  # 1xL
            neighbourhood_embeddings.append(cur_neighbourhood_emb)
        # 邻居嵌入
        neighbourhood_embeddings = torch.cat(neighbourhood_embeddings, dim=0)

        # 与H水平拼接 (N, 2L)
        H = torch.cat((H, neighbourhood_embeddings), dim=1)

        # 与多个模板的得分 (N, P)
        patch_scores = torch.mm(
            self.proto_attention(H),  # H,
            self.templates.T
        )
        patch_scores = torch.transpose(patch_scores, 1, 0)  # PxN
        betas = F.softmax(patch_scores, dim=1)  # PxN
        # (P, 2L)
        embs = torch.mm(betas, H)  # PxL2

        # 全局注意力 (P, K)
        template_scores = self.global_attention(embs)  # PxK
        # (K, P)
        template_scores = torch.transpose(template_scores, 1, 0)
        gammas = F.softmax(template_scores, dim=1)
        M = torch.mm(gammas, embs)  # KxL2
        # 注意力 (1, 4)
        A = torch.mm(gammas, betas)

        Y_prob = self.classifier(M)
        Y_hat = Y_prob.max(1)[1]

        return Y_prob, Y_hat, A


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, C=2, dropout_v=0.0, nonlinear=True):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.lin = nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU())
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.Tanh())
        else:
            self.lin = nn.Identity()
            self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, C, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        feats = self.lin(feats)
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0,
                                  descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0,
                                     index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0,
                                        1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
                      0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        Y_prob = C.view(1, -1)
        Y_hat = Y_prob.max(1)[1]

        return Y_prob, Y_hat, A


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        x = x.reshape(-1, x.shape[-1])
        feats, classes = self.i_classifier(x)
        Y_prob, Y_hat, A = self.b_classifier(feats, classes)

        return Y_prob, Y_hat, A


def main1():
    from MILFool.MIL import MIL
    data_path = "D:/OneDrive/Files/Code/Data/MIL/Image/tiger.mat"
    mil = MIL(data_path)
    from MILFool.BagLoader import BagGenerator
    bags = BagGenerator(mil.bag_space, mil.bag_lab)
    net = LossAttention(mil.d)

    # net.forward(bags[87][0])
    for batch_idx, (bag, label) in enumerate(bags):
        print(batch_idx)
        net.forward(bag)
        break


def main2():
    from MILFool.MIL import MIL
    data_path = "D:/OneDrive/Files/Code/Data/MIL/Image/tiger.mat"
    mil = MIL(data_path)
    from MILFool.BagLoader import BagGenerator
    bags = BagGenerator(mil.bag_space, mil.bag_lab)
    i_classifier = FCLayer(mil.d, 1)
    b_classifier = BClassifier(input_size=mil.d, output_class=1)
    net = MILNet(i_classifier, b_classifier)
    for batch_idx, (bag, label) in enumerate(bags):
        print(batch_idx)
        print(bag.shape)
        net.forward(bag)
        break


if __name__ == '__main__':
    main2()
