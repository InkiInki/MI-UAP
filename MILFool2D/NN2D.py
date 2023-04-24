import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, C=2, d=50*4*4, num_channel=1):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.d = d

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(num_channel, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.d, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, C),
            nn.Sigmoid(),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.d)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = Y_prob.max(1)[1]

        return Y_prob, Y_hat, A


class GatedAttention(nn.Module):
    def __init__(self, C=2, d=50*4*4, num_channel=1):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.d = d

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(num_channel, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.d, self.L),
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
            nn.Linear(self.L*self.K, C),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.d)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = Y_prob.max(1)[1]
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # # AUXILIARY METHODS
    # def calculate_classification_error(self, X, Y):
    #     Y = Y.float()
    #     _, Y_hat, _ = self.forward(X)
    #     error = 1. - Y_hat.eq(Y).cpu().float().mean().item()
    #
    #     return error, Y_hat
    #
    # def calculate_objective(self, X, Y):
    #     Y = Y.float()
    #     Y_prob, _, A = self.forward(X)
    #     Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
    #     neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli
    #
    #     return neg_log_likelihood, A


def _is_neighbour(a, b, distance: int = 1):
    return 0 < max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= distance


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
    def __init__(self, n_class=2, d=50*4*4, num_channel=1):
        super(LossAttention, self).__init__()

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(num_channel, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_len = d

        self.linear_1 = nn.Linear(self.feature_len, 256)
        self.linear_2 = nn.Linear(256, 128)
        self.linear_3 = nn.Linear(128, 64)
        self.drop = nn.Dropout()
        self.linear = nn.Linear(64, n_class)
        self.att_layer = AttentionLayer(self.feature_len)

    def forward(self, bag, flag=1):
        bag = bag.squeeze(0)
        bag = bag.float()
        bag = self.feature_extractor_part1(bag)
        bag = bag.view(-1, self.feature_len)
        bag_1 = self.drop(F.relu(self.linear_1(bag)))
        bag_2 = self.drop(F.relu(self.linear_2(bag_1)))
        bag_3 = self.drop(F.relu(self.linear_3(bag_2)))
        out, out_c, alpha = self.att_layer(bag_3, self.linear.weight, self.linear.bias, flag)
        out = out.mean(0, keepdim=True)

        Y_prob = self.linear(out)
        Y_hat = Y_prob.max(1)[1]

        return Y_prob, Y_hat, alpha


class MAMIL(nn.Module):
    def __init__(self, C: int = 2,
                 d: int = 50 * 4 * 4,
                 num_channel: int = 1,
                 n_templates: int = 10,
                 use_neighbourhood: bool = False,
                 bottleneck_width: int = 6):
        """Initializes 2D MAMIL model.

        Args:
          n_templates: Number of templates.
          use_neighbourhood: Use neighbourhood attention.
          bottleneck_width: Bottleneck spatial width (and height), depends on input patches size.
        """
        super().__init__()
        self.C = C
        self.n_templates = n_templates
        self.use_neighbourhood = use_neighbourhood
        self.L = 512
        if self.use_neighbourhood:
            self.L2 = self.L * 2
        else:
            self.L2 = self.L
        self.D = 128
        self.K = 1
        self.bottleneck_width = bottleneck_width
        self.channels = 48
        # self.embedding_dim = self.channels * self.bottleneck_width ** 2
        self.embedding_dim = d

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(num_channel, 36, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, self.channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.embedding_dim, self.L),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.L, self.L),
            nn.ReLU(inplace=True),
            nn.Dropout(),
        )

        # self.attention = nn.Sequential(
        #     nn.Linear(self.L, self.D),
        #     nn.Tanh(),
        #     nn.Linear(self.D, self.K)
        # )

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

    def forward(self, x, positions=None, temperature=1.0):
        """Calculates instance-level probabilities and an attention matrix.

        Args:
          x: Input batch with exactly one bag of batches.
          positions: Patches positions.
          temperature: Softmax temperature.

        Returns:
          Tuple of (instance probabilities, instance labels, attention matrix)
        """
        x = x.squeeze(0)
        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.embedding_dim)
        H = self.feature_extractor_part2(H)  # NxL

        # Classical Attention-MIL:
        # A = self.attention(H)  # NxK
        # A = torch.transpose(A, 1, 0)  # KxN
        # A = F.softmax(A, dim=1)  # softmax over N
        # M = torch.mm(A, H)  # KxL

        # Neighbourhood attention:
        #   H shape: NxL
        #   Naive loop-based implementation:
        if self.use_neighbourhood:
            assert positions is not None

            neighbourhood_embeddings = []
            # for i in range(H.shape[0]):
            assert len(positions) == H.shape[0]
            for i, pos in enumerate(positions):
                nbrs = [j for j, jpos in enumerate(positions) if j != i and _is_neighbour(jpos, pos)]
                if len(nbrs) > 0:
                    cur_neighbours = torch.cat([H[j:j + 1] for j in nbrs], axis=0)
                else:
                    cur_neighbours = H[i: i + 1]
                cur_instance_embedding = H[i: i + 1]
                # cur_neighbours shape: 2xL
                cur_alphas = torch.mm(
                    self.neighbours_attention(cur_neighbours),
                    cur_instance_embedding.T
                )  # 2x1
                cur_alphas = torch.transpose(cur_alphas, 1, 0)  # 1x2
                cur_alphas = F.softmax(cur_alphas, dim=1)  # 1x2
                cur_neighbourhood_emb = torch.mm(cur_alphas, cur_neighbours)  # 1xL
                neighbourhood_embeddings.append(cur_neighbourhood_emb)
            # raise Exception()
            neighbourhood_embeddings = torch.cat(neighbourhood_embeddings, dim=0)
            H = torch.cat((H, neighbourhood_embeddings), dim=1)  # Nx2L

        # Multi-template:
        betas = torch.mm(
            self.proto_attention(H),  # H,
            self.templates.T
        )  # NxP, P = n_templates
        betas = torch.transpose(betas, 1, 0)  # PxN
        betas = F.softmax(betas / temperature, dim=1)  # PxN
        embs = torch.mm(betas, H)  # PxL2

        gammas = self.global_attention(embs)  # PxK
        gammas = torch.transpose(gammas, 1, 0)  # KxP
        gammas = F.softmax(gammas / temperature, dim=1)  # KxP
        M = torch.mm(gammas, embs)  # KxL2

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

        self.input_size = input_size
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
    def __init__(self, d, i_classifier, b_classifier, num_channel=1):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(num_channel, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.d = d

    def forward(self, x):

        x = x.squeeze(0)
        x = self.feature_extractor_part1(x)
        x = x.view(-1, self.d)
        feats, classes = self.i_classifier(x)

        Y_prob, Y_hat, A = self.b_classifier(feats, classes)

        return Y_prob, Y_hat, A
