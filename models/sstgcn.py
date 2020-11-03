import torch
import torch.nn as nn

class SSGC(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True,
                 att=True,
                 gate=True,
                 n_head=4,
                 d_kc=0.25,
                 d_vc=0.25
                 ):
        super().__init__()

        self.kernel_size = kernel_size
        self.gate = gate

        self.bn = nn.BatchNorm2d(in_channels)

        if int(d_kc * in_channels) == 0:
            d_kc = 1
            d_vc = 1

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

        if att is True:
            self.att = MultiHeadSelfAttention(n_head, d_in=in_channels, d_out=out_channels, d_k=d_kc*out_channels, d_v=out_channels*d_vc, residual=True, res_fc=False)

        if gate is True:
            print('[Info] gate activated.')
            g = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
            self.register_parameter('g', g)

class Graph():
    ''' The Graph to model the 3D skeletal data

    Args:
        strategy: Select one of following adjacency strategies
        - unilabel
        - distance
        - spatial
        max_dis_connect: max connection distance
    '''
    def __init__(self, strategy='spatial', max_dis_connect=1):
        self.strategy = strategy
        self.max_dis_connect = max_dis_connect

        self.get_edge()
        self.get_adjacency()

    def get_edge(self):
        self.center = 2
        self.num_joint = 10
        self_connect = [(i, i) for i in range(self.num_joint)]
        neighbor_connect = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6),
                            (2, 7), (7, 8), (8, 9)]
        self.edge = self_connect + neighbor_connect

    def get_adjacency(self):
        adjacency = np.zeros((self.num_joint, self.num_joint))
        for i, j in self.edge:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
        dis_matrix = np.zeros((self.num_joint, self.num_joint)) + np.inf
        trans_matrix = [
            np.linalg.matrix_power(adjacency, p)
            for p in range(self.max_dis_connect + 1)
        ]
        N = np.zeros((self.num_joint, self.num_joint))
        for dis in range(self.max_dis_connect, -1, -1):
            dis_matrix[trans_matrix[dis] > 0] = dis
            N[trans_matrix[dis] > 0] = 1
        N = N / np.sum(N, 0)

        if self.strategy == 'unilabel':
            self.A = N[np.newaxis, :]
        elif self.strategy == 'distance':
            A = np.zeros(
                (self.max_dis_connect + 1, self.num_joint, self.num_joint))
            for dis in range(self.max_dis_connect + 1):
                A[dis][dis_matrix == dis] = N[dis_matrix == dis]
            self.A = A
        elif self.strategy == 'spatial':
            A = []
            for dis in range(self.max_dis_connect + 1):
                root = np.zeros((self.num_joint, self.num_joint))
                close = np.zeros((self.num_joint, self.num_joint))
                further = np.zeros((self.num_joint, self.num_joint))
                for i in range(self.num_joint):
                    for j in range(self.num_joint):
                        if dis_matrix[i, j] == dis:
                            if dis_matrix[i, self.center] == dis_matrix[
                                    j, self.center]:
                                root[i, j] = N[i, j]
                            elif dis_matrix[i, self.center] < dis_matrix[
                                    j, self.center]:
                                close[i, j] = N[i, j]
                            else:
                                further[i, j] = N[i, j]
                if dis == 0:
                    A.append(root)
                else:
                    A.append(root + close)
                    A.append(further)
            self.A = np.stack(A)
        else:
            raise ValueError('[Error] Strategy not existing.')