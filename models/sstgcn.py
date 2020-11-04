#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
#%%
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
                 attbranch=True,
                 gate=True,
                 n_head=4,
                 d_kc=0.25,
                 d_vc=0.25
                 ):
        super().__init__()

        self.kernel_size = kernel_size
        self.attbranch = attbranch
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

        if attbranch is True:
            self.att = SelfAttentionBranch(n_head, d_in=in_channels, d_out=out_channels, d_k=int(d_kc*out_channels), d_v=int(out_channels*d_vc), residual=True, res_fc=False)

        if gate is True:
            print('[Info] gate activated.')
            g = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
            self.register_parameter('g', g)

    def forward(self, x, A):
        inp = self.bn(x)

        f_c = self.conv(inp)

        N, KC, T, V = f_c.size()
        f_c = f_c.view(N, self.kernel_size, KC//self.kernel_size, T, V)
        f_c = torch.einsum('nkctv,kvw->nctw', (f_c, A))

        if self.attbranch:
            N, C, T, V = inp.size()
            f_a = inp.permute(0, 2, 3, 1).contiguous().view(N*T, V, C)
            f_a, _ = self.att(f_a, f_a, f_a)
            f_a = f_a.view(N, T, V, -1).permute(0, 3, 1, 2) # N, C, T, V

            if self.gate:
                f = (f_a*self.g+f_c)/2
            else:
                f = (f_a+f_c)/2
        else:
            f = f_c

        return f.contiguous(), A

class SelfAttentionBranch(nn.Module):
    def __init__(self, n_head, d_in, d_out, d_k, d_v, residual=True, res_fc=False, dropout=0.1, att_dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_in = d_in
        self.d_out = d_out
        self.d_k = d_k
        self.d_v = d_v

        self.residual = residual
        self.res_fc = res_fc

        self.w_q = nn.Linear(d_in, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_in, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_in, n_head * d_v, bias=False)

        self.fc = nn.Linear(n_head * d_v, d_out, bias=False)

        self.attention = ScaledAttention(temperature=d_k ** 0.5)

        if residual:
            self.res = nn.Linear(d_in, d_out) if res_fc or (d_in != d_out) else lambda x: x

        self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.att_drop = nn.Dropout(att_dropout)

    def forward(self, q, k, v):

        assert self.d_in == v.size(2)

        # sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        NT, V, C = v.size()

        if self.residual:
            res = self.res(v)

        # q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_q(q).view(NT, V, self.n_head, self.d_k)
        k = self.w_k(k).view(NT, V, self.n_head, self.d_k)
        v = self.w_v(v).view(NT, V, self.n_head, self.d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        att = torch.matmul(q, k.transpose(2, 3))/(self.d_k**0.5)
        att = self.att_drop(F.softmax(att,dim=3))

        x = torch.matmul(att, v) # NT, H, V, D_v

        x = x.transpose(1,2).view(NT,V,-1)
        x = self.dropout(self.fc(x))

        if self.residual:
            x += res

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)

        return x, att

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