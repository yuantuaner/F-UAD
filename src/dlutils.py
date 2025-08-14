import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
import copy
# from flash_attn.flash_attention import FlashAttention

# class flash_MultiHeadAttention(nn.Module):
#     def __int__(self, d_model, heads, dropout=0.1):
#         super().__init__()
#         self.d_model = d_model
#         self.d_k = d_model // heads
#         self.h = heads         

#         #初始化线性层
#         self.q_linear = nn.Linear(d_model, d_model)
#         self.k_linear = nn.Linear(d_model, d_model)
#         self.v_linear = nn.Linear(d_model, d_model)

#         self.dropout = nn.Dropout(dropout)

#         self.out = nn.Linear(d_model, d_model)

#     def forward(self, q, k, v, mask=None):
#         batch_size = q.size(0)

#         # 对q, k, v 进行线性变化
#         Q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
#         K = self.q_linear(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
#         V = self.q_linear(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)  

#         scores = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)       

#         ## mask
#         if mask is not None:
#             ## mask:  [N, T_k] --> [h, N, T_q, T_k]
#             mask = mask.unsqueeze(1).unsqueeze(0).repeat(self.num_heads, 1, Q.shape[2], 1)
#             scores = scores.masked_fill(mask, -np.inf)
#         scores = F.softmax(scores, dim=3)
 
# #         ## out = score * V
# #         out = torch.matmul(scores, V)  # [h, N, T_q, num_units/h]
# #         out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
 
#         return out, scores
    
class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)         # 创建一个形状为 [max_len, d_model] 的零张量 pe，即 [100, 12] 的零张量，用于存储位置编码。
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # torch.arange(0, max_len, dtype=torch.float) 生成从 0 到 99 的浮点数序列。
        # unsqueeze(1) 将形状从 [100] 改变为 [100, 1]，以便后续广播操作。

        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        # torch.arange(0, d_model).float() 生成从 0 到 d_model-1（即 11）的浮点数序列。
        # -math.log(10000.0) / d_model 计算一个缩放常数，将位置索引缩放到不同频率。
        # torch.exp(...) 计算上述序列的指数。

        # 通过广播得到形状为 [100, 12] 的张量，每个元素表示位置索引和频率缩放的乘积。
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        
        # pe.unsqueeze(0) 将 pe 的形状从 [100, 12] 改为 [1, 100, dim]，增加一个批次维度。
        # transpose(0, 1) 将形状从 [1, 100, 12] 变为 [100, 1, dim]，这样后续计算时可以方便地与输入数据进行广播操作。
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        #注册为模型的缓冲区（不作为参数进行优化）
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos+x.size(0), :]     #  self.pe的维度是[1, 128, dim], x的维度是[100, 128, dim]
        return self.dropout(x)           # 通过在位置编码后使用 Dropout，可以有效防止模型过拟合，提高模型在训练数据和测试数据上的泛化能力。这种方法尤其适用于处理高维序列数据的模型，
##---------------------------------------------flash attention--------------------------------------------------------
#使用flash attention
# class TransformerEncoderLayer_flashatt(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
#         super(TransformerEncoderLayer_flashatt, self).__init__()
#         self.self_attn = flash_MultiHeadAttention(d_model, nhead, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)

#         self.activation = nn.LeakyReLU(True)

#     def forward(self, src, src_mask=None, src_key_padding_mask=None):
#         src2 = self.self_attn(src, src, src)[0]
#         src = src + self.dropout1(src2)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         return src

# class TransformerDecoderLayer_flashatt(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
#         super(TransformerDecoderLayer_flashatt, self).__init__()
#         self.self_attn = flash_MultiHeadAttention(d_model, nhead, dropout=dropout)
#         self.multihead_attn = flash_MultiHeadAttention(d_model, nhead, dropout=dropout)
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)

#         self.activation = nn.LeakyReLU(True)

#     def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
#         tgt2 = self.self_attn(tgt, tgt, tgt)[0]
#         tgt = tgt + self.dropout1(tgt2)
#         tgt2 = self.multihead_attn(tgt, memory, memory)[0]
#         tgt = tgt + self.dropout2(tgt2)
#         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout3(tgt2)
#         return tgt

##---------------------------------------------多头自注意力机制--------------------------------------------------------
### 使用多头自注意力机制
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, is_causal=False, src_key_padding_mask=None):
        # print('src:', src.shape)
        src2, src_weights = self.self_attn(src, src, src)               # src:形状为 [seq_len, batch_size, embed_dim], src_weights:注意力权重矩阵[batch_size, num_heads, seq_len, seq_len]
        # print("src1:", src2.shape)
        # print("tgt_weight1", src_weights.shape)       
        src = src + self.dropout1(src2)
        # print("src_:", src.shape)        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.linear2(src) + self.dropout2(src2)
        # print('src_encoder:', src.shape)
        return src
    
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # print(tgt.dtype)  # 检查输入的张量类型
        # QK = tgt[:, :, -5:]  # 使用 tgt 的最后 5 列作为 Q 和 K
        # V = memory[:, :, :7]  # 使用 memory 的前 7 列作为 V       
        tgt2, tgt_weights = self.self_attn(tgt, tgt, tgt)   
        # print("tgt1:", tgt2.shape)   # [1, 128, 16]
        # print("tgt_weight1", tgt_weights.shape)   #[128, 1, 1]

        tgt = tgt + self.dropout1(tgt2)
        tgt2, tgt_weights2 = self.multihead_attn(tgt, memory, memory)
        # print("tgt2:", tgt2.shape)          #[1, 128, 16]
        # print("tgt_weight2", tgt_weights2.shape)         #[128, 1, 100]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class ComputeLoss:
    def __init__(self, model, lambda_energy, lambda_cov, device, n_gmm):
        self.model = model
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.device = device
        self.n_gmm = n_gmm
    
    def forward(self, x, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = torch.mean((x-x_hat).pow(2))

        sample_energy, cov_diag = self.compute_energy(z, gamma)

        loss = reconst_loss + self.lambda_energy * sample_energy + self.lambda_cov * cov_diag
        return Variable(loss, requires_grad=True)
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = (z.unsqueeze(1)- mu.unsqueeze(0))

        eps = 1e-12
        cov_inverse = []
        det_cov = []
        cov_diag = 0
        for k in range(self.n_gmm):
            cov_k = cov[k] + (torch.eye(cov[k].size(-1))*eps).to(self.device)
            cov_inverse.append(torch.inverse(cov_k).unsqueeze(0))
            det_cov.append((Cholesky.apply(cov_k.cpu() * (2*np.pi)).diag().prod()).unsqueeze(0))
            cov_diag += torch.sum(1 / cov_k.diag())
        
        cov_inverse = torch.cat(cov_inverse, dim=0)
        det_cov = torch.cat(det_cov).to(self.device)

        E_z = -0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = -torch.log(torch.sum(phi.unsqueeze(0)*E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1) + eps)
        if sample_mean==True:
            E_z = torch.mean(E_z)            
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = torch.sum(gamma, dim=0)/gamma.size(0) 

        #mu = KxD
        mu = torch.sum(z.unsqueeze(1) * gamma.unsqueeze(-1), dim=0)
        mu /= torch.sum(gamma, dim=0).unsqueeze(-1)

        z_mu = (z.unsqueeze(1) - mu.unsqueeze(0))
        z_mu_z_mu_t = z_mu.unsqueeze(-1) * z_mu.unsqueeze(-2)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov /= torch.sum(gamma, dim=0).unsqueeze(-1).unsqueeze(-1)

        return phi, mu, cov
        
class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s

