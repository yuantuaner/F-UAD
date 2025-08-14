import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
import math
torch.manual_seed(1)
from src.sheji import *  # 导入 MultiScaleCNN 类
import pmdarima as pm
from src.weight_init import * 

class TranAD_classi(nn.Module):
    def __init__(self, feats):
        super(TranAD_classi, self).__init__()
        self.name = 'TranAD_classi'
        self.lr = lr
        self.batch = 512
        self.n_feats = feats
        self.nhead = 2
        self.n_window = 100
        self.n = self.n_feats * self.n_window	
        self.n_encoder_outputs = feats//2
        self.len_kernel = 3  # 卷积核的多少
		# encoder部分
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=self.nhead, dim_feedforward=feats, dropout=0.5)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 4)		  	
	    
        self.dlinear = DlinearModel(input_dim=self.n_window, hidden_dim=4*feats, output_dim=1)
		# CNN卷积部分
        self.multi_scale_cnn = MultiScaleCNN(in_channels=feats, out_channels=feats)
        # 多个decoder
        self.transformer_decoder1 = TransformerDecoder(TransformerDecoderLayer(
		   d_model=feats, nhead=self.nhead, dim_feedforward=8*feats, dropout=0.1), 4)
        self.transformer_decoder2 = TransformerDecoder(TransformerDecoderLayer(
		   d_model=feats, nhead=self.nhead, dim_feedforward=8*feats, dropout=0.1), 4)
        self.transformer_decoder3 = TransformerDecoder(TransformerDecoderLayer(
		   d_model=feats, nhead=self.nhead, dim_feedforward=8*feats, dropout=0.1), 4)				
        self.fcn = nn.Tanh()
        self.fc1 = nn.Sequential(nn.Linear(feats * self.len_kernel, feats),
            nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(feats*2, feats),
                                nn.Tanh())
        self.apply(weight_init)

    def forward(self, src, tgt):
	
        src = src * math.sqrt(self.n_feats)             # 将张量 src 的每个元素乘以特征维度数的平方根。这种缩放通常用于调整输入的标准差，使其适合于后续的计算。


        src = self.pos_encoder(src)

        qushi_output = qushifenjie(src)

        dlinear_output = self.dlinear(qushi_output)
        memory = self.transformer_encoder(src)
        multi_scale_features1, multi_scale_features2, multi_scale_features3 = self.multi_scale_cnn(memory)
        transformer_outputs1 = self.transformer_decoder1(tgt, multi_scale_features1)
        transformer_outputs2 = self.transformer_decoder2(tgt, multi_scale_features2)	
        transformer_outputs3 = self.transformer_decoder3(tgt, multi_scale_features3)      
        fused_output = torch.cat((transformer_outputs1, transformer_outputs2, transformer_outputs3), dim=2)  # 在特征维度上拼接
        prediction = self.fc1(fused_output)
		#将 Transformer 的输出与 ARIMA 的输出拼接
        combined_output = torch.cat((prediction[0,:,:], dlinear_output[0,:,:]), dim=1)

        # 通过全连接层输出最终结果
        output = self.fc2(combined_output)
        return output

