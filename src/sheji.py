import torch
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import pickle
import dgl
from dgl.nn import GATConv
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.dlutils import *
from src.constants import *
import pmdarima as pm
from joblib import Parallel, delayed
from statsmodels.tsa.seasonal import seasonal_decompose     # 做趋势分解用的库
from statsmodels.tsa.seasonal import STL

torch.manual_seed(1)

class MultiScaleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(MultiScaleCNN, self).__init__()
        self.channel = in_channels   #16     
        self.out_channels1 = 64
        self.kernel_size1 = 3           # 30
        
        self.out_channels2 = 64         # 3
        self.kernel_size2 = 33		
        
        self.out_channels3 = 64
        self.kernel_size3 = 53          # 3
        # self.convs = nn.ModuleList([
        #     nn.Conv1d(in_channels, out_channels, kernel_size=k, padding=k//2)
        #     for k in kernel_sizes
        # ])
        self.convs1 = nn.Sequential((
            nn.Conv1d(in_channels, self.out_channels1, kernel_size=self.kernel_size1, padding=self.kernel_size1//2)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Linear(50, 100))       
        self.convs2 = nn.Sequential((
            nn.Conv1d(in_channels, self.out_channels2, kernel_size=self.kernel_size2, padding=self.kernel_size2//2)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Linear(50, 100))
        self.convs3 = nn.Sequential((
            nn.Conv1d(in_channels, self.out_channels3, kernel_size=self.kernel_size3, padding=self.kernel_size3//2)),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Linear(50, 100))
        # Linear 层
        self.Linear1 = nn.Linear(self.out_channels1, out_channels)
        self.Linear2 = nn.Linear(self.out_channels2, out_channels)
        self.Linear3 = nn.Linear(self.out_channels3, out_channels)

    def forward(self, x):
        # x shape: (batch_size, seq_len, features)
        x = x.permute(1, 2, 0)  # 转置为 (batch_size, features, seq_len) 以适应1D卷积
        # conv_outputs = [conv(x) for conv in self.convs]  # 对不同卷积核的输出进行求和
        # outputs = [out.transpose(1, 2) for out in conv_outputs]  # 转置回 (batch_size, seq_len, features)
        
        conv_outputs1 = self.convs1(x)  # 对不同卷积核的输出进行求和
        # print('outputs1:', conv_outputs1.shape)      [128, 100, 32]
        conv_outputs1 = conv_outputs1.permute(2, 0, 1)
        outputs1 = self.Linear1(conv_outputs1)  # 转置回 (batch_size, seq_len, features)  
        # print('conv_outputs1:', outputs1.shape)  
            
        conv_outputs2 = self.convs2(x)  # 对不同卷积核的输出进行求和
        # print('outputs1:', conv_outputs2.shape)
        conv_outputs2 = conv_outputs2.permute(2, 0, 1)
        outputs2 = self.Linear1(conv_outputs2) 
        # print('conv_outputs2:', outputs1.shape)   
             
        conv_outputs3 = self.convs3(x)  # 对不同卷积核的输出进行求和
        # print('outputs1:', conv_outputs3.shape)
        conv_outputs3 = conv_outputs3.permute(2, 0, 1)
        outputs3 = self.Linear1(conv_outputs3)
        # print('conv_outputs3:', outputs1.shape) 
        return outputs1, outputs2, outputs3

class DlinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DlinearModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = x.permute(1, 2, 0)          # [batch_size, dim, windows]
        if isinstance(x, tuple): x = x[1]          # z:[1, 128, 12]
        # print(x.shape)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(2, 0, 1)
        # print('x2:', x.shape)        # [1, batch_size, dim]
        return x

class Dlinear_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Dlinear_Model, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x): 
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        # print('x2:', x.shape)        # [1, batch_size, dim]
        return x

class LSTMAnomalyDetectionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMAnomalyDetectionNet, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # LSTM 输出 [batch_size, hidden_dim]
        out = self.fc(out)  # 全连接层输出 [batch_size, output_dim]
        return out
    
class trans_basic(nn.Module):
	def __init__(self, feats):
		super(trans_basic, self).__init__()
		self.name = 'trans_basic'
		self.lr = lr
		self.batch = 32
		self.n_feats = feats
		self.nhead = 4
		self.n_window = 100
		self.n = self.n_feats * self.n_window
        # 确保 d_model 可以被 nhead 整除，确保 d_model = nhead * head_dim
		assert self.n_feats % self.nhead == 0, "feats should be divisible by nhead"
		self.head_dim = self.n_feats // self.nhead
            
		self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=feats, nhead=self.nhead, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 4)
		decoder_layers = TransformerDecoderLayer(d_model=feats, nhead=self.nhead, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder = TransformerDecoder(decoder_layers, 4)
		# self.fcn = nn.Sigmoid()
		self.fcn = nn.Tanh()

	def forward(self, src, tgt):
		src = src * math.sqrt(self.n_feats)             # 将张量 src 的每个元素乘以特征维度数的平方根。这种缩放通常用于调整输入的标准差，使其适合于后续的计算。
		src = self.pos_encoder(src)
		# print('src:', src.shape)                       # [100, 128, 12] 带有dropout的位置编码数据
		memory = self.transformer_encoder(src)
		# print('memory:', memory.shape)               # [100, 128, 12]
		x = self.transformer_decoder(tgt, memory)    # [1, 128, 12]
		# print('x输出1：', x.shape)                 
		x = self.fcn(x)                              # [1, 128, 12]
		# print('x输出:', x.shape)
		return x

# --------------------------------------
def fit_arima(series):
    # 处理 NaN 值，用均值填充
    if np.isnan(series).any():
        series = np.nan_to_num(series, nan=np.nanmean(series))
        
    if np.var(series) == 0:
        return series[0]  # 序列是常量，直接返回常量
    model = pm.auto_arima(series, start_p=1, start_q=1, max_p=3, max_q=3, information_criterion='bic', seasonal=False, stepwise=True, trace=False
                          , error_action='ignore', d=None, enforce_stationarity=False, enforce_invertibility=False)   # 自动选择差分阶数
    return model.predict(n_periods=1)[0]  # 预测下一个值

class arima(nn.Module):
    def __init__(self):
        super(arima, self).__init__()

    def forward(self, x):
        data = x.permute(1, 0, 2)      # [100, 128, 16]----> [128, 100, 16]
        batch_size, seq_len, num_features = data.shape

        # 打印数据形状
        # print('data:', data.shape)

        prediction = Parallel(n_jobs=-1)(delayed(fit_arima)(data[i, :, j].cpu().numpy()) for i in range(batch_size) for j in range(num_features))
        predictions = np.array(prediction).reshape(batch_size, num_features)
        # print('qqqxingzhaung:', predictions.shape)
        # print('qqqqq', predictions)
        return torch.tensor(predictions, dtype=torch.float32).to(x.device)

def qushifenjie(src):
    # 初始化用于保存所有列的趋势信息的张量
    trend_tensor = torch.zeros_like(src)
    resid_tensor = torch.zeros_like(src)
    # 对每一个特征列分别进行趋势分解
    for i in range(src.shape[2]):  # 遍历每一列特征
        for j in range(src.shape[1]):
            y_train = src[:, j, i].cpu().numpy()  # 提取出每一列特征的时间序列
            y_train = pd.Series(y_train)
            yy = STL(y_train, period = 2).fit()      # period确定一个周期的采样数
            # trend = y_train.rolling(window = 5).mean()          #先给5个窗口
            # result = seasonal_decompose(y_train, model='additive', period=2, two_sided=False)  #效果没有rolling好
            # trend = result.trend  # 提取趋势部分
            # resid = result.resid
            trend = np.nan_to_num((y_train-yy.seasonal))  # 处理 NaN 值（有可能是因为分解后的数据中出现了NaN）
            # resid = np.nan_to_num(resid)
            # 将趋势部分转换为张量并扩展维度
            trend_column = torch.tensor(trend, dtype=torch.float32).to(src.device)
            # resid_column = torch.tensor(resid, dtype=torch.float32).to(src.device)           
            trend_tensor[:, j, i] = trend_column  # 将趋势加入到 trend_tensor 中
            # resid_tensor[:, j, i] = resid_column
    # print(trend_tensor.shape)          # [window, batch_size, dim]
    return trend_tensor# ,  resid_tensor