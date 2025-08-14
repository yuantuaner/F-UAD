import torch
import torch.nn.functional as F

def compute_mmd(x, y, kernel='rbf'):
    """
    计算 x 和 y 之间的 MMD。x 和 y 是两个来自不同分布的样本集。
    
    参数:
    x: tensor, 样本集 X (batch_size, feature_dim)
    y: tensor, 样本集 Y (batch_size, feature_dim)
    kernel: str, 核函数类型, 支持 'rbf'
    
    返回:
    mmd_value: tensor, MMD 距离
    """
    def rbf_kernel(x1, x2, sigma=1.0):
        # 高斯 RBF 核函数
        gamma = 1.0 / (2.0 * sigma**2)
        diff = x1.unsqueeze(1) - x2.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        return torch.exp(-gamma * dist_sq)
    
    if kernel == 'rbf':
        # 样本之间的核相似度
        K_xx = rbf_kernel(x, x)
        K_yy = rbf_kernel(y, y)
        K_xy = rbf_kernel(x, y)
        
        # 计算 MMD
        mmd_value = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return mmd_value
    else:
        raise ValueError("目前仅支持 'rbf' 核")