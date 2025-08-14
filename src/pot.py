import os
import numpy as np
from src.spot import SPOT
from src.constants import *
from sklearn.metrics import *
from src.models import *
from scipy.stats import genextreme as gev  # 用于极值理论

#使用IIR对数据进行滤波
def iir_filter(data, alpha=0.96):
    """
    使用IIR滤波器对输入数据进行平滑
    :param data: 输入的重建损失序列，类型为 numpy array
    :param alpha: 滤波系数，控制平滑程度 (0 < alpha < 1)
    :return: 平滑后的序列
    """
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]  # 初始化第一个值

    # 应用IIR滤波器
    for i in range(1, len(data)):
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]
    
    return smoothed_data

#进行动态阈值计算
def adaptive_threshold(train_scores, test_scores, alpha=3):
    """
    自适应阈值计算函数，基于滑动窗口内的得分动态计算阈值。
    Args:
        scores (np.ndarray): 输入的分数序列
        window_size (int): 滑动窗口大小，用于计算自适应阈值
        alpha (float): 乘以标准差的系数，决定异常点的敏感度
    Returns:
        thresholds (np.ndarray): 每个时间点的动态阈值
    """
    #计算静态阈值
    mean_score = np.mean(train_scores)
    std_dev = np.std(train_scores)
    static_thro = (mean_score + 3 * std_dev)
    # # 计算阈值
    window_size = 10
    thresholds = np.zeros_like(test_scores)
    for i in range(window_size, len(test_scores)):
        window_data = test_scores[i - window_size:i]
        mean_val = np.mean(window_data)
        std_val = np.std(window_data)
        thresholds[i] = 0.96*(mean_val + std_val) + 0.04*static_thro  
    thresholds[:window_size] = thresholds[window_size]  
    return thresholds

def calc_point2point(predict, actual):
    """
    calculate f1 score by predict and actual.
    Args:
        predict (np.ndarray): the predict label
        actual (np.ndarray): np.ndarray
    """
    act = (actual != 0).astype(int)      

    predict = predict.astype(int)       

    TP = np.sum(predict * act)
    TN = np.sum((1 - predict) * (1 - act))
    FP = np.sum(predict * (1 - act))
    FN = np.sum((1 - predict) * act)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    Acc = (TP + TN) / (TP + TN + FP + FN)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    try:
        roc_auc = roc_auc_score(act, predict)
    except:
        roc_auc = 0
    return f1, precision, recall, Acc, TP, TN, FP, FN, roc_auc


# the below function is taken from OmniAnomaly code base directly
def adjust_predicts(score, label,
                    threshold=None,
                    # pred=0,
                    pred=None,
                    calc_latency=False):
    """
    Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.
    Args:
        score (np.ndarray): The anomaly score
        label (np.ndarray): The ground-truth label
        threshold (float): The threshold of anomaly score.
            A point is labeled as "anomaly" if its score is lower than the threshold.
        pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,
        calc_latency (bool):
    Returns:
        np.ndarray: predict labels
    """

    # 判断训练集的异常分数和标签的长度是否一样
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)

    latency = 0               # 检测到异常和实际异常发生之间的时间差
    predict = score > threshold

    return predict, latency


def calc_seq(score, label, threshold, calc_latency=False):
    """
    Calculate f1 score for a score sequence
    """
    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label)


def bf_search(score, label, start, end=None, step_num=1, display_freq=1, verbose=True):
    """
    Find the best-f1 score by searching best `threshold` in [`start`, `end`).
    Returns:
        list: list for results
        float: the `threshold` for best-f1
    """
    if step_num is None or end is None:
        end = start
        step_num = 1
    search_step, search_range, search_lower_bound = step_num, end - start, start
    if verbose:
        print("search range: ", search_lower_bound, search_lower_bound + search_range)
    threshold = search_lower_bound
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(search_step):
        threshold += search_range / float(search_step)
        target = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
        if verbose and i % display_freq == 0:
            print("cur thr: ", threshold, target, m, m_t)
    print(m, m_t)
    return m, m_t


def eval(init_score, score, label, model, name, name_special, MM = False, q=1e-5, level=0.02, weidu = False):
    # init_score训练集损失, score测试集损失
    result_savepath = os.path.join(
        "/home/jupyter-yyf-yyy/program/keyan/Anomaly-Detection-for-UAV/MLTD/save_dataresult",
        name, name_special, "test_materic"
    )

    # 确保路径存在
    os.makedirs(result_savepath, exist_ok=True)
    """
    Run POT method on given score.
    Args:
        init_score (np.ndarray): The data to get init threshold.训练的损失
            it should be the anomaly score of train set.
        score (np.ndarray): The data to run POT method.    测试的损失
            it should be the anomaly score of test set.
        label:
        q (float): Detection level (risk)     标签
        level (float): Probability associated with the initial threshold t
    Returns:
        dict: pot result dict
    """

    # 使用POT方法
    lms = lm[0]          

    ###使用SPOT放大计算阈值
    while True:          
        try:              
            s = SPOT(q)  
            s.fit(init_score, score)  
            s.initialize(level=lms, min_extrema=False, verbose=False)  
        except: lms = lms * 0.999
        else: break
    ret = s.run(dynamic=False)  # run

    pot_th = np.mean(ret['thresholds']) * lm[1]           # 调整计算出来的阈值


    pred, _ = adjust_predicts(score, label, pot_th, calc_latency=True)

    p_t = calc_point2point(pred, label)

    result = {
        'f1': p_t[0],
        'precision': p_t[1],
        'recall': p_t[2],
        'Acc': p_t[3],
        'TP': p_t[4],
        'TN': p_t[5],
        'FP': p_t[6],
        'FN': p_t[7],
        'ROC/AUC': p_t[8],
        'threshold': pot_th
    } 

    if MM == False:
        # **保存到指定路径**
        result_savepath = os.path.join(
            "save_dataresult",
            name, name_special, "test_materic" #TODO
        )
        os.makedirs(result_savepath, exist_ok=True)
        save_file = os.path.join(result_savepath, "12_evaluation_results.txt")
        with open((save_file), "a") as f:  # "a" 代表 append 模式，每次写入不会覆盖

            f.write(f"\n==== Evaluation Results for {weidu} ====\n")  # 使用数据集名作为标题
            for key, value in result.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")  # 添加换行以便分割不同循环的结果
    else :
        result_savepath = os.path.join(
            "save_dataresult",
            name, name_special, "test_materic" #TODO
        )
        os.makedirs(result_savepath, exist_ok=True)
        save_file = os.path.join(result_savepath, "final_evaluation_results.txt")
        with open(save_file, "w") as f:  # "w" 代表 write 模式，覆盖已有内容
            f.write(f"\n==== Evaluation Results ====\n")  # 使用数据集名作为标题
            for key, value in result.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")  # 添加换行以便格式清晰       

    return result, score, np.array(pred)