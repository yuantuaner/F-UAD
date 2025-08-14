import numpy as np
from sklearn.metrics import ndcg_score
from src.constants import lm

#计算命中率，前 p% 的预测中成功识别异常的比例，即命中率。
def hit_att(ascore, labels, ps = [100, 150]):
	res = {}
	for p in ps:
		hit_score = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			a, l = np.argsort(a).tolist()[::-1], set(np.where(l == 1)[0])
			if l:
				size = round(p * len(l) / 100)
				a_p = set(a[:size])
				intersect = a_p.intersection(l)
				hit = len(intersect) / len(l)
				hit_score.append(hit)
		# print(hit_score.shape)
		# print(hit_score)
		res[f'Hit@{p}%'] = np.mean(hit_score)
	return res

# 前 p% 的预测中成功识别异常的 NDCG 分数，值越高越好，衡量排序任务中预测结果的相关性。
def ndcg(ascore, labels, ps = [100, 150]):
	res = {}
	for p in ps:
		ndcg_scores = []
		for i in range(ascore.shape[0]):
			a, l = ascore[i], labels[i]
			labs = list(np.where(l == 1)[0])
			if labs:
				k_p = round(p * len(labs) / 100)
				try:
					hit = ndcg_score(l.reshape(1, -1), a.reshape(1, -1), k = k_p)
				except Exception as e:
					return {}
				ndcg_scores.append(hit)
		# print(ndcg_score.shape)
		# print(ndcg_scores)
		res[f'NDCG@{p}%'] = np.mean(ndcg_scores)
	return res



