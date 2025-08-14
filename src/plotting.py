import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statistics
import os, torch
import numpy as np
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(plt.style.available)
# plt.style.use(['classic', 'dark_background'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs('plots', exist_ok=True)

def smooth(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def plotter_final(name, name_special, re, scfinal, act):
	# 生成保存路径
    save_path = os.path.join('save_dataresult', name, name_special, 'test_data', 'final')
	# 确保目录存在，如果不存在则创建
    os.makedirs(save_path, exist_ok=True)

	# 保存文件
    np.save(os.path.join(save_path, f'yuzhi.npy'), re)
    np.save(os.path.join(save_path, f'score.npy'), scfinal)
    np.save(os.path.join(save_path, 'act.npy'), act[:, 0])

    for dim in range(re.shape[1]):
        re_dim, score, l = re[:, dim], scfinal[:, dim], act[:, dim]
        y_max = max(re_dim.max(), score.max()) * 1.1  # 稍微扩大上限	
        fig, ax = plt.subplots()
        ax.set_facecolor('white')
        ax.set_ylabel('Value')
        ax.set_title(f'final')
        # 设置动态的纵轴范围
        ax.set_ylim(0, y_max)
        ax.plot(smooth(score), linewidth=0.5, color='r', label='Anomaly Score')
        ax.plot(smooth(re_dim), linewidth=1.0, color ='blue', label='Reconstructed')
        ax.plot(l, '--', linewidth=0.3, alpha=0.5, label='labels')
        ax.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Anomaly Score')
        ax.legend()

        # Save the figure as .png
        plt.savefig(f'plots/{name}/test/{name_special}/final_result.png', format='png', dpi=600)
        plt.close()

# def plotter_final(name, re, scfinal, act):
# 	os.makedirs(os.path.join('plots', name), exist_ok=True)
# 	pdf = PdfPages(f'plots/{name}/test/final_rudder01.pdf')
# 	# plt.rcParams['figure.facecolor'] = 'white'
# 	for dim in range(re.shape[1]):
# 		re, score, l = re[:, dim], scfinal[:, dim], act[:, dim]
# 		fig, ax = plt.subplots()# facecolor='w')
# 		# ax.set_facecolor('white')
# 		# ax.set_facecolor("white")
# 		ax.set_ylabel('Value')
# 		ax.set_title(f'final')
# 		ax.set_facecolor("white")
# 		ax.plot(smooth(score), linewidth=0.5, color='r')
# 		ax.plot(smooth(re), linewidth=1.0, color ='blue')
# 		ax.plot(l, '--', linewidth=0.3, alpha=0.5)
# 		ax.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
# 		ax.set_xlabel('Timestamp')
# 		ax.set_ylabel('Anomaly Score')
# 		# ax4.set_facecolor("gray")
# 		# ax4.plot(smooth(thre), linewidth=0.7, color='c')
# 		# ax4.set_xlabel('Timestamp')
# 		# ax4.set_ylabel('thresholds')
# 		pdf.savefig(fig)
# 		plt.close()
# 	pdf.close()

#画出picture
def plotter_picture(name, name_special, y_true, y_pred, ascore, labels, thre):
    # if 'TranAD' in name:
    #     y_true = torch.roll(y_true, 1, 0)
    
	# 生成保存路径
    save_path = os.path.join('save_dataresult', name, name_special, 'test_data', 'weidu')

	# 确保目录存在，如果不存在则创建
    os.makedirs(save_path, exist_ok=True)

	# 保存文件
    np.save(os.path.join(save_path, 'y_true.npy'), y_true)
    np.save(os.path.join(save_path, f'y_pred{name}.npy'), y_pred)
    np.save(os.path.join(save_path, f'score_f{name}.npy'), ascore)
    np.save(os.path.join(save_path, 'labels.npy'), labels[0])
    np.save(os.path.join(save_path, f'thre{name}.npy'), thre)
    for dim in range(y_true.shape[1]):
        y_t, y_p, l, a_s, thr = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim], thre[:, dim]
        
        # 第一张图：显示 y_true 和 y_pred
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 6), facecolor='w')
        fig.set_facecolor('white')
        
        ax1.set_facecolor("white")
        ax1.set_ylabel('Value')
        ax1.set_title(f'Dimension = {dim} (True vs Predicted)')
        
        ax1.plot(smooth(y_t), linewidth=0.6, label='True', color='y')
        ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.5, label='Predicted', color='g')
        
        ax3 = ax1.twinx()
        ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
        ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
        
        if dim == 0:
            ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
        
        # 保存第一张图 (True vs Predicted)
	    # 生成保存路径
        save_fig = os.path.join('plots', name, 'test', name_special, '无拼接', 'output')
	    # 确保目录存在，如果不存在则创建
        os.makedirs(save_fig, exist_ok=True)
        plt.savefig(f'plots/{name}/test/{name_special}/无拼接/output/output_dim{dim}.png', format='png', dpi=600)
        plt.close()

        # 第二张图：显示 ascore 和 thre
        fig, ax2 = plt.subplots(1, 1, figsize=(12, 6), facecolor='w')
        fig.set_facecolor('white')
        
        ax2.set_facecolor("white")
        ax2.plot(smooth(a_s), linewidth=0.5, color='r', label='Anomaly Score')
        ax2.plot(smooth(thr), linewidth=1.0, color='blue', label='Threshold')
        ax2.set_xlabel('Timestamp')
        ax2.set_ylabel('Anomaly Score')
        ax2.set_title(f'Dimension = {dim} (Anomaly Score vs Threshold)')
        ax2.legend()

        # 保存第二张图 (Anomaly Score vs Threshold)
        plt.savefig(f'plots/{name}/test/{name_special}/无拼接/output/anomaly_score_dim{dim}.png', format='png', dpi=600)
        plt.close()

# 画出pdf
def plotter_pdf(name, name_special, y_true, y_pred, ascore, labels, thre):
	if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
	os.makedirs(os.path.join('plots', name, 'test', name_special, '无拼接', 'output'), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/test/{name_special}/无拼接/output/all.pdf')
	for dim in range(y_true.shape[1]):
		y_t, y_p, l, a_s, thr = y_true[:, dim], y_pred[:, dim], labels[:, dim], ascore[:, dim], thre[:, dim]
		y_max = max(a_s.max(), thr.max()) * 1.1  # 稍微扩大上限
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,9), sharex=True, facecolor='w')
		fig.set_facecolor('white')
		ax1.set_facecolor("white")
		ax1.set_ylabel('Value')
		ax1.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax1.plot(smooth(y_t), linewidth=0.6, label=y_t, color='y')
		ax1.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.5, label=y_p, color='g')
		ax3 = ax1.twinx()
		ax3.plot(l, '--', linewidth=0.3, alpha=0.5)
		ax3.fill_between(np.arange(l.shape[0]), l, color='blue', alpha=0.3)
		if dim == 0: ax1.legend(ncol=2, bbox_to_anchor=(0.6, 1.02))
		ax2.set_facecolor("white")
		ax2.set_ylim(0, y_max)
		ax2.plot(smooth(a_s), linewidth=0.5, color='r')
		ax2.plot(smooth(thr), linewidth=1.0, color ='blue')
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel('Anomaly Score')
		# ax4.set_facecolor("gray")
		# ax4.plot(smooth(thre), linewidth=0.7, color='c')
		# ax4.set_xlabel('Timestamp')
		# ax4.set_ylabel('thresholds')
		pdf.savefig(fig)
		plt.close()
	pdf.close()

def plotter_frist(name, y):
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/train/12.pdf')
	for dim in range(y.shape[1]):
		y_t = y[:, dim]
		fig, ax = plt.subplots(facecolor='w')
		ax.set_facecolor('white')
		ax.set_facecolor("white")
		ax.set_ylabel('Value')
		ax.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax.plot(smooth(y_t), linewidth=0.6, label=y_t, color='y')
		# ax4.set_facecolor("gray")
		# ax4.plot(smooth(thre), linewidth=0.7, color='c')
		# ax4.set_xlabel('Timestamp')
		# ax4.set_ylabel('thresholds')
		pdf.savefig(fig)
		plt.close()
	pdf.close()

def plotter_train(name, y_true, y_pred):
	# if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/train/训练结果.pdf')
	for dim in range(y_true.shape[1]):
		y_t, y_p = y_true[:, dim], y_pred[:, dim]
		fig, ax = plt.subplots()#facecolor='w')
		ax.set_facecolor('white')
		ax.set_facecolor("white")
		ax.set_ylabel('Value')
		ax.set_title(f'Dimension = {dim}')
		# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
		ax.plot(smooth(y_t), linewidth=1.0, label=y_t, color='y')
		ax.plot(smooth(y_p), '-', alpha=0.6, linewidth=0.5, label=y_p, color='g')
		pdf.savefig(fig)
		plt.close()
	pdf.close()

def plotter_train_GDN(name, y_true, y_pred):
	# if 'TranAD' in name: y_true = torch.roll(y_true, 1, 0)
	os.makedirs(os.path.join('plots', name), exist_ok=True)
	pdf = PdfPages(f'plots/{name}/train/训练结果.pdf')
	y_t, y_p = y_true, y_pred
	fig, ax = plt.subplots(facecolor='w')
	ax.set_facecolor('white')
	ax.set_facecolor("white")
	ax.set_ylabel('Value')
	# if dim == 0: np.save(f'true{dim}.npy', y_t); np.save(f'pred{dim}.npy', y_p); np.save(f'ascore{dim}.npy', a_s)
	ax.plot(y_t, linewidth=1.0, label=y_t, color='y')
	ax.plot(y_p, '-', alpha=0.6, linewidth=0.5, label=y_p, color='g')
	pdf.savefig(fig)
	plt.close()
	pdf.close()