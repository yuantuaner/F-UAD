#####改数据的时候需要修改画图的存储命名包括：最终的，每个维度的
#####要是换参数的话，得将保存模型和加载模型的路径改掉
import os
import pandas as pd
from tqdm import tqdm
from src.models import *
from src.constants import *
from src.plotting import *
from src.pot import *
from src.utils import *
from src.diagnosis import *
from src.merlin import *
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
from time import time
from pprint import pprint
from src.loss import compute_mmd
from sklearn.linear_model import LinearRegression
from torch.cuda import empty_cache
from sklearn.feature_selection import VarianceThreshold
import pmdarima as pm
from statsmodels.tsa.seasonal import seasonal_decompose     # 做趋势分解用的库
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas   #将值保存到PDF中
import preprocess

# 计算反归一化
def denormalize4(a, min_a, max_a):
    # 使用广播机制对a进行反归一化
    aa = ((a + 1) * (max_a - min_a + 0.0001)) / 2 + min_a
    return aa

# 计算MAE,MSE
def calculate_mse(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    return mse
# 计算 MAE
def calculate_mae(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


# 计算相对斜率
def compute_relative_slopes(data):
    # 计算相邻点的差值
    slopes = data[1:] - data[:-1]  # 计算相邻点的差
    slopes = torch.cat((torch.zeros((1, data.shape[1], data.shape[2]), device=data.device), slopes), dim=0)  # 在开头补零
    return slopes


def convert_to_windows(data, model):
	windows = []; w_size = model.n_window
    # 将数据的第一列（标识符）移除，保留后面的16列

	labels = data[:, 0]  # 提取标识符
	data_no_label = data[:, 1:]  # 保留数据部分
	data_no_label = torch.from_numpy(data_no_label)  # 转换为 torch 张量
	unique_labels = np.unique(labels)  # 获取所有的标识符

	for label in unique_labels:
		# 取出相同标识符的数据
		indices = np.where(labels == label)[0]
		data_for_label = data_no_label[indices]  # 对应标识符的数据
		
		for i in range(len(data_for_label)):  # [0, len-1]
			# 创建窗口，取去掉标识符后的数据
			if i >= w_size:
				w = data_for_label[i-w_size+1:i+1]
			else:   
				# 补齐窗口前面的部分
				w = torch.cat([data_for_label[0].unsqueeze(0).repeat(w_size-i-1, 1), data_for_label[0:i+1]], dim=0)
			
			# 根据模型选择不同的数据处理方式
			windows.append(w if 'TranAD' in args.model or 'Attention' in args.model else w.view(-1))       
	return torch.stack(windows)

def load_dataset(dataset, specific_dataset=None):
	# LJH DO
	if specific_dataset is not None:
		PATH = '/home/jupyter-yyf-yyy/program/keyan/Anomaly-Detection-for-UAV/MLTD/data/ALFA/'
		csv_prefix = specific_dataset.split('-')[1]
		if specific_dataset.split('-')[0] == 'fy': # 副翼数据
			test_path = PATH + '副翼数据/12/只加标识符没有拼接/' + csv_prefix
			label_path = PATH + '副翼数据/12/只加标识符没有拼接/标签/' + csv_prefix
		else: # 方向舵
			test_path = PATH + '方向舵/12/' + csv_prefix
			label_path = PATH + '方向舵/标签/' + csv_prefix
		preprocess.load_data(dataset, test_path, label_path)

	folder = os.path.join(output_folder, dataset)        # 构建文件夹路径
	if not os.path.exists(folder):                       # 检查该路径是否存在
		raise Exception('Processed Data not found.')     
	loader = []
	for file in ['train', 'test', 'labels', 'vail', 'min', 'max']:                 
		if dataset == 'SMD': file = 'machine-1-1_' + file   
		if dataset == 'SMAP': file = 'P-1_' + file
		if dataset == 'MSL': file = 'C-1_' + file
		if dataset == 'UCR': file = '136_' + file
		if dataset == 'NAB': file = 'ec2_request_latency_system_failure_' + file
		if dataset == 'ALFA': file = file
		if dataset == 'thor': file = file
		loader.append(np.load(os.path.join(folder, f'{file}.npy')))            # 加载训练集、测试集和标签数据
	if args.less: loader[0] = cut_array(0.2, loader[0])                        # 训练数据进行裁剪，以使用更少的数据
	train_loader = loader[0]              
	test_loader = loader[1]                
	labels = loader[2]                     
	vail_loader = loader[3]
	min_loader = loader[4]
	max_loader = loader[5] 
	return train_loader, test_loader, labels, vail_loader, min_loader, max_loader                  # 返回训练和测试数据的加载器，以及标签

def save_model( model, optimizer, scheduler, epoch, accuracy_list, best_val_loss, lstm_model= False):
	folder = f'checkpoints/ALFA/{args.model}_{args.dataset}/'
	os.makedirs(folder, exist_ok=True)
	file_path = f'{folder}/12/model_rudder_fuyi.ckpt'
	torch.save({
		'epoch': epoch,
		'yucemodel_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'scheduler_state_dict': scheduler.state_dict(),
		'accuracy_list': accuracy_list,
		'best_val_loss': best_val_loss}, file_path)


def load_model(modelname, dims, is_test=False):
	import src.models
	model_class = getattr(src.models, modelname)
	model = model_class(dims).double()
	weight_decay = 1e-5
	optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=weight_decay)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)          # 学习率调度器，这会在每隔5个 epoch 后将学习率乘以 0.9，从而逐步减小学习率，避免训练过早收敛
	fname = f'checkpoints/ALFA/{args.model}_{args.dataset}/12/model_rudder_fuyi.ckpt'
	if torch.cuda.is_available():
		model.cuda()
	if os.path.exists(fname) and (not args.retrain or args.test or is_test or args.train):   # 重训练+测试需要用到保存的模型参数
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(fname)
		model.load_state_dict(checkpoint['yucemodel_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
		best_val_loss = checkpoint['best_val_loss']				
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		# 当没有预训练模型或需要重新训练时，初始化一个新的模型
		epoch = -1; accuracy_list = []; best_val_loss = float('inf')
	return model, optimizer, scheduler, epoch, accuracy_list, best_val_loss

# 定义验证函数
def validate(model, criterion, val_loader, feats):
	with torch.no_grad():
		model.eval()
		l1s =[]
		# with torch.no_grad():   # 不计算梯度
		for d, _ in val_loader:
			d = d.to(torch.float64).to(device)
			local_bs = d.shape[0]             # 表示批次中包含的数据点数
			window = d.permute(1, 0, 2)      # [window, 2903, dim]
			elem = window[-1, :, :].view(1, local_bs, feats)
			z  = model(window, elem)
			l1 = criterion(z, elem[0,:,:]) if not isinstance(z, tuple) else (1 / feats) * l(z[0], elem) + (1 - 1/feats) * l(z[1], elem)	
			l1s.append(l1.mean())	
		val_loss = torch.mean(torch.stack(l1s))  # 将列表转为张量，计算均值			
		return val_loss.item()    # 返回标量
	

def backprop(epoch, model, data, dataO, val_loader, optimizer, scheduler, best_val_loss, training = True):
	if 'TranAD_classi' in model.name:
		feats = dataO.shape[1]                                                                                                         
		l = nn.MSELoss(reduction = 'none')                                               # 初始化损失函数
		# 将data数据转化为DoubleTensor, 并创建一个'TensorDataset',其中数据本身作为输入和标签.这种做法常见于自编码器,其中目标是重构输入
		data_x = torch.DoubleTensor(data); dataset = TensorDataset(data_x, data_x)     
		data_vail = torch.DoubleTensor(val_loader); vail = TensorDataset(data_vail, data_vail)     
		bs = model.batch 
		dataloader = DataLoader(dataset, batch_size = bs)
		vail = DataLoader(vail, batch_size = bs)
		n = epoch + 1; w_size = model.n_window
		z1, elem1, l1s = [], [], []
		i = 0
		accumulator = 4
		if training:
			for d, _ in dataloader:
				d = d.to(torch.float64).to(device)
				i = i+1
				local_bs = d.shape[0]             # 表示批次中包含的数据点数
				window = d.permute(1, 0, 2)       
				# # 调整维度后的'window'中最后一个时间步的所有数据. -1表示最后一个元素, 因此这是取出批次的每个窗口的最后一时间步的特征
				# # 将选取的数据重新塑形, 1表示新的维度, 0保持时间步的一致性, 'local_bs'是批次的大小,'feats'是特征数. 
				elem = window[-1, :, :].view(1, local_bs, feats)        
				
				z = model(window, elem)            
				# 使用单一损失
				l1 = l(z, elem[0,:,:]) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)	

				if isinstance(z, tuple): z = z[1]         
				z1.append(z.detach())            # z1是脱离计算图的
				l1s.append(torch.mean(l1).item())			         
				loss1 = torch.mean(l1)	
				loss1.backward(retain_graph=True)              # 产生梯度
				if (i + 1) % accumulator == 0:
					optimizer.step()                           # 根据梯度进行优化
					optimizer.zero_grad()	
					i = 0				
			scheduler.step()			
			tqdm.write(f'Epoch {epoch},\tL1 = {np.mean(l1s)}')
			#print(len(z1))
			# 显存清理
			del z, loss1 
			empty_cache()
			# 每个 epoch 结束后计算验证损失
			val_loss = validate(model, l, vail, feats)  # 假设 val_loader 是你的验证集加载器
			# 检查是否是最佳验证损失 
			if val_loss < best_val_loss:
				best_val_loss = val_loss
				# 保存当前最佳模型
				print(f"Best model saved at epoch {epoch} with validation loss: {best_val_loss}")		
				save_model(model, optimizer, scheduler, epoch, accuracy_list, best_val_loss)
			return np.mean(l1s), optimizer.param_groups[0]['lr'], z1, val_loss
		else:
			with torch.no_grad():
				model.eval()
				for d, _ in dataloader:
					d = d.to(torch.float64).to(device)
					local_bs = d.shape[0]             # 表示批次中包含的数据点数
					window = d.permute(1, 0, 2)
					elem = window[-1, :, :].view(1, local_bs, feats)
					z = model(window, elem)
					l1 = l(z, elem[0,:,:]) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1/n) * l(z[1], elem)
					if isinstance(z, tuple): z = z[1]         
					z1.append(z)
					elem1.append(elem[0, :, :])
					l1s.append(l1)
				l1ss = torch.cat(l1s, dim=0)  
				z1s = torch.cat(z1, dim=0)  	
				elem1s = torch.cat(elem1, dim=0)		
				return l1ss.cpu().numpy(), z1s.cpu().numpy(), elem1s.cpu().numpy()

if __name__ == '__main__':    

    #准备模型，数据集阶段
    # LJH DO
    # 添加一个args参数'specific_dataset'来指定具体的如数据集：
    # 具体来说可以选择 副翼：`fy-i`(i取值范围是1~7)，或者方向舵：`fxd-j`（j取值范围是1~4)

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))
    #准备模型，数据集阶段
    train_loader, test_loader, labels, vail, min, max = load_dataset(args.dataset, args.specific_dataset)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    label = labels[:, :]
	### Plot curves
    model, optimizer, scheduler, epoch, accuracy_list, best_val_loss = load_model(args.model, label.shape[1])    # 加载一些参数
    seed = 42
    torch.manual_seed(seed)
	
    trainO, testO =	train_loader[:, 1:]  , test_loader[:, 1:]
    trainD, testD, vailD = convert_to_windows(train_loader, model), convert_to_windows(test_loader, model), convert_to_windows(vail, model)         

	## Prepare data
    trainD = DataLoader(trainD, batch_size=trainD.shape[0], shuffle=True)        # 创建加载器，设置批量大小为整个数据集的大小(每批处理全部数据
    trainO = DataLoader(trainO, batch_size=trainO.shape[0], shuffle=True)
    vailD = DataLoader(vailD, batch_size=vailD.shape[0])
    testD = DataLoader(testD, batch_size=testD.shape[0])
    testO = DataLoader(testO, batch_size=testO.shape[0])
    trainD, testD, vailD = next(iter(trainD)), next(iter(testD)), next(iter(vailD))
    trainO, testO = next(iter(trainO)), next(iter(testO))       # 返回第一个批次
	
	### Training phase 接着训练
    if args.train:
		# 训练阶段
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        val_loss_list = []
        num_epochs = 400; e = epoch + 1; start = time()
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            lossT, lr, z1 , val_loss = backprop(e, model, trainD, trainO, vailD, optimizer, scheduler, best_val_loss)
            accuracy_list.append((lossT, lr))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            val_loss_list.append(val_loss)		    
        print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)

    	### Plot curves train 
        z1 = torch.cat(z1)
        z1 = z1.cpu().numpy()

        plotter_train(f'{args.model}_{args.dataset}', trainO, z1)

	### Testing phase
    elif args.test:
		#测试阶段
        model.eval()
        print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')

		# 清除优化器中的梯度
        optimizer.zero_grad()
        loss, y_pred, y_true = backprop(0, model, testD, testO, vailD, optimizer, scheduler, best_val_loss, training=False)
        mse = calculate_mse(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)

	
		### Scores
        df = pd.DataFrame()
        thre = []
        scores = []
        lossT, _, _ = backprop(0, model, trainD, trainO, vailD, optimizer, scheduler, best_val_loss, training=False)	
        for i in range(loss.shape[1]):
            lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]    #  参数分别为训练，测试，标签
            result, score, pred = eval(lt, l, ls, model, f'{args.model}_{args.dataset}', f'{args.specific_dataset}', weidu = i)           
            preds.append(pred)
            df = df._append(result, ignore_index=True)

		# # 将标签改为原始的
        act = (labels != 0).astype(int)
        #使用静态阈值
        re = np.tile(df['threshold'], (testO.shape[0], 1))         

		# 反归一化
        testO_deno = denormalize4(testO, min, max)

        y_pred_deno = denormalize4(y_pred, min, max)
   
        plotter_picture(f'{args.model}_{args.dataset}', f'{args.specific_dataset}', testO_deno, y_pred_deno, loss, act, re)
        plotter_pdf(f'{args.model}_{args.dataset}', f'{args.specific_dataset}', testO_deno, y_pred_deno, loss, act, re)
		

		# loss是测试集的损失, lossT是训练集的损失		
        lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0		
        result, scorefinal, _ = eval(lossTfinal, lossFinal, labelsFinal, model, f'{args.model}_{args.dataset}', f'{args.specific_dataset}', MM = 'True')   # result：好多的结果值
		#####使用静态阈值
        refinal = np.tile(result['threshold'], (testO.shape[0], 1))
	
        score_reshaped = lossFinal.reshape(-1, 1)
        plotter_final(f'{args.model}_{args.dataset}', f'{args.specific_dataset}', refinal, score_reshaped, act)
        result.update(hit_att(loss, act))
        result.update(ndcg(loss, act))
        print('MSE:', mse)
        print('MAE:', mae)
        print(df)
        pprint(result)

    
    elif args.retrain:
		# 训练阶段
        print(f'{color.HEADER}Training {args.model} on {args.dataset}{color.ENDC}')
        val_loss_list = []
        num_epochs = 400; e = epoch + 1; start = time()
        for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
            lossT, lr, z1 , val_loss = backprop(e, model, trainD, trainO, vailD, optimizer, scheduler, best_val_loss)
            accuracy_list.append((lossT, lr))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            val_loss_list.append(val_loss)		    
        print(color.BOLD+'Training time: '+"{:10.4f}".format(time()-start)+' s'+color.ENDC)

        plot_accuracies(accuracy_list, val_loss_list, f'{args.model}_{args.dataset}/train/lr_loss.pdf')    # 红色是学习率

    	### Plot curves train 
        z1 = torch.cat(z1)
        z1 = z1.cpu().numpy()
        plotter_train(f'{args.model}_{args.dataset}', trainO, z1)
        
		#测试阶段
        model, optimizer, scheduler, epoch, accuracy_list, best_val_loss = load_model(args.model, label.shape[1], is_test=True)
        model.eval()
        print(f'{color.HEADER}Testing {args.model} on {args.dataset}{color.ENDC}')

		# 清除优化器中的梯度
        optimizer.zero_grad()
        loss, y_pred, y_true = backprop(0, model, testD, testO, vailD, optimizer, scheduler, best_val_loss, training=False)
        mse = calculate_mse(y_true, y_pred)
        mae = calculate_mae(y_true, y_pred)

		### Scores
        df = pd.DataFrame()

        thre = []
        scores = []

        lossT, _, _ = backprop(0, model, trainD, trainO, vailD, optimizer, scheduler, best_val_loss, training=False)

        for i in range(loss.shape[1]):
            lt, l, ls = lossT[:, i], loss[:, i], labels[:, i]    #  参数分别为训练，测试，标签
            result, score, pred = eval(lt, l, ls, model, f'{args.model}_{args.dataset}', f'{args.specific_dataset}', weidu = i)           
            preds.append(pred)
            df = df._append(result, ignore_index=True)

		# # 将标签改为原始的
        act = (labels != 0).astype(int)
        re = np.tile(df['threshold'], (testO.shape[0], 1))         # 将计算的阈值【12】个数展成【test.shape【0】, 12】

		# 反归一化
        testO_deno = denormalize4(testO, min, max)
        y_pred_deno = denormalize4(y_pred, min, max)
        plotter_picture(f'{args.model}_{args.dataset}', f'{args.specific_dataset}', testO_deno, y_pred_deno, loss, act, re)
        plotter_pdf(f'{args.model}_{args.dataset}', f'{args.specific_dataset}', testO_deno, y_pred_deno, loss, act, re)

		# loss是测试集的损失, lossT是训练集的损失
        lossTfinal, lossFinal = np.mean(lossT, axis=1), np.mean(loss, axis=1)
        labelsFinal = (np.sum(labels, axis=1) >= 1) + 0	
        result, scorefinal, _ = eval(lossTfinal, lossFinal, labelsFinal, model, f'{args.model}_{args.dataset}', f'{args.specific_dataset}', MM = 'True')   # result：好多的结果值
        refinal = np.tile(result['threshold'], (testO.shape[0], 1))
        score_reshaped = lossFinal.reshape(-1, 1)

        plotter_final(f'{args.model}_{args.dataset}', f'{args.specific_dataset}', refinal, score_reshaped, act)

        result.update(hit_att(loss, act))
        result.update(ndcg(loss, act))
        print('MSE:', mse)
        print('MAE:', mae)
        print(df)
        pprint(result)

