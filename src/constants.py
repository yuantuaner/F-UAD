from src.parser import *
from src.folderconstants import *


#rudder:'ALFA':[(0.9, 1), (0.9999, 0.99)], 【0.93，0.95】结果非常大
#第一个值越大，阈值越小，第二个值越小，阈值越小
# 副翼：副翼03：[(0.9, 1), (0.87, 0.01)]
#副翼几不记得了：(0.23, 0.25)
####01 
# (0.5, 0.59),'f1': 0.9486436233645534, 'precision': 0.9410187415276478, 'recall': 0.9564032436947346,'Acc': 0.9783228750713063,
# (0.48,0.58)'f1': 0.9489197066036866,'precision': 0.9363394977098277,'recall': 0.9618528348268983,
####02
#副翼07：(0.0388, 0.0400)    --最佳
##副翼06：(0.355, 0.366)
# Threshold parameters
lm_d = {
		'SMD': [(0.99995, 1.04), (0.99995, 1.06)],
		'synthetic': [(0.999, 1), (0.999, 1)],
		'SWaT': [(0.993, 1), (0.993, 1)],
		'UCR': [(0.993, 1), (0.99935, 1)],
		'NAB': [(0.991, 1), (0.99, 1)],
		'SMAP': [(0.98, 1), (0.98, 1)],
		'MSL': [(0.97, 1), (0.999, 1.04)],
		'WADI': [(0.99, 1), (0.999, 1)],
		'MSDS': [(0.91, 1), (0.9, 1.04)],
		'MBA': [(0.87, 1), (0.93, 1.04)],
        'ALFA':[(0.9, 1), (0.03, 0.04)],    #(0.193, 0.233)，(0.25,0.35),
	}
lm = lm_d[args.dataset][1 if 'TranAD' in args.model else 0]

# Hyperparameters
lr_d = {
		'SMD': 0.0001, 
		'synthetic': 0.0001, 
		'SWaT': 0.008, 
		'SMAP': 0.001, 
		'MSL': 0.002, 
		'WADI': 0.0001, 
		'MSDS': 0.001, 
		'UCR': 0.006, 
		'NAB': 0.009, 
		'MBA': 0.001, 
        'ALFA': 0.009,
		'thor': 0.001
	}
lr = lr_d[args.dataset]

# Debugging
percentiles = {
		'SMD': (98, 2000),
		'synthetic': (95, 10),
		'SWaT': (95, 10),
		'SMAP': (97, 5000),
		'MSL': (97, 150),
		'WADI': (99, 1200),
		'MSDS': (96, 30),
		'UCR': (98, 2),
		'NAB': (98, 2),
		'MBA': (99, 2),
        'ALFA': (99, 21),
        'thor': (99, 20)
	}
percentile_merlin = percentiles[args.dataset][0]
cvp = percentiles[args.dataset][1]
preds = []
debug = 9