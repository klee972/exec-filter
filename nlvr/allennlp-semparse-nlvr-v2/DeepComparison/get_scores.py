import json
import glob
import numpy as np


# exp_name = 'iwbs5_aww8'
# exp_name = 'paired'
exp_name = 'iwbs5_paired'

path = f'../resources/checkpoints/nlvr/{exp_name}/'


seed_list = [path + i for i in ['SEED_5', 'SEED_21', 'SEED_42', 'SEED_1337']]
bvda, bvc, btda, btc, bhda, bhc, bsda, bsc = [], [], [], [], [], [], [], []
for p in seed_list:
    dev = p + '/ERM/Iter5_MDS22/metrics.json'
    test = p + '/ERM/Iter5_MDS22/metrics.txt'
    hidden = p + '/ERM/Iter5_MDS22/metrics_hidden.txt'

    with open(dev, 'r') as f:
        dev = json.load(f)
        bvda.append(dev['best_validation_denotation_accuracy'])
        bvc.append(dev['best_validation_consistency'])

    with open(test, 'r') as f1, open(hidden, 'r') as f2:
        precision1 = float(f1.readline().split('=')[-1])
        consistency1 = float(f1.readline().split('=')[-1])
        btda.append(precision1)
        btc.append(consistency1)

        precision2 = float(f2.readline().split('=')[-1])
        consistency2 = float(f2.readline().split('=')[-1])
        bhda.append(precision2)
        bhc.append(consistency2)

        bsda.append((precision1+precision2)/2)
        bsc.append((consistency1+consistency2)/2)




with open(f'./scores/{exp_name}_pcon', 'w') as f:
    for item in btc:
        f.write(str(item))
        f.write('\n')

with open(f'./scores/{exp_name}_vcon', 'w') as f:
    for item in bvc:
        f.write(str(item))
        f.write('\n')

with open(f'./scores/{exp_name}_pacc', 'w') as f:
    for item in btda:
        f.write(str(item))
        f.write('\n')

with open(f'./scores/{exp_name}_vacc', 'w') as f:
    for item in bvda:
        f.write(str(item))
        f.write('\n')

with open(f'./scores/{exp_name}_hcon', 'w') as f:
    for item in bhc:
        f.write(str(item))
        f.write('\n')

with open(f'./scores/{exp_name}_tcon', 'w') as f:
    for item in bsc:
        f.write(str(item))
        f.write('\n')

with open(f'./scores/{exp_name}_hacc', 'w') as f:
    for item in bhda:
        f.write(str(item))
        f.write('\n')

with open(f'./scores/{exp_name}_tacc', 'w') as f:
    for item in bsda:
        f.write(str(item))
        f.write('\n')
