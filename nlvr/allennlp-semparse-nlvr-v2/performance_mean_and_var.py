import json
import glob
import numpy as np


path = './resources/checkpoints/nlvr/iwbs5_aww8_erm_soft/'
mode = 'ERM'

# seed_list = glob.glob(path + 'SEED*')
seed_list = [path + i for i in ['SEED_5', 'SEED_21', 'SEED_42', 'SEED_1337']]
# seed_list = [path + i for i in ['SEED_5', 'SEED_21', 'SEED_42', 'SEED_1337']]
# seed_list = [path + i for i in ['SEED_5', 'SEED_21', 'SEED_1337']]
bvda, bvc, btda, btc, bhda, bhc = [], [], [], [], [], []
for p in seed_list:
    dev = p + f'/{mode}/Iter5_MDS22/metrics.json'
    tests = {
        'test': p + f'/{mode}/Iter5_MDS22/metrics.txt', 
        'hidden': p + f'/{mode}/Iter5_MDS22/metrics_hidden.txt'
    }

    with open(dev, 'r') as f:
        dev = json.load(f)
        bvda.append(dev['best_validation_denotation_accuracy'])
        bvc.append(dev['best_validation_consistency'])

    with open(tests['test'], 'r') as f:
        precision = float(f.readline().split('=')[-1])
        consistency = float(f.readline().split('=')[-1])
        btda.append(precision)
        btc.append(consistency)

    with open(tests['hidden'], 'r') as f:
        precision = float(f.readline().split('=')[-1])
        consistency = float(f.readline().split('=')[-1])
        bhda.append(precision)
        bhc.append(consistency)

bvda = np.array(bvda)
bvc = np.array(bvc)
btda = np.array(btda)
btc = np.array(btc)
bhda = np.array(bhda)
bhc = np.array(bhc)

with open(path + f'/performance_p.txt', 'w') as f:
    for p in seed_list:
        f.write(p.split('/')[-1] + ' ')
    f.write('\n\n')
    f.write(f'dev acc: {np.mean(bvda):.3f} \pm {np.std(bvda)}\n')
    f.write(f'dev con: {np.mean(bvc):.3f} \pm {np.std(bvc)}\n')
    f.write(f'test acc: {np.mean(btda):.3f} \pm {np.std(btda)}\n')
    f.write(f'test con: {np.mean(btc):.3f} \pm {np.std(btc)}\n')

with open(path + f'/performance_h.txt', 'w') as f:
    for p in seed_list:
        f.write(p.split('/')[-1] + ' ')
    f.write('\n\n')
    f.write(f'dev acc: {np.mean(bvda):.3f} \pm {np.std(bvda)}\n')
    f.write(f'dev con: {np.mean(bvc):.3f} \pm {np.std(bvc)}\n')
    f.write(f'test acc: {np.mean(bhda):.3f} \pm {np.std(bhda)}\n')
    f.write(f'test con: {np.mean(bhc):.3f} \pm {np.std(bhc)}\n')

btda = (btda + bhda) / 2
btc = (btc + bhc) / 2

with open(path + f'/performance_test.txt', 'w') as f:
    for p in seed_list:
        f.write(p.split('/')[-1] + ' ')
    f.write('\n\n')
    f.write(f'dev acc: {np.mean(bvda):.3f} \pm {np.std(bvda)}\n')
    f.write(f'dev con: {np.mean(bvc):.3f} \pm {np.std(bvc)}\n')
    f.write(f'test acc: {np.mean(btda):.3f} \pm {np.std(btda)}\n')
    f.write(f'test con: {np.mean(btc):.3f} \pm {np.std(btc)}\n')