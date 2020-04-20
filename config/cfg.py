import numpy as np


arch = 'ro_34'
heads = {'ct_hm': 7, 'wh': 2}
head_conv = 256
num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
arch = arch[:arch.find('_')] if '_' in arch else arch
down_ratio = 4
ct_score = 0.2

mean = np.array([0.40789654, 0.44719302, 0.47026115], dtype=np.float32).reshape(1, 1, 3)
std = np.array([0.28863828, 0.27408164, 0.27809835], dtype=np.float32).reshape(1, 1, 3)
