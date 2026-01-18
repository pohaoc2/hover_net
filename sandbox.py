import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import glob
folder_path = 'dataset/training_data/consep/consep/train/540x540_164x164/'
save_path = 'dataset/training_data/consep/consep/train/540x540_164x164/pngs/'
os.makedirs(save_path, exist_ok=True)
img_paths = [
    #'sandbox_1_0_original.png',
    #'train_1_0.mask.png',
#'train_9_9_0000.000000.population.count.black_bg.png',
#'binary_nuclei_map_1_0.png'
]
npy_paths = glob.glob(folder_path + '*.npy')

# load the npy file
for i, npy_path in enumerate(npy_paths[:]):
    if i%100 == 0: print(f"Processing: {npy_path}, [{i+1}/{len(npy_paths)}]")
    npy = np.load(npy_path)
    img = npy[..., :3].astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path + os.path.basename(npy_path).replace('.npy', '.png'), img_bgr)

