import matplotlib.pyplot as plt
folder_path = 'dataset/training_data/consep/consep/train/540x540_164x164/mask_original/'
img_paths = [
    'sandbox_1_0_original.png',
    'train_1_0.mask.png',
'train_1_0_0000.000000.population.count.black_bg.png',
'binary_nuclei_map_1_0.png']
fig, ax = plt.subplots(1, len(img_paths)+1, figsize=(4*(1+len(img_paths)), 4))
for i, img_path in enumerate(img_paths):
    img = plt.imread(folder_path + img_path)
    ax[i].imshow(img)
    ax[i].axis('off')
ax[-1].imshow(plt.imread(folder_path + img_paths[2])[..., 0] != 0, cmap='gray')
plt.show()
#plt.savefig('sandbox.png', dpi=300, bbox_inches='tight')
plt.close()