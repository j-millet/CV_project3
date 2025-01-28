from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch

path = 'data/gtFine/train/aachen/aachen_000000_000019_gtFine_instanceIds.png'
mask = Image.open(path)

mask = F.resize(mask, (256, 512), interpolation=F.InterpolationMode.NEAREST)
mask = F.pil_to_tensor(mask).to(dtype=torch.int64)

print(torch.unique(mask))

zeroed_mask = mask.copy()
zeroed_mask[zeroed_mask < 33] = 0

print(np.unique(zeroed_mask))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(mask)
ax[1].imshow(zeroed_mask)
plt.show()