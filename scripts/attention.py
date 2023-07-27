import torch
import torch.nn as nn
from PIL import Image
import numpy as np

x = Image.open('/home/hhn/tencent-test/corgi_rgb.jpg')
# 定义输入图片
x = np.array(x)#[800,800,3]
x = torch.FloatTensor(x).to("cuda")
x = x.permute(2,0,1)#[3,800,800ss]
x = x.unsqueeze(0)#[1,3,800,800]
x = torch.nn.functional.interpolate(x, size=(400,400), mode='bilinear', align_corners=False).to("cuda")#插值变成[1,3,200,200]

# 定义 MultiheadAttention 模块
attn = nn.MultiheadAttention(embed_dim=400*400, num_heads=8).to("cuda")
print(x.shape)
# 将图片展平成序列
x_flat = x.view(1, 3, -1)#[1,3,1024]
print(x_flat.shape)#[1,3,1024]

# 计算注意力
attn_output, attn_weights = attn(x_flat,x_flat,x_flat)#q\k\v

# 输出注意力结果的形状
print('注意力输出的形状：', attn_output.shape)
print('注意力权重的形状：', attn_weights.shape)

attn_output = attn_output.reshape(1, 3, 400, 400)

import matplotlib.pyplot as plt

# 将张量转换为 numpy 数组，并移动到 CPU 上
attn_output_np = attn_output.squeeze().permute(1, 2, 0).cpu().detach().numpy()

# 显示图像
plt.imshow(attn_output_np)
plt.savefig("attn_corgi.jpg")

y = Image.open('/home/hhn/tencent-test/corgi_depth.jpg')
y = np.array(y)#[800,800]
y = y.reshape((1,400,400))
y = y.repeat(3,axis=0)#[3,800,800]
y = torch.FloatTensor(y).to("cuda")
# y = y.permute(2,0,1)#[3,800,800]
y = y.unsqueeze(0)#[1,3,800,800]
y = torch.nn.functional.interpolate(y, size=(400,400), mode='bilinear', align_corners=False)#插值变成[1,3,200,200]

attn_output_np = attn_output.cpu().detach().numpy() # 将张量转换为NumPy数组
attn_output_np = attn_output_np.reshape(3, 400, 400) # 调整形状 [3,200,200]


overlay = y.squeeze().cpu().detach().numpy() # 将输入图片从张量格式转换为NumPy数组 [3,200,200]
overlay = np.clip(overlay, 0, 1) # 将像素值限制在 [0,1] 范围内,归一化
overlay += attn_output_np # 将输出的特征图叠加到输入图片上
print(overlay.shape)
overlay = np.clip(overlay, 0, 1) # 将像素值限制在 [0,1] 范围内
overlay = torch.FloatTensor(overlay).to("cuda")
overlay = overlay.permute(1,2,0)#[200,200,3]
overlay = overlay.cpu().detach().numpy()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
axs[0].imshow(y.squeeze().permute(1,2,0).cpu().detach().numpy())
axs[0].set_title("depth Image")
axs[1].imshow(x.squeeze().permute(1,2,0).cpu().detach().numpy())
axs[1].set_title("rgb  Image")
axs[2].imshow(overlay)
axs[2].set_title("Attention Map Overlay")
plt.savefig("attention_visiual.jpg")
plt.show()






