#利用StableDiffusionDepth2ImgPipeline里面内置的rgb转换depth的函数将pipeline中rgb转换为depth的部份可视化出来
from diffusers import StableDiffusionDepth2ImgPipeline
import PIL
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import imageio


# image = PIL.Image.open('/home/hhn/tencent-test/corgi_rgb.jpg')
# image = np.array(image)
# image = [image]
# image = torch.tensor(image,dtype = torch.float16)
# # import pdb;pdb.set_trace()
# pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
#    "stabilityai/stable-diffusion-2-depth",
#    torch_dtype=torch.float16,
# ).to("cuda:1")
# depth = pipe.prepare_depth_map(image = image,depth_map = None,batch_size = 1,do_classifier_free_guidance = None,dtype = torch.float16,device = "cuda:1")
# print(depth.shape)#类型是tensor
# depth = torch.squeeze(depth,0)
# depth = depth.permute(1,2,0).cpu()
# depth = depth.detach().numpy()
# plt.imshow(depth)
# plt.savefig('corgi_rgb2depth.jpg')#!!!成了！！！可以展示转换的图片


#展示出rgb2depth的视频
cap = cv2.VideoCapture('/home/hhn/tencent-test/cxk1.mp4')
# 获取视频的宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
   "stabilityai/stable-diffusion-2-depth",
   torch_dtype=torch.float16,
).to("cuda:1")

i = 0#count
frames = []
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    i += 1
    print(i,'/',frame_count)
    frame = [frame]
    frame = torch.tensor(frame,dtype = torch.float16)
    depth = pipe.prepare_depth_map(image = frame,depth_map = None,batch_size = 1,do_classifier_free_guidance = None,dtype = torch.float16,device = "cuda:1")
    depth = torch.squeeze(depth,0)
    depth = depth.permute(1,2,0).cpu()
    depth = depth.detach().numpy()
    plt.imshow(depth)
    plt.savefig('5.jpg')
    frames.append(depth)
imageio.mimsave('output_diffuser_cxk_rgb2depth.mp4', frames, fps=fps)




