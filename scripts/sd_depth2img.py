#是直接使用diffuser中StableDiffusionDepth2ImgPipeline进行生成的，但是pipeline会自行将rgb转换为depth再辅助生成，所以输入还不是depth
import torch
import requests
import PIL
from diffusers import StableDiffusionDepth2ImgPipeline
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio
from models.unet import UNet3DConditionModel
from diffusers.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin


cap = cv2.VideoCapture("/home/hhn/tencent-test/output_diffuser_cxk_rgb2depth.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# print(width,height)  [800,800]
# import pdb;pdb.set_trace()
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 创建视频编写器
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# out = cv2.VideoWriter('output_video.mp4', fourcc, 100.0, (width, height), False)

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(#这个pipe只能一帧一帧的生成，是不能直接video2video生成的
   "stabilityai/stable-diffusion-2-depth",
   torch_dtype=torch.float16
).to("cuda:4")

frames = []

i = 0#显示进度
while(cap.isOpened()):
   ret,frame = cap.read()#frame是800x800x3的numpy array
   if ret == True:
      i += 1
      print(i,"/",frame_count)
      # print(frame.shape)  [800,800,3]
      # import pdb;pdb.set_trace()
      #init_image = Image.open(frame)#格式不匹配，cv2打开的不论是视频还是图片都是numpy格式的，而PIL是自己一个格式
      init_image = PIL.Image.fromarray(frame)#转换格式,将array的frame转换为pipe(image=)可以读取的PIL格式
      prompt = "spiderman is playing basketball on the street"
      n_propmt = "bad, deformed, ugly, bad anotomy"
      #import pdb;pdb.set_trace()
      frame = torch.FloatTensor(frame)
      frame = frame.permute(2,0,1)
      image = pipe(prompt=prompt, depth_map=None,image=init_image, negative_prompt=n_propmt, strength=0.7).images[0]#输出的image仍然是PIL格式
      plt.imshow(image)
      plt.show()
      plt.savefig("./8.jpg")
      output_frame = np.array(image)#转换为numpy格式
      import pdb;pdb.set_trace()
      frames.append(output_frame)#将np.array 读入一个新的列表中
      #out.write(output_frame)
   else:
      break
   
cap.release()
# out.release()
imageio.mimsave('output_depth2video_cxk_input_i.mp4', frames, fps=fps)#亲测imageio可以写入mp4！！！
      

# init_image = Image.open('/home/hhn/tencent-test/corgi_rgb.jpg')
# # import pdb;pdb.set_trace()

# prompt = "a rabit"
# n_propmt = "bad, deformed, ugly, bad anotomy"
# image = pipe(prompt=prompt, image=init_image, negative_prompt=n_propmt, strength=0.7).images[0]
# plt.imshow(image)
# plt.show()
# plt.savefig("./2.jpg")