#是直接使用diffuser中StableDiffusionDepth2ImgPipeline进行生成的，但是pipeline会自行将rgb转换为depth再辅助生成，所以输入还不是depth
import torch
import requests
import PIL
from diffusers import StableDiffusionDepth2ImgPipeline
import matplotlib.pyplot as plt
import cv2
import numpy as np
import imageio
import torch.nn as nn
from models.unet import UNet3DConditionModel
from diffusers.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin
from PIL import Image

def attn_similarity(a,b):#a是彩图应该是numpy array:frames[-1]，b是depth图是torch.tensor:frame[3,800,800]
    x = np.array(a)#[800,800,3]
    x = torch.FloatTensor(x).to("cuda")
    x = x.permute(2,0,1)#[3,800,800]
    x = x.unsqueeze(0)#[1,3,800,800]
    x = torch.nn.functional.interpolate(x, size=(100,100), mode='bilinear', align_corners=False).to("cuda")#插值变成[1,3,100,100]
    # y = np.array(y)#[800,800]
    # y = y.reshape((1,800,800))
    # y = y.repeat(3,axis=0)#[3,800,800]
    y = b.to("cuda")
    y = y.unsqueeze(0)#[1,3,800,800]
    y = torch.nn.functional.interpolate(y, size=(100,100), mode='bilinear', align_corners=False).to("cuda")#插值变成[1,3,100,100]
    attn = nn.MultiheadAttention(embed_dim=100*100, num_heads=8).to("cuda")
    x_flat = x.view(1, 3, -1)
    y_flat = y.view(1, 3, -1)
    attn_output, attn_weights = attn(x_flat,y_flat,y_flat)
    attn_output = attn_output.reshape(1,3,100,100)
    attn_output = torch.nn.functional.interpolate(attn_output, size=(800,800), mode='bilinear', align_corners=False).to("cuda")#[1,3,800,800]
    # import pdb;pdb.set_trace()
    attn_output = attn_output.squeeze().permute(1,2,0).cpu().detach().numpy()#[800,800,3]
    attn_output = (attn_output * 255).astype(np.uint8)
    attn_output += b.permute(1,2,0).cpu().detach().numpy().astype(np.uint8)#[800,800,3] + [800,800,3]
    return attn_output
    
    

cap = cv2.VideoCapture("/home/hhn/tencent-test/hamburger_depth.mp4")
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
).to("cuda:3")

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
        init_image = PIL.Image.fromarray(frame)#转换格式,将array的frame转换为pipe(image=)可以读取的PIL格式,pil是直接将[800,800,3]变成PIL格式的
        prompt = "a brown hamburger with red meat and green vegetables"
        n_propmt = "bad, deformed, ugly, bad anotomy"
        #转换为depth_map可以读取的格式
        frame = torch.FloatTensor(frame)#将frame从array变为tensor
        frame = frame.permute(2,0,1)#转换维度为[3,800,800],这样就是depth_map可以读取的格式了
        if frames == []:
            image = pipe(prompt=prompt, depth_map=None,image=init_image, negative_prompt=n_propmt, strength=0.7).images[0]#输出的image仍然是PIL格式
        else:
            image = pipe(prompt=prompt, depth_map=frame,image=PIL.Image.fromarray(frames[-1]), negative_prompt=n_propmt, strength=0.7).images[0]

        plt.imshow(image)
        plt.show()
        plt.savefig("./11.jpg")
        output_frame = np.array(image)#转换为numpy格式 [800,800,3]
        frames.append(output_frame)#将np.array 读入一个新的列表中
        #out.write(output_frame)
    else:
        break
   
cap.release()
# out.release()
imageio.mimsave('output_depth2video_hamburger_frames-1_diffratio.mp4', frames, fps=fps)#亲测imageio可以写入mp4！！！
