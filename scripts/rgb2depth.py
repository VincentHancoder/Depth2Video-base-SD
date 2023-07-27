# import cv2
import numpy as np
# import tqdm
# import matplotlib.pyplot as plt
# # 打开视频文件
# cap = cv2.VideoCapture('/home/hhn/cxk.mp4')

# # 获取视频的宽度和高度
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # 创建视频编写器
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output_video.mp4', fourcc, 100.0, (width, height), False)

# # 读取视频的每一帧并转换为Depth图像
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         # 将RGB图像转换为Depth图像
#         depth_image = cv2.cvtColor(frame, cv2.COLOR_RGB2DEPTH)
#         depth_image = np.stack((depth_image,)*3, axis=-1)
#         plt.imshow(depth_image)
#         plt.savefig("1.jpg")

#         # 将Depth图像写入视频文件
#         out.write(depth_image)

#         # 显示当前帧的Depth图像
#         # cv2.imshow('frame', depth_image)

#         # 按'q'键退出
#         # if cv2.waitKey(1) & 0xFF == ord('q'):
#         #     break
#     else:
#         break

# # 释放所有对象并关闭所有窗口
# # cap.release()
# # out.release()
# # cv2.destroyAllWindows()



import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import imageio

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda:1")
else:
    device = torch.device("cpu")

# 加载模型
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
model.to(device)
model.eval()

# 定义批处理大小
batch_size = 350
new_video = []
# 将多个视频帧一次性转换为深度图像
def convert_frames_to_depth(frames):
    # 将帧转换为PyTorch张量，并将其缩放为适当的大小（384x384）
    imgs = []
    for frame in frames:
        img = cv2.resize(frame, (384, 384))
        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        imgs.append(img)

    # 将输入数据组合成一个张量，准备进行批处理
    imgs = torch.stack(imgs, dim=0)
    imgs = imgs.to(device)

    # 使用MiDaS模型获取深度估计
    with torch.no_grad():
        prediction = model(imgs).to(device)

    # 将深度图像从张量中提取出来并将其归一化
    depths = prediction.squeeze().cpu().numpy()
    depths = [cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) for depth in depths]

    # 将深度图像调整为与输入图像相同的大小
    depths = [cv2.resize(depth, (frame.shape[1], frame.shape[0])) for depth, frame in zip(depths, frames)]

    return depths

# 打开输入视频文件
cap = cv2.VideoCapture('corgi_rgb.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# # # 创建视频编写器
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output_video.mp4', fourcc, 100.0, (width, height), False)

# 读取视频帧并将其转换为深度图像
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    

    # 如果当前帧数达到批处理大小，则进行深度估计
    if len(frames) == batch_size:
        depths = convert_frames_to_depth(frames)
        for depth in tqdm(depths):
            depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            new_video.append(depth)
            plt.imshow(depth)
            plt.savefig("1.jpg")
            # out.write(depth)
        #     cv2.imshow('depth', depth)
        #     cv2.waitKey(1)
        frames = []

# 如果还有未处理的视频帧，则进行深度估计
if len(frames) > 0:
    depths = convert_frames_to_depth(frames)
    for depth in tqdm(depths):
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        new_video.append(depth)
        plt.imshow(depth)
        plt.savefig("1.jpg")
        # out.write(depth)
    #     cv2.imshow('depth', depth)
    #     cv2.waitKey(1)

imageio.mimsave('output_rgb2depth.mp4', new_video, fps=fps)
cap.release()
# out.release()
