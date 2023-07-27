# Depth2Video-base-SD

本Project是针对 视频生成 任务，区别于类似TuneAVideo、FollowYourPose等论文需要依靠2D数据进行finetune的手段，在这里是无需训练过程完全依靠注意力机制+Stable Diffusion实现zero-shot的depth2video视频生成推理过程。**注意**，这里注意力机制主要是作用在输入的深度图与生成的rgb图像之间来指导下一帧的生成。

## 输入视频

<div style="display: flex;">
  <video src="input/corgi_depth.mp4" width="50%" controls></video>
  <video src="input/hamburger_depth.mp4" width="50%" controls></video>
</div>

- [视频标题 1](output/视频1.mp4)
- [视频标题 2](output/视频2.mp4)

## 文件目录结构
