import cv2
import numpy as np
import torch
from models.matching import Matching
from models.utils import frame2tensor

torch.set_grad_enabled(False)

def main():
    # 设置设备为CPU
    device = 'cpu'
    print('Running inference on device \"{}\"'.format(device))

    # 配置参数
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.001,
            'max_keypoints': -1,
            "weights_path": r"C:\Users\Administrator\Desktop\superpoint_my_data\checkpoints\superPointNet_21800_checkpoint.pth.tar"
        },
        'superglue': {
            'weights': 'outdoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.25,
        }
    }

    # 初始化Matching模型
    matching = Matching(config).eval().to(device)

    # 读取图像
    image_path = r"E:\picture\0706split\segmented_image_2.png"  # 替换为你的图像路径
    image = cv2.imread(image_path)

    # 将图像转换为Tensor
    frame_tensor = frame2tensor(image, device)

    # 提取特征点
    data = matching.superpoint({'image': frame_tensor})
    keypoints = data['keypoints0'][0].cpu().numpy()

    # 绘制特征点
    out = image.copy()
    for pt in keypoints:
        cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)  # 绿色空心小圆

    # 显示图像
    cv2.imshow('Keypoints', out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()