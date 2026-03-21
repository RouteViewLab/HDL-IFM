import cv2
import numpy as np
import os
import time


def rotate_white_regions(input_image_path, output_dir, rotation_step=1):
    # 记录程序开始时间
    start_time = time.time()

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取输入图像
    image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("无法读取图像，请检查路径")
        return

    # 二值化图像（确保白色区域为255，背景为0）
    _, binary = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("未找到白色闭合区域")
        return

    # 合并所有轮廓（处理多个白色区域的情况）
    mask = np.zeros_like(binary)
    cv2.drawContours(mask, contours, -1, 255, -1)

    # 计算几何中心（所有白色像素的质心）
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        print("无法计算几何中心")
        return

    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])

    # 创建旋转中心
    center = (cx, cy)

    # 初始化计数器
    image_count = 0

    # 旋转并保存图像
    for angle in range(0, 360, rotation_step):
        # 创建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 应用旋转
        rotated = cv2.warpAffine(mask, rotation_matrix, (mask.shape[1], mask.shape[0]))

        # 确保输出只有旋转后的白色区域
        result = np.zeros_like(image)
        result[rotated == 255] = 255

        # 保存图像
        output_path = os.path.join(output_dir, f"rotated_{angle:03d}.png")
        cv2.imwrite(output_path, result)
        image_count += 1
        print(f"已保存: {output_path}")

    # 计算总运行时间
    end_time = time.time()
    total_time = end_time - start_time

    # 打印统计信息
    print("\n===== 程序执行完成 =====")
    print(f"处理图像数量: {image_count}张")
    print(f"总运行时间: {total_time:.2f}秒")
    print(f"平均每张处理时间: {total_time / image_count:.4f}秒" if image_count > 0 else "无图片生成")


# 使用示例
image_path = r"E:\picture\0706split\segmented_image_6.png"  # 替换为你的输入图像路径
output_dir = r"E:\picture\0706split\B4360"  # 替换为你的输出目录

rotate_white_regions(image_path, output_dir)