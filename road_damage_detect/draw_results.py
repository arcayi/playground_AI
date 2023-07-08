# -*- coding: utf-8 -*-

# 此文件中定义画图相关的函数

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread

from insects_reader import INSECT_NAMES

import cv2


# 定义画矩形框的函数
def draw_rectangle(currentAxis, bbox, edgecolor="k", facecolor="y", fill=False, linestyle="-"):
    # currentAxis，坐标轴，通过plt.gca()获取
    # bbox，边界框，包含四个数值的list， [x1, y1, x2, y2]
    # edgecolor，边框线条颜色
    # facecolor，填充颜色
    # fill, 是否填充
    # linestype，边框线型
    # patches.Rectangle需要传入左上角坐标、矩形区域的宽度、高度等参数
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0] + 1,
        bbox[3] - bbox[1] + 1,
        linewidth=1,
        edgecolor=edgecolor,
        facecolor=facecolor,
        fill=fill,
        linestyle=linestyle,
    )
    currentAxis.add_patch(rect)


# 定义绘制预测结果的函数
def draw_results(result, filename, draw_thresh=0.5):
    plt.figure(figsize=(10, 10))
    im = imread(filename)
    plt.imshow(im)
    currentAxis = plt.gca()
    colors = ["r", "g", "b", "k", "y", "pink", "purple"]
    for item in result:
        box = item[2:6]
        label = int(item[0])
        name = INSECT_NAMES[label]
        if item[1] > draw_thresh:
            draw_rectangle(currentAxis, box, edgecolor=colors[label])
            plt.text(box[0], box[1], name, fontsize=12, color=colors[label])
    plt.savefig(".output/output_pic.png")
    plt.show()


# 绘制结果
def visualize_results(results, image, draw_thresh=0.5):
    plt.figure(figsize=(10, 10))
    # im = imread(filename)
    # plt.imshow(image)
    # currentAxis=plt.gca()
    # colors = ["r", "g", "b", "k", "y", "pink", "purple"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 0, 0), (0, 255, 255), (203, 192, 255), (255, 0, 255)]
    for res in results:
        for item in res:
            box = item[2:6]
            label = int(item[0])
            name = INSECT_NAMES[label]
            if item[1] > draw_thresh:
                # draw_rectangle(currentAxis, box, edgecolor=colors[label])
                # plt.text(box[0], box[1], name, fontsize=12, color=colors[label])
                xmin = (int)(box[0])
                ymin = (int)(box[1])
                # xmax = (int)(box[2] + box[0])
                # ymax = (int)(box[3] + box[1])
                xmax = (int)(box[2])
                ymax = (int)(box[3])

                # cv2.rectangle(image, box[:2], box[2:4], colors[label])
                # cv2.putText(image,name,box[:2])
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), colors[label])
                cv2.putText(image, name, (xmin, ymin), 3, 1, colors[label])
    return image

    # plt.savefig(".output/output_pic.png")
    # plt.show()
