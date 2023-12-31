# -*- coding: utf-8 -*-

# 此文件中定义数据读取相关的函数

import os
import cv2
import numpy as np
import paddle
import functools

from image_utils import image_augment
from insects_reader import get_insect_names, get_annotations


def get_bbox(gt_bbox, gt_class):
    # 对于一般的检测任务来说，一张图片上往往会有多个目标物体
    # 设置参数MAX_NUM = 50， 即一张图片最多取50个真实框；如果真实
    # 框的数目少于50个，则将不足部分的gt_bbox, gt_class和gt_score的各项数值全设置为0
    MAX_NUM = 50
    gt_bbox2 = np.zeros((MAX_NUM, 4))
    gt_class2 = np.zeros((MAX_NUM,))
    for i in range(len(gt_bbox)):
        gt_bbox2[i, :] = gt_bbox[i, :]
        gt_class2[i] = gt_class[i]
        if i >= MAX_NUM:
            break
    return gt_bbox2, gt_class2


# 根据record里面保存的信息，获取单张图片及图片里面的物体真实框和标签等信息
def get_img_data_from_file(record):
    """
    record is a dict as following,
      record = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            12a
            'difficult': difficult
            }
    """
    im_file = record["im_file"]
    # print(im_file)
    h = record["h"]
    w = record["w"]
    is_crowd = record["is_crowd"]
    gt_class = record["gt_class"]
    gt_bbox = record["gt_bbox"]
    difficult = record["difficult"]

    img = cv2.imread(im_file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#图片转换

    # check if h and w in record equals that read from img
    assert img.shape[0] == int(h), "image height of {} inconsistent in record({}) and img file({})".format(
        im_file, h, img.shape[0]
    )

    assert img.shape[1] == int(w), "image width of {} inconsistent in record({}) and img file({})".format(
        im_file, w, img.shape[1]
    )

    gt_boxes, gt_labels = get_bbox(gt_bbox, gt_class)

    # gt_bbox 用相对值
    gt_boxes[:, 0] = gt_boxes[:, 0] / float(w)
    gt_boxes[:, 1] = gt_boxes[:, 1] / float(h)
    gt_boxes[:, 2] = gt_boxes[:, 2] / float(w)
    gt_boxes[:, 3] = gt_boxes[:, 3] / float(h)

    return img, gt_boxes, gt_labels, (h, w)


# 读取图片信息，并且对图片做图像增广，调用image_augment完成
def get_img_data(record, size):
    img, gt_boxes, gt_labels, scales = get_img_data_from_file(record)  # 从文件读入图像和标注信息
    img, gt_boxes, gt_labels = image_augment(img, gt_boxes, gt_labels, size)  # 图像增广只做了改变亮度，随即缩放
    # mean = [0.485, 0.456, 0.406] #均值方差
    # std = [0.229, 0.224, 0.225]
    mean = [0.45844197, 0.48597776, 0.52515776]  # 此处选用了“白鱼”提供的计算均值、方差的算法计算出来的结果
    std = [0.20162111, 0.20163414, 0.2009494]  # 同上
    mean = np.array(mean).reshape((1, 1, -1))
    std = np.array(std).reshape((1, 1, -1))
    img = (img / 255.0 - mean) / std  # 数据标准化
    img = img.astype("float32").transpose((2, 0, 1))  # 将数据维度从[H, W, C] 转为[C, H, W]
    return img, gt_boxes, gt_labels, scales


# 获取图片缩放的尺寸
def get_img_size(mode):
    if (mode == "train") or (mode == "valid"):  # 训练：随机产生尺寸
        inds = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ii = np.random.choice(inds)  # 测试：固定尺寸608
        img_size = 320 + ii * 32
    else:
        img_size = 608
    return img_size


# 将 list形式的batch数据 转化成多个array构成的tuple
def make_array(batch_data):
    img_array = np.array([item[0] for item in batch_data], dtype="float32")
    gt_box_array = np.array([item[1] for item in batch_data], dtype="float32")
    gt_labels_array = np.array([item[2] for item in batch_data], dtype="int32")
    img_scale = np.array([item[3] for item in batch_data], dtype="int32")
    return img_array, gt_box_array, gt_labels_array, img_scale


# 定义数据读取类，继承Paddle.io.Dataset
class TrainDataset(paddle.io.Dataset):
    def __init__(self, datadir, mode="train"):
        self.datadir = datadir
        cname2cid = get_insect_names()
        self.records = get_annotations(cname2cid, datadir)
        self.img_size = get_img_size(mode)

    def __getitem__(self, idx):
        record = self.records[idx]
        img, gt_bbox, gt_labels, im_shape = get_img_data(record, size=self.img_size)

        return img, gt_bbox, gt_labels, np.array(im_shape)

    def __len__(self):
        return len(self.records)


# 将 list形式的测试batch数据 转化成多个array构成的tuple
def make_test_array(batch_data):
    img_name_array = np.array([item[0] for item in batch_data])
    img_data_array = np.array([item[1] for item in batch_data], dtype="float32")
    img_scale_array = np.array([item[2] for item in batch_data], dtype="int32")
    return img_name_array, img_data_array, img_scale_array


# 测试数据读取
def test_data_loader(datadir, batch_size=10, test_image_size=608, mode="test"):
    """
    加载测试用的图片，测试数据没有groundtruth标签
    """
    image_names = os.listdir(datadir)

    def reader():
        batch_data = []
        img_size = test_image_size
        for image_name in image_names:
            file_path = os.path.join(datadir, image_name)
            img = cv2.imread(file_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H = img.shape[0]
            W = img.shape[1]
            img = cv2.resize(img, (img_size, img_size))

            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            mean = [0.45844197, 0.48597776, 0.52515776]  # 同上注释
            std = [0.20162111, 0.20163414, 0.2009494]  # 同上注释
            mean = np.array(mean).reshape((1, 1, -1))
            std = np.array(std).reshape((1, 1, -1))
            out_img = (img / 255.0 - mean) / std
            out_img = out_img.astype("float32").transpose((2, 0, 1))
            img = out_img  # np.transpose(out_img, (2,0,1))
            im_shape = [H, W]

            batch_data.append((image_name.split(".")[0], img, im_shape))
            if len(batch_data) == batch_size:
                yield make_test_array(batch_data)
                batch_data = []
        if len(batch_data) > 0:
            yield make_test_array(batch_data)

    return reader


# 批量读取数据，同一批次内图像的尺寸大小必须是一样的
# 不同批次之间的大小是随机的
# 由上面定义的get_img_size函数产生
def data_loader(datadir, batch_size=10, mode="train"):
    cname2cid = get_insect_names()
    records = get_annotations(cname2cid, datadir)

    def reader():
        if mode == "train":
            np.random.shuffle(records)
        batch_data = []
        img_size = get_img_size(mode)
        for record in records:
            img, gt_bbox, gt_labels, im_shape = get_img_data(record, size=img_size)
            batch_data.append((img, gt_bbox, gt_labels, im_shape))
            if len(batch_data) == batch_size:
                yield make_array(batch_data)
                batch_data = []
                img_size = get_img_size(mode)
        if len(batch_data) > 0:
            yield make_array(batch_data)

    return reader


def multithread_loader(datadir, batch_size=10, mode="train"):
    cname2cid = get_insect_names()
    records = get_annotations(cname2cid, datadir)

    def reader():
        if mode == "train":
            np.random.shuffle(records)
        batch_data = []
        img_size = get_img_size(mode)
        for record in records:
            img, gt_bbox, gt_labels, im_shape = get_img_data(record, size=img_size)
            batch_data.append((img, gt_bbox, gt_labels, im_shape))
            if len(batch_data) == batch_size:
                yield make_array(batch_data)
                batch_data = []
                img_size = get_img_size(mode)
        if len(batch_data) > 0:
            yield make_array(batch_data)

    def get_data(samples):
        batch_data = []
        for sample in samples:
            record = sample[0]
            img_size = sample[1]
            img, gt_bbox, gt_labels, im_shape = get_img_data(record, size=img_size)

            batch_data.append((img, gt_bbox, gt_labels, im_shape))
        return make_array(batch_data)

    mapper = functools.partial(
        get_data,
    )
    return paddle.reader.xmap_readers(mapper, reader, 8, 10)


# 读取单张测试图片
def single_image_data_loader(filename, test_image_size=608, mode="test"):
    """
    加载测试用的图片，测试数据没有groundtruth标签
    """
    batch_size = 1

    def reader():
        batch_data = []
        img_size = test_image_size
        file_path = os.path.join(filename)
        img = cv2.imread(file_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H = img.shape[0]
        W = img.shape[1]
        img = cv2.resize(img, (img_size, img_size))

        mean = [0.45844197, 0.48597776, 0.52515776]
        std = [0.20162111, 0.20163414, 0.2009494]
        mean = np.array(mean).reshape((1, 1, -1))
        std = np.array(std).reshape((1, 1, -1))
        out_img = (img / 255.0 - mean) / std
        out_img = out_img.astype("float32").transpose((2, 0, 1))
        img = out_img  # np.transpose(out_img, (2,0,1))
        im_shape = [H, W]

        batch_data.append((filename.split(".")[0], img, im_shape))
        if len(batch_data) == batch_size:
            yield make_test_array(batch_data)
            batch_data = []

    return reader


if __name__ == "__main__":
    TRAINDIR = "./dataset/VOC"
    # train_dataset = TrainDataset(TRAINDIR, mode='train')
    d = data_loader(TRAINDIR, batch_size=10, mode="train")
    # d = multithread_loader(train_dataset, batch_size=10, mode='train')
    for i, data in enumerate(d()):
        img, gt_boxes, gt_labels, scales = data
        print(img.shape, gt_boxes.shape, gt_labels.shape)
