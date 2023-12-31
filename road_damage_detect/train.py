# -*- coding: utf-8 -*-

import time
import os
import numpy as np
import paddle
from reader import TrainDataset, data_loader, multithread_loader
from yolov3 import YOLOv3

# train.py
# 提升点： 可以改变anchor的大小，注意训练和测试时要使用同样的anchor
ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

IGNORE_THRESH = 0.7
NUM_CLASSES = 8


# 定义学习率
def get_lr(base_lr=0.0001, lr_decay=0.1):
    bd = [10000, 20000]
    lr = [base_lr, base_lr * lr_decay, base_lr * lr_decay * lr_decay]
    learning_rate = paddle.optimizer.lr.PiecewiseDecay(boundaries=bd, values=lr)
    return learning_rate


if __name__ == "__main__":
    TRAINDIR = "./dataset/VOC"
    # TESTDIR = 'dataset/test_A'
    # VALIDDIR = 'dataset/test_A'
    # paddle.set_device("gpu:0")
    # 创建数据读取类
    # train_dataset = TrainDataset(TRAINDIR, mode='train')
    # valid_dataset = TrainDataset(VALIDDIR, mode='valid')
    # 使用paddle.io.DataLoader创建数据读取器，并设置batchsize，进程数量num_workers等参数
    # train_loader = paddle.io.DataLoader(train_dataset, batch_size=30, shuffle=True, num_workers=8, drop_last=True, use_shared_memory=False)
    # valid_loader = paddle.io.DataLoader(valid_dataset, batch_size=10, shuffle=False, num_workers=0, drop_last=False, use_shared_memory=False)

    # 单线程
    train_loader = data_loader(TRAINDIR, batch_size=10, mode="train")

    # 多线程
    # train_loader = multithread_loader(TRAINDIR, batch_size=10, mode='train')

    # 定义网络
    model = YOLOv3(num_classes=NUM_CLASSES)  # 创建模型

    # 定义学习率
    learning_rate = get_lr()

    # 定义优化器
    opt = paddle.optimizer.Momentum(
        learning_rate=learning_rate,
        momentum=0.9,
        weight_decay=paddle.regularizer.L2Decay(0.0005),
        parameters=model.parameters(),
    )  # 创建优化器

    #
    # opt = paddle.optimizer.Adam(learning_rate=learning_rate, weight_decay=paddle.regularizer.L2Decay(0.0005), parameters=model.parameters())

    MAX_EPOCH = 200
    for epoch in range(MAX_EPOCH):
        for i, data in enumerate(train_loader()):
            img, gt_boxes, gt_labels, img_scale = data
            gt_scores = np.ones(gt_labels.shape).astype("float32")
            gt_scores = paddle.to_tensor(gt_scores)
            img = paddle.to_tensor(img)
            # print(img)
            gt_boxes = paddle.to_tensor(gt_boxes)
            gt_labels = paddle.to_tensor(gt_labels)
            outputs = model(img)  # 前向传播，输出[P0, P1, P2]
            loss = model.get_loss(
                outputs,
                gt_boxes,
                gt_labels,
                gtscore=gt_scores,
                anchors=ANCHORS,
                anchor_masks=ANCHOR_MASKS,
                ignore_thresh=IGNORE_THRESH,
                use_label_smooth=False,
            )  # 计算损失函数

            loss.backward()  # 反向传播计算梯度
            opt.step()  # 更新参数
            opt.clear_grad()
            if i % 50 == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                print("{}[TRAIN]epoch {}, iter {}, output loss: {}".format(timestring, epoch, i, loss.numpy()))

        # save params of model
        if (epoch % 20 == 0) or (epoch == MAX_EPOCH - 1):
            paddle.save(model.state_dict(), "yolo_epoch{}".format(epoch))

        # 每个epoch结束之后在验证集上进行测试
        # model.eval()
        # for i, data in enumerate(valid_loader()):#
        #     img, gt_boxes, gt_labels, img_scale = data
        #     gt_scores = np.ones(gt_labels.shape).astype('float32')
        #     gt_scores = paddle.to_tensor(gt_scores)
        #     img = paddle.to_tensor(img)
        #     gt_boxes = paddle.to_tensor(gt_boxes)
        #     gt_labels = paddle.to_tensor(gt_labels)
        #     outputs = model(img)
        #     loss = model.get_loss(outputs, gt_boxes, gt_labels, gtscore=gt_scores,
        #                           anchors = ANCHORS,
        #                           anchor_masks = ANCHOR_MASKS,
        #                           ignore_thresh=IGNORE_THRESH,
        #                           use_label_smooth=False)
        # if i % 1 == 0:
        #     timestring = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))
        #     print('{}[VALID]epoch {}, iter {}, output loss: {}'.format(timestring, epoch, i, loss.numpy()))
        model.train()
    paddle.save(model.state_dict(), "./best/model.pdparams")
