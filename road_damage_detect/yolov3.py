# -*- coding: utf-8 -*-

import numpy as np
import time

import paddle

from darknet import ConvBNLayer, DarkNet53_conv_body
from anchor_lables import get_objectness_label
from reader import data_loader


# 定义生成YOLO-V3预测输出的模块
# 也就是教程中图x所示的由ci生成ri和ti的过程
# 从骨干网络输出特征图C0得到跟预测相关的特征图P0
class YoloDetectionBlock(paddle.nn.Layer):
    # define YOLOv3 detection head
    # 使用多层卷积和BN提取特征
    def __init__(self, ch_in, ch_out, is_test=True):
        super(YoloDetectionBlock, self).__init__()

        assert ch_out % 2 == 0, "channel {} cannot be divided by 2".format(ch_out)

        self.conv0 = ConvBNLayer(ch_in=ch_in, ch_out=ch_out, kernel_size=1, stride=1, padding=0)
        self.conv1 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out * 2, kernel_size=3, stride=1, padding=1)
        self.conv2 = ConvBNLayer(ch_in=ch_out * 2, ch_out=ch_out, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBNLayer(ch_in=ch_out, ch_out=ch_out * 2, kernel_size=3, stride=1, padding=1)
        self.route = ConvBNLayer(ch_in=ch_out * 2, ch_out=ch_out, kernel_size=1, stride=1, padding=0)
        self.tip = ConvBNLayer(ch_in=ch_out, ch_out=ch_out * 2, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        out = self.conv0(inputs)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        route = self.route(out)
        tip = self.tip(route)
        return route, tip


# 定义上采样模块
class Upsample(paddle.nn.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = paddle.shape(inputs)
        shape_hw = paddle.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = paddle.cast(shape_hw, dtype="int32")
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = paddle.nn.functional.interpolate(x=inputs, scale_factor=self.scale, mode="NEAREST")
        return out


# 定义YOLOv3模型
class YOLOv3(paddle.nn.Layer):
    def __init__(self, num_classes=8):
        super(YOLOv3, self).__init__()

        self.num_classes = num_classes
        # 提取图像特征的骨干代码
        self.block = DarkNet53_conv_body()
        self.block_outputs = []
        self.yolo_blocks = []
        self.route_blocks_2 = []
        # 生成3个层级的特征图P0, P1, P2
        for i in range(3):
            # 添加从ci生成ri和ti的模块
            yolo_block = self.add_sublayer(
                "yolo_detecton_block_%d" % (i),
                YoloDetectionBlock(
                    ch_in=512 // (2**i) * 2 if i == 0 else 512 // (2**i) * 2 + 512 // (2**i),
                    ch_out=512 // (2**i),
                ),
            )
            self.yolo_blocks.append(yolo_block)

            num_filters = 3 * (self.num_classes + 5)

            # 添加从ti生成pi的模块，这是一个Conv2D操作，输出通道数为3 * (num_classes + 5)
            block_out = self.add_sublayer(
                "block_out_%d" % (i),
                paddle.nn.Conv2D(
                    in_channels=512 // (2**i) * 2,
                    out_channels=num_filters,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    weight_attr=paddle.ParamAttr(initializer=paddle.nn.initializer.Normal(0.0, 0.02)),
                    bias_attr=paddle.ParamAttr(
                        initializer=paddle.nn.initializer.Constant(0.0), regularizer=paddle.regularizer.L2Decay(0.0)
                    ),
                ),
            )
            self.block_outputs.append(block_out)
            if i < 2:
                # 对ri进行卷积
                route = self.add_sublayer(
                    "route2_%d" % i,
                    ConvBNLayer(ch_in=512 // (2**i), ch_out=256 // (2**i), kernel_size=1, stride=1, padding=0),
                )
                self.route_blocks_2.append(route)
            # 将ri放大以便跟c_{i+1}保持同样的尺寸
            self.upsample = Upsample()

    def forward(self, inputs):
        outputs = []
        blocks = self.block(inputs)
        for i, block in enumerate(blocks):
            if i > 0:
                # 将r_{i-1}经过卷积和上采样之后得到特征图，与这一级的ci进行拼接
                block = paddle.concat([route, block], axis=1)
            # 从ci生成ti和ri
            route, tip = self.yolo_blocks[i](block)
            # 从ti生成pi
            block_out = self.block_outputs[i](tip)
            # 将pi放入列表
            outputs.append(block_out)

            if i < 2:
                # 对ri进行卷积调整通道数
                route = self.route_blocks_2[i](route)
                # 对ri进行放大，使其尺寸和c_{i+1}保持一致
                route = self.upsample(route)

        return outputs

    def get_loss(
        self,
        outputs,
        gtbox,
        gtlabel,
        gtscore=None,
        anchors=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
        anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        ignore_thresh=0.7,
        use_label_smooth=False,
    ):
        """
        使用paddle.vision.ops.yolo_loss，直接计算损失函数，过程更简洁，速度也更快
        """
        self.losses = []
        downsample = 32
        for i, out in enumerate(outputs):  # 对三个层级分别求损失函数
            anchor_mask_i = anchor_masks[i]
            loss = paddle.vision.ops.yolo_loss(
                x=out,  # out是P0, P1, P2中的一个
                gt_box=gtbox,  # 真实框坐标
                gt_label=gtlabel,  # 真实框类别
                gt_score=gtscore,  # 真实框得分，使用mixup训练技巧时需要，不使用该技巧时直接设置为1，形状与gtlabel相同
                anchors=anchors,  # 锚框尺寸，包含[w0, h0, w1, h1, ..., w8, h8]共9个锚框的尺寸
                anchor_mask=anchor_mask_i,  # 筛选锚框的mask，例如anchor_mask_i=[3, 4, 5]，将anchors中第3、4、5个锚框挑选出来给该层级使用
                class_num=self.num_classes,  # 分类类别数
                ignore_thresh=ignore_thresh,  # 当预测框与真实框IoU > ignore_thresh，标注objectness = -1
                downsample_ratio=downsample,  # 特征图相对于原图缩小的倍数，例如P0是32， P1是16，P2是8
                use_label_smooth=False,
            )  # 使用label_smooth训练技巧时会用到，这里没用此技巧，直接设置为False
            self.losses.append(paddle.mean(loss))  # mean对每张图片求和
            downsample = downsample // 2  # 下一级特征图的缩放倍数会减半
        return sum(self.losses)  # 对每个层级求和

    def get_pred(
        self,
        outputs,
        im_shape=None,
        anchors=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
        anchor_masks=[[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        valid_thresh=0.01,
    ):
        downsample = 32
        total_boxes = []
        total_scores = []
        for i, out in enumerate(outputs):
            anchor_mask = anchor_masks[i]
            anchors_this_level = []
            for m in anchor_mask:
                anchors_this_level.append(anchors[2 * m])
                anchors_this_level.append(anchors[2 * m + 1])

            boxes, scores = paddle.vision.ops.yolo_box(
                x=out,
                img_size=im_shape,
                anchors=anchors_this_level,
                class_num=self.num_classes,
                conf_thresh=valid_thresh,
                downsample_ratio=downsample,
                name="yolo_box" + str(i),
            )
            total_boxes.append(boxes)
            total_scores.append(paddle.transpose(scores, perm=[0, 2, 1]))
            downsample = downsample // 2

        yolo_boxes = paddle.concat(total_boxes, axis=1)
        yolo_scores = paddle.concat(total_scores, axis=2)
        return yolo_boxes, yolo_scores


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_yolo_box_xxyy(pred, anchors, num_classes, downsample):
    batchsize = pred.shape[0]
    num_rows = pred.shape[-2]
    num_cols = pred.shape[-1]
    input_h = num_rows * downsample
    input_w = num_cols * downsample
    num_anchors = len(anchors) // 2
    pred = pred.reshape([-1, num_anchors, num_classes + 5, num_rows, num_cols])
    pred_location = pred[:, :, 0:4, :, :]  # 计算出预测框坐标[tx,ty,th,tw]
    # 将[tx,ty,th,tw]转换成预测框坐标[x1,y1,x2,y2]
    pred_location = np.transpose(pred_location, (0, 3, 4, 1, 2))
    anchors_this = []
    for ind in range(num_anchors):
        anchors_this.append([anchors[ind * 2], anchors[ind * 2 + 1]])
    anchors_this = np.array(anchors_this).astype("float32")

    pred_box = np.zeros(pred_location.shape)
    for n in range(batchsize):
        for i in range(num_rows):
            for j in range(num_cols):
                for k in range(num_anchors):
                    pred_box[n, i, j, k, 0] = j
                    pred_box[n, i, j, k, 1] = i
                    pred_box[n, i, j, k, 2] = anchors_this[k][0]
                    pred_box[n, i, j, k, 3] = anchors_this[k][1]
    pred_box[:, :, :, :, 0] = (sigmoid(pred_location[:, :, :, :, 0]) + pred_box[:, :, :, :, 0]) / num_cols
    pred_box[:, :, :, :, 1] = (sigmoid(pred_location[:, :, :, :, 1]) + pred_box[:, :, :, :, 1]) / num_rows
    pred_box[:, :, :, :, 2] = np.exp(pred_location[:, :, :, :, 2]) * pred_box[:, :, :, :, 2] / input_w
    pred_box[:, :, :, :, 3] = np.exp(pred_location[:, :, :, :, 3]) * pred_box[:, :, :, :, 3] / input_h

    pred_box[:, :, :, :, 0] = pred_box[:, :, :, :, 0] - pred_box[:, :, :, :, 2] / 2.0
    pred_box[:, :, :, :, 1] = pred_box[:, :, :, :, 1] - pred_box[:, :, :, :, 3] / 2.0
    pred_box[:, :, :, :, 2] = pred_box[:, :, :, :, 0] + pred_box[:, :, :, :, 2]
    pred_box[:, :, :, :, 3] = pred_box[:, :, :, :, 1] + pred_box[:, :, :, :, 3]

    pred_box = np.clip(pred_box, 0, 1.0)
    return pred_box


def get_iou_above_thresh_inds(pred_box, gt_boxes, iou_threshold):
    batchsize = pred_box.shape[0]
    num_rows = pred_box.shape[1]
    num_cols = pred_box.shape[2]
    num_anchors = pred_box.shape[3]
    ret_inds = np.zeros([batchsize, num_rows, num_cols, num_anchors])
    # 对图片做循环
    for i in range(batchsize):
        pred_box_i = pred_box[i]
        gt_boxes_i = gt_boxes[i]
        #         print("pre_box_i",pre_box_i.shape,"gt_box_i",gt_boxes_i.shape)
        # 对真实框做循环，转换为xyxy形式
        for k in range(len(gt_boxes_i)):
            gt = gt_boxes_i[k]
            gtx_min = gt[0] - gt[2] / 2.0
            gty_min = gt[1] - gt[3] / 2.0
            gtx_max = gt[0] + gt[2] / 2.0
            gty_max = gt[1] + gt[3] / 2.0
            if (gtx_max - gtx_min < 1e-3) or (gty_max - gty_min < 1e-3):  # 跳过无效的真实框
                continue
            x1 = np.maximum(pred_box_i[:, :, :, 0], gtx_min)
            y1 = np.maximum(pred_box_i[:, :, :, 1], gty_min)
            x2 = np.minimum(pred_box_i[:, :, :, 2], gtx_max)
            y2 = np.minimum(pred_box_i[:, :, :, 3], gty_max)  # 计算预测框跟真实框的交叠区域坐标，一次性计算所有预测框
            intersection = np.maximum(x2 - x1, 0.0) * np.maximum(y2 - y1, 0.0)
            s1 = (gty_max - gty_min) * (gtx_max - gtx_min)
            s2 = (pred_box_i[:, :, :, 2] - pred_box_i[:, :, :, 0]) * (
                pred_box_i[:, :, :, 3] - pred_box_i[:, :, :, 1]
            )  # 计算预测框的面积
            union = s2 + s1 - intersection
            iou = intersection / union
            above_inds = np.where(iou > iou_threshold)  # 选出IOU超过阈值的预测框
            ret_inds[i][above_inds] = 1  # 将IOU超过阈值的预测框对应编号信息设置为1
    ret_inds = np.transpose(ret_inds, (0, 3, 1, 2))
    return ret_inds.astype("bool")  # ret_inds数值为True或False，超过阈值的地方为True


# 负样本中IOU大于阈值的预测框对应的objectness标记为-1
def label_objectness_ignore(labe_objectness, iou_above_thresh_indices):
    negative_indices = label_objectness < 0.5
    ignore_indices = negative_indices * iou_above_thresh_indices
    label_objectness[ignore_indices] = -1
    return label_objectness


# 分别建立表征是否包含目标物体的损失函数、表征物体位置的损失函数、表征物体类别的损失函数
def get_loss_self(output, label_objectness, label_location, label_classification, scales, num_anchors=3, num_classes=7):
    reshaped_output = paddle.fluid.layers.reshape(
        output, [-1, num_anchors, num_classes + 5, output.shape[2], output.shape[3]]
    )
    pred_objectness = reshaped_output[:, :, 4, :, :]
    # 是否包含目标物体的损失函数
    loss_objectness = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(
        pred_objectness, label_objectness, ignore_index=-1
    )

    pos_objectness = label_objectness > 0  # pos_sample 里面只有在对应正样本编号的地方为1
    pos_samples = paddle.fluid.layers.cast(pos_objectness, "float32")
    pos_samples.stop_gradient = True

    # 表征物体位置的损失函数，通过pred_location 和 label_location计算
    tx = reshaped_output[:, :, 0, :, :]
    ty = reshaped_output[:, :, 1, :, :]
    tw = reshaped_output[:, :, 2, :, :]
    th = reshaped_output[:, :, 3, :, :]

    dx_label = label_location[:, :, 0, :, :]
    dy_label = label_location[:, :, 1, :, :]
    tw_label = label_location[:, :, 2, :, :]
    th_label = label_location[:, :, 3, :, :]

    loss_location_x = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(tx, dx_label)
    loss_location_y = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(ty, dy_label)
    loss_location_w = paddle.fluid.layers.abs(tw - tw_label)
    loss_location_h = paddle.fluid.layers.abs(th - th_label)  # 计算x,y,w,h四个位置的损失函数
    loss_location = loss_location_x + loss_location_y + loss_location_w + loss_location_h
    loss_location = loss_location * scales  # 只计算正样本的位置损失函数
    loss_location = loss_location * pos_samples

    pred_classification = reshaped_output[:, :, 5 : 5 + num_classes, :, :]
    loss_classification = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(
        pred_classification, label_classification
    )
    loss_classification = paddle.fluid.layers.reduce_sum(loss_classification, dim=2, keep_dim=False)
    loss_classification = loss_classification * pos_samples

    total_loss = loss_objectness + loss_location + loss_classification
    total_loss = paddle.fluid.layers.reduce_sum(total_loss, dim=[1, 2, 3], keep_dim=False)
    total_loss = paddle.fluid.layers.reduce_mean(total_loss)
    return total_loss


if __name__ == "__main__":
    TRAINDIR = "dataset/VOC"
    anchors = [116, 90, 156, 198, 373, 326]  # p0的锚框
    num_classes = 8
    downsample = 32
    iou_threshold = 0.7
    # train_dataset = TrainDataset(TRAINDIR, mode='train')
    reader = data_loader(TRAINDIR, batch_size=10, mode="train")
    # d = multithread_loader(train_dataset, batch_size=10, mode='train')

    for i, data in enumerate(reader()):
        img, gt_boxes, gt_labels, scales = data
        label_objectness, label_location, label_classification, scale_location = get_objectness_label(
            img, gt_boxes, gt_labels, iou_threshold, anchors, num_classes, downsample
        )

        # anchors = [116, 90, 156, 198, 373, 326]
        num_anchors = len(anchors) // 2

        num_filters = num_anchors * (num_classes + 5)
        backbone = DarkNet53_conv_body()
        detection = YoloDetectionBlock(ch_in=1024, ch_out=512)
        conv2d_pred = paddle.nn.Conv2D(in_channels=1024, out_channels=num_filters, kernel_size=1)
        # x = np.random.randn(1, 3, 1400, 1100).astype('float32')#一张图片，三个通道，大小1400*1100
        # x = paddle.to_tensor(x)
        x = paddle.to_tensor(img)
        c0, c1, c2 = backbone(x)  # c0步幅为32，c1步幅为16，c2步幅为8，为输入图片尺寸除以输出图片尺寸
        route, tip = detection(c0)
        p0 = conv2d_pred(tip)  # 打印出来是39通道，0-12（4+8）共13通道，前4个跟位置关联，第5个跟obj关联，后边8个跟类别关联，标注第一个预测框类别、obj、位置信息，依次跟第k个预测框相关联
        reshaped_p0 = paddle.reshape(p0, [-1, num_anchors, num_classes + 5, p0.shape[2], p0.shape[3]])
        pred_objectness = reshaped_p0[:, :, 4, :, :]  # 取出obj相关的值
        pred_objectness_probability = paddle.nn.functional.sigmoid(pred_objectness)
        pred_classification = reshaped_p0[:, :, 5 : 5 + num_classes, :, :]  # 计算物体属于每个类别
        pred_classification_probability = paddle.nn.functional.sigmoid(pred_classification)  # 计算物体属于每个类别的概率
        pred_location = reshaped_p0[:, :, 0:4, :, :]  # 计算出预测框坐标[tx,ty,th,tw]
        # 将[tx,ty,th,tw]转换成预测框坐标[x1,y1,x2,y2]
        pred_location = np.transpose(pred_location, (0, 3, 4, 1, 2))

        pred = p0.numpy()  # 注意这里要转换
        pred_box = get_yolo_box_xxyy(pred, anchors, num_classes, downsample)  # 计算预测框
        iou_above_thresh_indices = get_iou_above_thresh_inds(pred_box, gt_boxes, iou_threshold=iou_threshold)  # 计算IOU
        label_objectness = label_objectness_ignore(
            label_objectness, iou_above_thresh_indices
        )  # 负样本中IOU大于阈值的预测框对应的objectness标记为-1

        label_objectness = paddle.to_tensor(label_objectness)
        label_location = paddle.to_tensor(label_location)
        label_classification = paddle.to_tensor(label_classification)
        scales = paddle.to_tensor(scale_location)

        label_objectness.stop_gradient = True
        label_location.stop_gradient = True
        label_classification.stop_gradient = True
        scales.stop_gradient = True

        total_loss = get_loss_self(
            p0,
            label_objectness,
            label_location,
            label_classification,
            scales,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

        total_loss_data = total_loss.numpy()

        print(total_loss_data)
