# -*- coding: utf-8 -*-

import cv2
import numpy as np

import paddle
from reader import single_image_data_loader
from multinms import multiclass_nms
from yolov3 import YOLOv3
from draw_results import draw_results, visualize_results

import argparse


def parse_args():
    parser = argparse.ArgumentParser("Evaluation Parameters")
    parser.add_argument(
        "-i",
        "--image_name",
        type=str,
        default="./dataset/test_A/images/00100.jpg",
        help="the directory of test images",
    )
    parser.add_argument(
        "-m",
        "--weight_file",
        type=str,
        default="./yolo_epoch199.pdparams",
        help="the path of model parameters",
    )
    args = parser.parse_args()
    return args


ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

VALID_THRESH = 0.01

NMS_TOPK = 400
NMS_POSK = 100
NMS_THRESH = 0.45

NUM_CLASSES = 8


if __name__ == "__main__":
    args = parse_args()
    IMAGE_NAME = args.image_name
    WEIGHT_FILE = args.weight_file
    # print(WEIGHT_FILE)
    # print(IMAGE_NAME)

    model = YOLOv3(num_classes=NUM_CLASSES)
    model_state_dict = paddle.load(WEIGHT_FILE)
    model.load_dict(model_state_dict)
    model.eval()

    total_results = []
    test_loader = single_image_data_loader(IMAGE_NAME, mode="test")
    for i, data in enumerate(test_loader()):
        img_name, img_data, img_scale_data = data
        img = paddle.to_tensor(img_data)
        img_scale = paddle.to_tensor(img_scale_data)

        outputs = model.forward(img)
        bboxes, scores = model.get_pred(
            outputs, im_shape=img_scale, anchors=ANCHORS, anchor_masks=ANCHOR_MASKS, valid_thresh=VALID_THRESH
        )

        bboxes_data = bboxes.numpy()
        scores_data = scores.numpy()
        results = multiclass_nms(
            bboxes_data,
            scores_data,
            score_thresh=VALID_THRESH,
            nms_thresh=NMS_THRESH,
            pre_nms_topk=NMS_TOPK,
            pos_nms_topk=NMS_POSK,
        )

    result = results[0]
    print(f"{results= }")
    # draw_results(result, IMAGE_NAME, draw_thresh=0.10)
    print(f"{img_data.shape= }")
    # print(f"{img_data= }")

    # image = np.zeros((512, 512, 3), np.uint8)
    # # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(image, (0, 0), (511, 511), (255, 0, 0), 5)

    # image = np.squeeze(img_data, axis=0).transpose((2, 1, 0))
    image = cv2.imread(IMAGE_NAME)
    print(f"{image.shape= }")
    print(f"{image= }")
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR, cv2.CV_8UC3)
    # image = (image * 255).astype(np.uint8)
    # print(f"{image= }")
    image = visualize_results(results, np.array(image), draw_thresh=0.10)
    cv2.imshow("results", image)
    # cv2.pollKey()
    cv2.waitKey(0)
