import json
import os, numpy
import paddle
from reader import test_data_loader
from yolov3 import YOLOv3
from multinms import multiclass_nms

ANCHORS = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
ANCHOR_MASKS = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
VALID_THRESH = 0.01
NMS_TOPK = 400
NMS_POSK = 100
NMS_THRESH = 0.45
NUM_CLASSES = 8

if __name__ == '__main__':
    TRAINDIR = './dataset/VOC/JPEGImages'
    TESTDIR = './dataset/VOC/JPEGImages'
    # VALIDDIR = '/home/aistudio/work/insects/val'

    model = YOLOv3(num_classes=NUM_CLASSES)
    params_file_path = './yolo_epoch199.pdparams'
    model_state_dict = paddle.load(params_file_path)
    model.load_dict(model_state_dict)
    model.eval()

    total_results = []
    test_loader = test_data_loader(TESTDIR, batch_size= 1, mode='test')
    for i, data in enumerate(test_loader()):
        img_name, img_data, img_scale_data = data
        img = paddle.to_tensor(img_data)
        img_scale = paddle.to_tensor(img_scale_data)

        outputs = model.forward(img)
        bboxes, scores = model.get_pred(outputs,
                                 im_shape=img_scale,
                                 anchors=ANCHORS,
                                 anchor_masks=ANCHOR_MASKS,
                                 valid_thresh = VALID_THRESH)

        bboxes_data = bboxes.numpy()
        scores_data = scores.numpy()
        result = multiclass_nms(bboxes_data, scores_data,
                      score_thresh=VALID_THRESH, 
                      nms_thresh=NMS_THRESH, 
                      pre_nms_topk=NMS_TOPK, 
                      pos_nms_topk=NMS_POSK)

        for j in range(len(result)):
            result_j = result[j]
            img_name_j = img_name[j]
            if result_j != []:
                total_results.append([img_name_j, result_j.tolist()])
            else:
                total_results.append([img_name_j, [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        print('processed {} pictures'.format(len(total_results)))

    print('测试')
    json.dump(total_results, open('.output/pred_results.json', 'w'))
