# -*- coding: utf-8 -*-

# 此文件中包含数据集读取相关的函数

import os
import numpy as np
import xml.etree.ElementTree as ET #读取XML

# 名称列表
INSECT_NAMES = ['Crack', 'Manhole', 'Net', 
                'Pothole', 'Patch-Crack', 'Patch-Net', 'Patch-Pothole', 'Other']

# 名字到数字类别的映射关系
def get_insect_names():
    """
    return a dict, as following,
    It can map the insect name into an integer label.
    """
    insect_category2id = {}
    for i, item in enumerate(INSECT_NAMES):
        insect_category2id[item] = i

    return insect_category2id


# 获取标注信息
def get_annotations(cname2cid, datadir):
    filenames = os.listdir(os.path.join(datadir, 'Annotations'))
    records = []
    ct = 0
    for fname in filenames:
        fid = fname.split('.')[0]
        fpath = os.path.join(datadir, 'Annotations', fname)
        img_file = os.path.join(datadir, 'JPEGImages', fid+'.jpg')
        # img_file =  datadir + "/" + "JPEGimages" + "/" + fid + '.jpg'
        tree = ET.parse(fpath)

        if tree.find('id') is None:
            im_id = np.array([ct])
        else:
            im_id = np.array([int(tree.find('id').text)])

        objs = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
        gt_class = np.zeros((len(objs), ), dtype=np.int32)
        is_crowd = np.zeros((len(objs), ), dtype=np.int32)
        difficult = np.zeros((len(objs), ), dtype=np.int32)
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            gt_class[i] = cname2cid[cname]    #获取名称并转换为数字类型
            _difficult = int(obj.find('difficult').text)
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            # 这里使用xywh格式来表示目标物体真实框
            gt_bbox[i] = [(x1+x2)/2.0 , (y1+y2)/2.0, x2-x1+1., y2-y1+1.]
            is_crowd[i] = 0
            difficult[i] = _difficult

        voc_rec = {
            'im_file': img_file,
            'im_id': im_id,
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
            }
        if len(objs) != 0:
            records.append(voc_rec)
        ct += 1
    return records

if __name__ == '__main__':
    datadir = "./dataset/VOC"
    cname2cid = get_insect_names()
    records = get_annotations(cname2cid, datadir)
    print(records[100])



