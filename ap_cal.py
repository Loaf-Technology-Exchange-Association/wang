import numpy as np
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
import mmcv
import torch
import os
import json
from datasets.select_perfect_sample import compute_iou
from datasets.select_perfect_sample import topleftxywh_to_xyxy

IOU_THR = 0.5
MAP_THR = 0.7


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    ground truth label (np array),true positive为1,false positive为0
        conf:  Objectness value from 0-1 (np array).
        pred_cls: Predicted object classes (np array).
        target_cls: True object classes (np array).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()  # 累加和列表
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(precision, recall):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (np.array).
        precision: The precision curve (np.array).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[:-1] != mrec[1:])[0]  # 错位比较，前一个元素与其后一个元素比较,np.where()返回下标索引数组组成的元组

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def cal_single_PR(labels, predict_matrix, item_num):
    label_category_nums = [0, 0, 0, 0]
    for label in labels:
        label_category_nums[label['category_id']] += 1
    category_related_AP = []
    for index, num in enumerate(label_category_nums):

        if num == 0:
            continue
        PR_matrix = np.full((item_num, 4), -1, dtype=np.float32)
        PR_matrix[:, 0] = predict_matrix[:, 0]
        PR_matrix[:, 1] = predict_matrix[:, 3]
        # 选出当前类的预测结果
        cur_category_matrix = PR_matrix[predict_matrix[:, 2] == index]
        TP = 0
        FP = 0
        FN = num
        for row in cur_category_matrix:
            if row[1] == 1:
                TP += 1
                FN = FN - 1 if FN > 0 else FN
            else:
                FP += 1
            row[2] = TP / (TP + FP)
            row[3] = TP / (TP + FN)
        AP = compute_ap(cur_category_matrix[:, 2], cur_category_matrix[:, 3])
        if AP > 1:
            print("error")
        category_related_AP.append(AP)
    return np.average(category_related_AP)


def cal_PR(config_file, checkpoint_file, img_path, anno_path):
    with open(anno_path, 'r', encoding='utf8') as fp:
        annos = json.load(fp)['annotations']
    fp.close()
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    i = 0
    while (i < len(annos)):
        anno = annos[i]
        image_id = anno['image_id']
        image_name = image_id + '.jpg'
        image = os.path.join(img_path, image_name)
        predict_result = inference_detector(model, image)
        labels = []
        for step, j in enumerate(range(i+1, len(annos)), 2):
            related_anno = annos[j]['image_id']
            if related_anno == image_id:
                labels.append(dict(bbox=annos[j]['bbox'], category_id=annos[j]['category_id']))
            else:
                i += step
                break
        item_num = 0
        for k in predict_result:
            item_num += len(k)
        id = 0
        predict_matrix = np.full((item_num, 4), -1, dtype=np.float32)
        for index, items in enumerate(predict_result):
            for item in items:
                confidence_score = item[4]
                bbox = [item[0], item[1], item[2], item[3]]
                category = index
                max_iou = 0
                for label in labels:
                    if label['category_id'] == category:
                        cur_iou = compute_iou(topleftxywh_to_xyxy(label['bbox']), bbox)
                        max_iou = cur_iou if cur_iou > max_iou else max_iou
                predict_matrix[id][0] = id
                predict_matrix[id][1] = confidence_score
                predict_matrix[id][2] = category
                predict_matrix[id][3] = 1 if max_iou > IOU_THR else 0
                id += 1
        predict_matrix = predict_matrix[np.argsort(-predict_matrix[:, 1])]
        AP = cal_single_PR(labels, predict_matrix, item_num)
        if AP >= MAP_THR:
            is_qualified = True
        else:
            is_qualified = False
        write_txt(r"D:\dataset\zhoucheng\map_split", image_name, is_qualified, AP)
        print(AP)
    print("error")


def write_txt(path, name, is_qualified, map):
    if is_qualified:
        txt = open(os.path.join(path, "qualified.txt"), mode="a+", encoding='utf-8')
    else:
        txt = open(os.path.join(path, "unqualified.txt"), mode="a+", encoding='utf-8')
    txt.write(name)
    txt.write("    ")
    txt.write(str(map))
    txt.write('\n')
    txt.close()


if __name__ == '__main__':
    config_file = 'E:\code\work_dirs\cascade_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = r'E:\code\work_dirs\epoch_24.pth'
    img_path = 'D:\dataset\zhoucheng\Images'
    anno_path = 'D:\dataset\zhoucheng/zhoucheng_coco.json'
    # model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # img = r'E:\code\work_dirs\2020-03-04_06_24_21_049.jpg'
    # with open("D:\dataset\zhoucheng/zhoucheng_train_coco.json", 'r', encoding='utf8') as fp:
    #     annos = json.load(fp)['annotations']
    # fp.close()
    # labels = []
    # for i in annos:
    #     if i['image_id'] == '2020-03-04_06_24_21_049':
    #         labels.append(dict(bbox=i['bbox'], category_id=i['category_id']))
    # predict_result = inference_detector(model, img)
    cal_PR(config_file, checkpoint_file, img_path, anno_path)
