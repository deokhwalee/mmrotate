import os
import sys
import os.path as osp
from PIL import Image
import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def load_coordinates(ann_dir):
    
    bboxes, labels = [], []

    if ann_dir is None:
            print("ann_dir is None")
            pass
    elif not osp.isfile(ann_dir):
        print(f"Can't find {ann_dir}, treated as empty ann_dir")
    else:
        with open(ann_dir, 'r') as f:
            for line in f:
                items = line.split(' ')
                bboxes.append([float(i) for i in items[:8]])
                labels.append(items[8])

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else \
        np.zeros((0, 8), dtype=np.float32)
    
    # print(bboxes[0][0])

    return bboxes


def bbox_to_yolo(bbox, img_dir):
    size = Image.open(img_dir).size
    dw = 1./size[0]
    dh = 1./size[1]
    x = (bbox[0] + bbox[2])/2.0
    y = (bbox[1] + bbox[3])/2.0
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x, y, w, h


def save_yolo_annotation(yolo_bboxes, save_dir):
    bboxes_num = yolo_bboxes.shape[0]
    label = '0'

    with open(save_dir, 'w') as f_out:
        # for i in range(0, len(dota_bboxes)):
        #     f_out.write(str(dota_bboxes[i]) + "\n")
        for idx in range(bboxes_num):
            outline = ' '.join(list(map(str, yolo_bboxes[idx])))
            outline = label + ' ' + outline
            f_out.write(outline + "\n")

def main():
    img_dir = "/datasets/posco/split_pile_yolo/images/"
    ann_dir = "/datasets/posco/split_pile_yolo/annfiles/"
    save_dir = "/datasets/posco/split_pile_yolo/converted/"

    imgs = os.listdir(img_dir)
    anns = os.listdir(ann_dir)
    imgs.sort()
    anns.sort()

    img_dirs = []
    ann_dirs = []

    for img in imgs:
        img_dirs.append(img_dir + img)
    for ann in anns:
        ann_dirs.append(ann_dir + ann)
    
    for i in range(len(ann_dirs)):
        bboxes = load_coordinates(ann_dirs[i])
        bboxes_num = bboxes.shape[0]
        yolo_bboxes = []
        for j in range(bboxes_num):
            bbox = [bboxes[j][0], bboxes[j][1], bboxes[j][4], bboxes[j][5]]
            x, y, w, h = bbox_to_yolo(bbox, img_dirs[i])
            yolo_bboxes.append([x, y, w, h])

        yolo_bboxes = np.array(yolo_bboxes, dtype=np.float32) if yolo_bboxes else \
        np.zeros((0, 4), dtype=np.float32)

        save_yolo_annotation(yolo_bboxes, save_dir + anns[i])


if __name__ == '__main__':
    main()
