import argparse
import os.path as osp
from PIL import Image
import cv2
import numpy as np


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
                bboxes.append([float(i) for i in items[1:]])
                labels.append(items[0])

    bboxes = np.array(bboxes, dtype=np.float32) if bboxes else \
        np.zeros((0, 4), dtype=np.float32)
    
    # print(bboxes[0][0])

    return bboxes


def yolo_to_bbox(x, y, w, h):
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return x1, y1, x2, y2


def yolo_to_dota(yolo_bboxes, img_dir):
    size = Image.open(img_dir).size
    width, height = size[0], size[1]
    dota_bboxes = []

    for i in range(len(yolo_bboxes)):
        x, y, w, h = yolo_bboxes[i]
        xmin, ymin, xmax, ymax = yolo_to_bbox(x, y, w, h)
        x1, y1 = xmin*width, ymin*height
        x2, y2 = xmax*width, ymin*height
        x3, y3 = xmax*width, ymax*height
        x4, y4 = xmin*width, ymax*height

        dota_bboxes.append([x1, y1, x2, y2, x3, y3, x4, y4])

    dota_bboxes = np.array(dota_bboxes, dtype=np.float32) if dota_bboxes else \
        np.zeros((0, 9), dtype=np.float32)
    return dota_bboxes

def save_dota_annotation(dota_bboxes, save_dir):
    bboxes_num = dota_bboxes.shape[0]
    label = '0'

    with open(save_dir, 'w') as f_out:
        # for i in range(0, len(dota_bboxes)):
        #     f_out.write(str(dota_bboxes[i]) + "\n")
        for idx in range(bboxes_num):
            outline = ' '.join(list(map(str, dota_bboxes[idx])))
            outline = label + ' ' + outline
            f_out.write(outline + "\n")

def main():
    parser = argparse.ArgumentParser(description='Converting YOLO format to DOTA')
    parser.add_argument('--img-dir', type=str, default="/datasets/posco/images/train/359.tif")
    parser.add_argument('--ann-dir', type=str, default="/datasets/posco/labels/train/359.txt")
    parser.add_argument('--save-dir', type=str, default="/datasets/posco/labels/converted/359.txt")
    args =parser.parse_args()

    yolo_bboxes = load_coordinates(args.ann_dir)
    dota_bboxes = yolo_to_dota(yolo_bboxes, args.img_dir)
    save_dota_annotation(dota_bboxes, args.save_dir)

if __name__ == '__main__':
    main()
