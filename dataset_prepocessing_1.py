# syn image dataset prepro
import os.path

import numpy as np  # linear algebra

import json
from io import StringIO
import shutil

from numpy import load

data_json = {
    "info": {
        "description": "Camera Param Dataset",
        "url": "",
        "version": "1.0",
        "year": 2022,
        "contributor": "Hannah Schieber",
        "date_created": " "
    },
    "licenses": [
        {
            "url": "",
            "id": 1,
            "name": ""
        }
    ],
    "images": [],
    "annotations": [],
    "categories": [],  # <-- Not in Captions annotations
    "segment_info": []  # <-- Only in Panoptic annotations
}

classes = {
    'pringles': 1,
    'mustard': 2,
    'tomato_soup': 3,
    'rubiks': 4,
    'cracker_box': 5,
    'baseball': 6
}

def search_dict(dict, search_id):
    for name, class_id in dict.items():
        if class_id == search_id:
            return name

# box form[x,y,w,h]
# parse it to the yolo file format
def convert(filename, size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x1_center = box[0] + box[2] / 2
    y1_center = box[1] + box[3] / 2
    x = x1_center * dw
    y = y1_center * dh
    w = box[2] * dw
    h = box[3] * dh
    if x > 1 or y > 1 or w > 1 or h > 1:
        print("1 ERROR", filename)
    if x < 0 or y < 0 or w < 0 or h < 0:
        print("0 ERROR", filename)
    return (x, y, w, h)


def write_labels_to_coco_yolo_format(output_file_name, output_file_path):
    with open(output_file_name, 'r+') as f:
        data = json.load(f)
        for item in data['images']:
            image_id = item['id']
            file_name = item['file_name'].split("/")[1]
            width = item['width']
            height = item['height']
            value = list(filter(lambda item1: item1['image_id'] == str(image_id), data['annotations']))
            # os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            outfile = open(output_file_path + '/%s.txt' % (file_name[:-4]),  'w')
            for item2 in value:
                category_id = item2['category_id']
                class_id = category_id
                box = item2['bbox']
                bb = convert(file_name[:-4], (width, height), box)
                outfile.write(str(class_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            outfile.close()


def add_image(data, image_id, syn_images, split, index):
    # print(filename)
    syn_images.append({
        "license": 1,
        "file_name": 'images/' + str(index) + '.jpg',
        "coco_url": "",
        "height": int(data["width"]),
        "width": int(data["height"]),
        "date_captured": "",
        "flickr_url": "",
        "id": image_id
    })
    return syn_images


def parse_task_info_to_file(dataset_destination, task_name, files, syn_images, syn_images_anno):
    print("Finished " + task_name)
    data_json['annotations'] = syn_images_anno
    data_json['images'] = syn_images
    with open(dataset_destination + task_name + '_coco_style.json', 'w+') as f:
        json.dump(data_json, f)
    with open(dataset_destination + task_name + '.txt', 'w+') as f:
        f.write(files)


def run_data_parsing(index, threshold, dataset_origin, dataset_destination):
    '''
    parse to yolo format
    '''
    syn_images = []
    syn_images_anno = []

    files = ""

    split = 'train'

    counter = 0

    json_file = open(dataset_origin + "coco_annotations2.json", 'r')
    data = json.load(json_file)[0]

    for elem_i in data["images"]:
        image_id = elem_i['id']

        source_path = dataset_origin + elem_i["file_name"]
        if os.path.exists(source_path):

            syn_images = add_image(elem_i, image_id, syn_images, split, index)

            # files += elem_i["file_name"] + "\n"
            files += 'images/' + str(index) + '.jpg\n'

            for anno in data["annotations"]:
                if image_id == anno["image_id"] and anno["category_id"] != 100:
                    # print(anno["image_id"], anno["category_id"], anno["bbox"])
                    mapped_id = anno['category_id']
                    elem_name = str(mapped_id)
                    syn_images_anno.append({
                        "segmentation": [],
                        "area": 0,
                        "iscrowd": 0,
                        "image_id": str(index),
                        "bbox": anno["bbox"],
                        "category_id": mapped_id,
                        "id": int(elem_name + str(index))
                    })

            if counter <= threshold[0]:
                # print(source_path, dataset_destination + "images/train/" + str(index) + '.jpg')
                dest_fpath = dataset_destination + "images/train/"
                os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
                shutil.copy(source_path,  dest_fpath + str(index) + '.jpg')

            elif threshold[0] < counter <= threshold[1]:
                dest_fpath = dataset_destination + "images/val/"
                os.makedirs(os.path.dirname(dest_fpath), exist_ok=True)
                shutil.copy(source_path, dest_fpath + str(index) + '.jpg')
            else:
                break
                #    shutil.copy(source_path, dataset_destination + "images/test")

            # print(index == threshold[0])
            if counter == threshold[0]:
                parse_task_info_to_file(dataset_destination, "train", files, syn_images, syn_images_anno)
                syn_images = []
                syn_images_anno = []
                files = ''
                split = 'val'

            if counter == threshold[1]:
                parse_task_info_to_file(dataset_destination, "val", files, syn_images, syn_images_anno)
                syn_images = []
                syn_images_anno = []
                files = ''

            counter += 1
            index += 1

    return counter

if __name__ == '__main__':
    threshold = [1211, 1808]
    # threshold = [1, 2, 3]
    # load array
    index = 0
    dataset_origin = "/media/hannah/Data/camera_parameters/out_oak_d_lite/coco_data/"
    dataset_destination = "/media/hannah/Data/camera_parameters/mixed_camera_param/"
    index = run_data_parsing(0, threshold, dataset_origin, dataset_destination)
    print(index)

    write_labels_to_coco_yolo_format(
        dataset_destination + 'train_coco_style.json',
        dataset_destination + 'labels/train/')

    write_labels_to_coco_yolo_format(
        dataset_destination + 'val_coco_style.json',
        dataset_destination + 'labels/val/')