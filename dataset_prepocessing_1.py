# syn image dataset prepro
import os.path

import numpy as np  # linear algebra

import json
from io import StringIO
import shutil

from numpy import load

data_json = {
    "info": {
        "description": "Mixed Tools Dataset",
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

old_classes = {
    "Cube": 100,
    "BallJoint": 0,
    "Base_Rev": 1,
    "circle": 2,
    "gear_big_thread": 3,
    "gear_small_inner": 4,
    "gear_small_pole": 5,
    "Headplate": 6,
    "Jaw1": 7,
    "Jaw2": 8,
    "Knobx2": 9,
    "nut_big": 10,
    "Nut_Deskclamp": 11,
    "nut_small": 12,
    "Padx2": 13,
    "Pinx4": 14,
    "pole_big_thread": 15,
    "Rep": 16,
    "Screw_Desk_Clamp": 17,
    "screw": 18,
    "screw_big_thread": 19,
    "screw_small_thread": 20,
    "Slider": 21,
    "SlidingHingex4": 22,
    "Threadsx2": 23,
    "Screw_Desk_Clamp_Vise": 24,
    "gear_outer": 25,
    "tool_1": 26,
    "Rotator": 27,
    "1x1_Plate": 28,
    "1x2_Plate": 29,
    "1x3_Plate": 30,
    "1x4_Plate": 31,
    "1x5_Plate": 32,
    "1x6_Plate": 33,
    "1x8_Plate": 34,
    "1x9_Plate": 35,
    "1x10_Plate": 36,
    "1x11_Plate": 37,
    "1x12_Plate": 38,
    "1x1_Brick": 39,
    "1x2_Brick": 40,
    "1x3_Brick": 41,
    "1x4_Brick": 42,
    "1x5_Brick": 43,
    "1x6_Brick": 44,
    "1x8_Brick": 45,
    "1x9_Brick": 46,
    "1x10_Brick": 47,
    "1x11_Brick": 48,
    "1x12_Brick": 49,
    "2x1_Plate": 50,
    "2x2_Plate": 51,
    "2x3_Plate": 52,
    "2x4_Plate": 53,
    "2x5_Plate": 54,
    "2x6_Plate": 55,
    "2x8_Plate": 56,
    "2x9_Plate": 57,
    "2x10_Plate": 58,
    "2x11_Plate": 59,
    "2x12_Plate": 60,
    "2x1_Brick": 61,
    "2x2_Brick": 62,
    "2x3_Brick": 63,
    "2x4_Brick": 64,
    "2x5_Brick": 65,
    "2x6_Brick": 66,
    "2x7_Brick": 67,
    "2x8_Brick": 68,
    "2x9_Brick": 69,
    "2x10_Brick": 70,
    "2x11_Brick": 71,
    "2x12_Brick": 72,
    "Nagel": 73,
    "AimingArm130": 74,
    "InsertionHandle": 75,
    "Covidien_Mesh_0": 76,
    "Covidien_Front_Cylinder": 77
}

new_classes = {
    "BallJoint": 0,
    "tool_1": 1,
    "Rotator": 2,
    "AimingArm130": 3,
    "Covidien_Mesh_0": 4,
    "Covidien_Front_Cylinder": 5,
    "Headplate": 6,
    "Nut_Deskclamp": 7,
    "Knobx2": 8,
    "Screw_Desk_Clamp_Vise": 9,
    "Screw_Desk_Clamp": 10,
    "screw_small_thread": 11,
    "Threadsx2": 12,
    "Rep": 13,
    "Base_Rev": 14,
    "Slider": 15
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
            value = list(filter(lambda item1: item1['image_id'] == image_id, data['annotations']))
            outfile = open(output_file_path + '/%s.txt' % (file_name[:-4]),
                           'w')
            for item2 in value:
                category_id = item2['category_id']
                class_id = category_id
                box = item2['bbox']
                bb = convert(file_name[:-4], (width, height), box)
                outfile.write(str(class_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            outfile.close()


def add_image(data, image_id, syn_images):
    # print(filename)
    syn_images.append({
        "license": 1,
        "file_name": data["file_name"],
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
    with open(dataset_destination + task_name + '_tools_mixed_coco_style.json',
              'w+') as f:
        json.dump(data_json, f)
    with open(dataset_destination + task_name + '.txt', 'w+') as f:
        f.write(files)


def run_data_parsing(threshold, dataset_origin, dataset_destination):
    '''
    parse to yolo format
    '''
    syn_images = []
    syn_images_anno = []

    train_ids = {}
    val_ids = {}
    test_ids = {}
    ids = {}
    files = ""
    index = 0

    json_file = open(dataset_origin + "coco_annotations.json", 'r')
    data = json.load(json_file)

    for elem_i in data["images"]:
        image_id = elem_i['id']

        source_path = dataset_origin + elem_i["file_name"]
        if os.path.exists(source_path):

            syn_images = add_image(elem_i, image_id, syn_images)

            files += elem_i["file_name"] + "\n"

            for anno in data["annotations"]:
                if image_id == anno["image_id"] and anno["category_id"] != 100:
                    # print(anno["image_id"], anno["category_id"], anno["bbox"])
                    if search_dict(old_classes, anno['category_id']) in new_classes.keys():
                        mapped_id = new_classes[search_dict(old_classes, anno['category_id'])]
                        elem_name = str(mapped_id)
                        syn_images_anno.append({
                            "segmentation": [],
                            "area": 0,
                            "iscrowd": 0,
                            "image_id": image_id,
                            "bbox": anno["bbox"],
                            "category_id": mapped_id,
                            "id": int(elem_name + str(image_id))
                        })
                        if elem_name not in ids.keys():
                            ids[elem_name] = 1
                        else:
                            ids[elem_name] = ids[elem_name] + 1

            if index <= threshold[0]:
                # print(source_path)
                shutil.copy(source_path, dataset_destination + "images/val")

            # elif threshold[0] < index <= threshold[1]:
            #    shutil.copy(source_path, dataset_destination + "images/val")
            # else:
            #    shutil.copy(source_path, dataset_destination + "images/test")

            # print(index == threshold[0])
            if index == threshold[0]:
                parse_task_info_to_file(dataset_destination, "val", files, syn_images, syn_images_anno)
                train_ids = ids
                break

            print(index)
            index += 1

    return train_ids, val_ids, test_ids

if __name__ == '__main__':
    threshold = [1831]
    # threshold = [1, 2, 3]
    # load array

    dataset_origin = "D:/BlenderProc_dataset3/validation/coco_data/"
    # dataset_origin = "C:/Users/Admin/Documents/GitHub/BlenderProc/examples/basics/physics_positioning_combined/output/coco_data_val/"
    dataset_destination = "D:/YoloV5_datasets/data/mixed_tools_marss/"
    run_data_parsing(threshold, dataset_origin, dataset_destination)

    write_labels_to_coco_yolo_format(
        dataset_destination + 'val_tools_mixed_coco_style.json',
        dataset_destination + 'labels/val')