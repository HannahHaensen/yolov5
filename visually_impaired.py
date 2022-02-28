import json

import cv2
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
import torch
from matplotlib import pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadImages2
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device

from flask_classful import FlaskView
from PIL import Image
import numpy as np

app = Flask(__name__)
api = Api(app)

imagesFolder = "data/"

coco_classes = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush"
}

# load model from given weights
device = select_device("")
# load FP32 model
model = attempt_load("yolov5s.pt", map_location=device)
stride = int(model.stride.max())
# get class names
names = model.module.names if hasattr(model, 'module') else model.names
stride = int(model.stride.max())  # model stride
half = device.type != 'cpu'
if half:
    model.half()  # to FP16


class Detector:
    def __init__(self):
        print("init detector")

    def detect(self, image):
        print("detecting image")
        print(image.shape)
        temp_image = cv2.imread('test.jpg')
        ret_dict = []

        dataset = LoadImages2([image], stride=stride)

        print(names)

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=False)[0]

            pred = non_max_suppression(pred, conf_thres=0.35, iou_thres=0.45, agnostic=False, max_det=1000)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                # print(p, s, im0, frame)
                print(im0.shape)
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xyxy = torch.tensor(xyxy).view(-1, 4)
                        res = np.copy(xyxy2xywh(xyxy)[0])  # boxes
                        # put a blue dot at (10, 20)
                        cv2.circle(temp_image, (int(res[0]), int(res[1])), 5, (0, 0, 255), -1)
                        # plt.scatter(res[0], res[1])
                        ret_dict.append(
                            {
                                "xCenter": float(res[0]),
                                "yCenter": float(res[1]),
                                "width": float(res[2]),
                                "height": float(res[3]),
                                "classId": names[int(cls)],
                                "className": coco_classes[int(cls)]
                                # "confidence": conf
                            }
                        )

        cv2.imwrite("temp.jpg", temp_image)

        return ret_dict


detector = Detector()

@app.route('/post/objectcoords', methods=['POST'])
def post_object_coords():
    # show the user profile for that user
    print("post received")
    filestr = request.files['image'].read()
    # convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    print(npimg.shape)

    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    if img.shape[-1]:
        img = img[..., :3]
    print(img.shape)
    cv2.imwrite('test.jpg', img)
    ret = detector.detect(img)

    print(ret)
    return jsonify(ret)


cache = {
    'distance': -1
}

@app.route('/get/distance', methods=['GET'])
def get_distance():
    print(cache['distance'])
    data = {}
    data['distance'] = cache['distance']
    print("Apple Watch")
    return jsonify([data])


@app.route('/post/distance', methods=['POST'])
def post_distance():
    print("post received")
    cache['distance'] = float(request.form.get('distance').replace(",", "."))
    print(cache['distance'])
    return 'Success'


if __name__ == '__main__':
    app.run(debug=True, host="192.168.1.112")
