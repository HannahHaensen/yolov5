import cv2
import pyk4a
import torch
from flask import Flask
from flask_restful import Resource, Api
from matplotlib import pyplot as plt
from pyk4a import PyK4A, CalibrationType

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadImages2
from utils.general import non_max_suppression, scale_coords, xyxy2xywh
from utils.torch_utils import select_device
import numpy as np

app = Flask(__name__)
api = Api(app)

imagesFolder = "data/"

# load model from given weights
device = select_device("")
# load FP32 model
model = attempt_load("runs/train/exp2/weights/best.pt", map_location=device)
stride = int(model.stride.max())
# get class names
names = model.module.names if hasattr(model, 'module') else model.names
stride = int(model.stride.max())  # model stride
half = device.type != 'cpu'
if half:
    model.half()  # to FP16


class Detector:
    def __init__(self):
        # load model from given weights
        print("INIT detector")

    def detect(self, image, depth_data, capture):
        print("detecting image")
        height, width = depth_data.shape

        temp_image = cv2.imread('data/test.jpg')
        ret_dict = []

        dataset = LoadImages2([image], stride=stride)

        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=False)[0]

            # non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
            #                                        max_det=opt.max_det)
            pred = non_max_suppression(pred, conf_thres=0.2, iou_thres=0.45, agnostic=False, max_det=1000)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
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
                        depth_default = 500
                        depth = float(depth_data[int(res[1]), int(res[0])])
                        # print(im0.shape, image.shape)
                        if depth == 0:
                            depth = depth_default
                        # x, y, z = capture._calibration.convert_2d_to_3d(point, depth, CalibrationType.COLOR)
                        r, g, b = image[int(res[1]), int(res[0])]

                        cv2.circle(temp_image, (int(res[0]), int(res[1])), 5, (0, 0, 255), -1)

                        use_res = False
                        color = "NONE"
                        class_name = str(names[int(cls)])
                        if str(names[int(cls)]) == "2x2_Brick" or str(names[int(cls)]) == "2x2_Plate":
                            print("2x2_Brick", r, g, b)
                            if r > 150 and g > 150 and b > 150:
                                use_res = True
                                color = "white"
                                class_name = "2x2_Brick"
                        elif str(names[int(cls)]) == "2x3_Brick" or str(names[int(cls)]) == "2x3_Plate":
                            print("2x3_Brick", r, g, b)
                            if r > 200 and b < 50:
                                use_res = True
                                color = "orange"
                                class_name = "2x3_Brick"
                        elif str(names[int(cls)]) == "2x4_Brick" or str(names[int(cls)]) == "2x4_Plate":
                            print("2x4_Brick", r, g, b)
                            if r < 150:
                                use_res = True
                                color = "blue"
                                class_name = "2x4_Brick"
                        elif str(names[int(cls)]) == "2x6_Brick" or str(names[int(cls)]) == "2x6_Plate":
                            print("2x6_Brick", r, g, b)
                            if r > 50:
                                use_res = True
                                color = "brown"
                                class_name = "2x6_Brick"
                        else:
                            use_res = False
                            print("color not set", r, g, b)
                        if use_res:
                            ret_dict.append(
                                {
                                    "xCenter": float(res[0]),
                                    "yCenter": float(res[1]),
                                    "zCenter": depth,
                                    "xCenterConverted": float(res[0]),
                                    "yCenterConverted": float(res[1]),
                                    "width": float(res[2]),
                                    "height": float(res[3]),
                                    "classId": class_name,
                                    "className": class_name,
                                    "color": color
                                }
                            )

        print(ret_dict)

        cv2.imwrite("temp.jpg", temp_image)
        # {k: v for k, v in sorted(x.items(), key=lambda item: item[1])}
        return sorted(ret_dict, key=lambda yolo_value: (yolo_value['xCenter'], yolo_value['yCenter']))


class ObjectCoords(Resource):

    def __init__(self):
        self.detector = Detector()

    def get(self):
        # Load camera with the default config
        k4a = PyK4A()
        k4a.start()
        capture = k4a.get_capture()
        img_color = capture.color[:, :, 2::-1]  # BGRA to RGB
        print(img_color.shape)

        img = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        cv2.imwrite("data/test.jpg", img)

        depth = pyk4a.depth_image_to_color_camera(capture.depth, capture._calibration, capture.thread_safe)

        ret_dict = self.detector.detect(img_color, depth, capture)

        print("collected and saved image")

        return ret_dict


api.add_resource(ObjectCoords, '/')

if __name__ == '__main__':
    app.run(debug=True, host="192.168.1.119")
