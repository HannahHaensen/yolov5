
from flask import Flask
from flask_restful import Resource, Api
import numpy as np

app = Flask(__name__)
api = Api(app)

class ObjectCoords(Resource):

    def __init__(self):
        print("init detector")
        # self.detector = Detector()

    def get(self):
        # Load camera with the default config
        # k4a = PyK4A()
        # k4a.start()
        # capture = k4a.get_capture()
        # img_color = capture.color[:, :, 2::-1]  # BGRA to RGB
        # print(img_color.shape)

        # img = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        # cv2.imwrite("data/test.jpg", img)

        # depth = pyk4a.depth_image_to_color_camera(capture.depth, capture._calibration, capture.thread_safe)

        # ret_dict = self.detector.detect(img_color, depth, capture)

        print("collected and saved image")

        return []


api.add_resource(ObjectCoords, '/')

if __name__ == '__main__':
    app.run(debug=True, host="192.168.1.119")