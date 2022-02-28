# Import packages
import cv2
import numpy as np

img = cv2.imread('temp.jpg')

lego = [
    {'xCenter': 1927.5, 'yCenter': 667.0, 'width': 211.0, 'height': 634.0, 'classId': 'bottle', 'className': 'bottle',
     'color': (0, 50, 255)},
    {'xCenter': 2784.5, 'yCenter': 452.5, 'width': 241.0, 'height': 247.0, 'classId': 'cup', 'className': 'cup',
     'color': (255, 150, 0)},
    {'xCenter': 896.5, 'yCenter': 489.5, 'width': 943.0, 'height': 341.0, 'classId': 'keyboard',
     'className': 'keyboard', 'color': (255, 50, 0)}
]

lego = [
    {'xCenter': 2877.5, 'yCenter': 599.0, 'width': 189.0, 'height': 194.0, 'classId': 'cup', 'className': 'cup',  'color': (0, 50, 255)},
    {'xCenter': 1425.5, 'yCenter': 652.5, 'width': 735.0, 'height': 241.0, 'classId': 'keyboard', 'className': 'keyboard',   'color': (255, 150, 0)},
    {'xCenter': 2567.0, 'yCenter': 1738.5, 'width': 802.0, 'height': 897.0, 'classId': 'person', 'className': 'person',  'color': (255, 150, 0)},
    {'xCenter': 2202.5, 'yCenter': 791.5, 'width': 171.0, 'height': 517.0, 'classId': 'bottle', 'className': 'bottle', 'color': (255, 50, 0)}
]


lego = [{'xCenter': 468.0, 'yCenter': 454.0, 'zCenter': 505.0, 'xCenterConverted': 468.0, 'yCenterConverted': 454.0, 'width': 84.0, 'height': 174.0, 'classId': 'Screw_Desk_Clamp_Vise', 'className': 'Screw_Desk_Clamp_Vise', 'color': 'None'}, {'xCenter': 493.5, 'yCenter': 656.0, 'zCenter': 500, 'xCenterConverted': 493.5, 'yCenterConverted': 656.0, 'width': 289.0, 'height': 126.0, 'classId': 'Nagel', 'className': 'Nagel', 'color': 'None'}, {'xCenter': 217.0, 'yCenter': 396.5, 'zCenter': 509.0, 'xCenterConverted': 217.0, 'yCenterConverted': 396.5, 'width': 434.0, 'height': 421.0, 'classId': 'InsertionHandle', 'className': 'InsertionHandle', 'color': 'None'}, {'xCenter': 531.0, 'yCenter': 678.0, 'zCenter': 892.0, 'xCenterConverted': 531.0, 'yCenterConverted': 678.0, 'width': 182.0, 'height': 84.0, 'classId': 'Threadsx2', 'className': 'Threadsx2', 'color': 'None'}, {'xCenter': 103.5, 'yCenter': 292.5, 'zCenter': 500, 'xCenterConverted': 103.5, 'yCenterConverted': 292.5, 'width': 207.0, 'height': 583.0, 'classId': 'InsertionHandle', 'className': 'InsertionHandle', 'color': 'None'}, {'xCenter': 975.5, 'yCenter': 472.0, 'zCenter': 500.0, 'xCenterConverted': 975.5, 'yCenterConverted': 472.0, 'width': 585.0, 'height': 376.0, 'classId': 'AimingArm130', 'className': 'AimingArm130', 'color': 'None'}, {'xCenter': 411.0, 'yCenter': 606.0, 'zCenter': 499.0, 'xCenterConverted': 411.0, 'yCenterConverted': 606.0, 'width': 438.0, 'height': 152.0, 'classId': 'Nagel', 'className': 'Nagel', 'color': 'None'}, {'xCenter': 580.0, 'yCenter': 425.5, 'zCenter': 481.0, 'xCenterConverted': 580.0, 'yCenterConverted': 425.5, 'width': 46.0, 'height': 79.0, 'classId': 'tool_1', 'className': 'tool_1', 'color': 'None'}, {'xCenter': 355.5, 'yCenter': 661.0, 'zCenter': 909.0, 'xCenterConverted': 355.5, 'yCenterConverted': 661.0, 'width': 147.0, 'height': 118.0, 'classId': 'Nagel', 'className': 'Nagel', 'color': 'None'}, {'xCenter': 635.0, 'yCenter': 677.0, 'zCenter': 489.0, 'xCenterConverted': 635.0, 'yCenterConverted': 677.0, 'width': 42.0, 'height': 86.0, 'classId': 'Nagel', 'className': 'Nagel', 'color': 'None'}, {'xCenter': 569.5, 'yCenter': 205.0, 'zCenter': 464.0, 'xCenterConverted': 569.5, 'yCenterConverted': 205.0, 'width': 67.0, 'height': 84.0, 'classId': 'Slider', 'className': 'Slider', 'color': 'None'}, {'xCenter': 702.0, 'yCenter': 230.5, 'zCenter': 459.0, 'xCenterConverted': 702.0, 'yCenterConverted': 230.5, 'width': 74.0, 'height': 71.0, 'classId': 'nut_small', 'className': 'nut_small', 'color': 'None'}, {'xCenter': 702.0, 'yCenter': 229.5, 'zCenter': 459.0, 'xCenterConverted': 702.0, 'yCenterConverted': 229.5, 'width': 72.0, 'height': 69.0, 'classId': 'Slider', 'className': 'Slider', 'color': 'None'}, {'xCenter': 655.0, 'yCenter': 414.0, 'zCenter': 505.0, 'xCenterConverted': 655.0, 'yCenterConverted': 414.0, 'width': 118.0, 'height': 222.0, 'classId': 'Rep', 'className': 'Rep', 'color': 'None'}, {'xCenter': 569.0, 'yCenter': 203.5, 'zCenter': 463.0, 'xCenterConverted': 569.0, 'yCenterConverted': 203.5, 'width': 66.0, 'height': 81.0, 'classId': 'nut_small', 'className': 'nut_small', 'color': 'None'}, {'xCenter': 846.5, 'yCenter': 258.5, 'zCenter': 504.0, 'xCenterConverted': 846.5, 'yCenterConverted': 258.5, 'width': 79.0, 'height': 73.0, 'classId': 'nut_big', 'className': 'nut_big', 'color': 'None'}, {'xCenter': 406.5, 'yCenter': 378.0, 'zCenter': 511.0, 'xCenterConverted': 406.5, 'yCenterConverted': 378.0, 'width': 57.0, 'height': 136.0, 'classId': 'Screw_Desk_Clamp_Vise', 'className': 'Screw_Desk_Clamp_Vise', 'color': 'None'}, {'xCenter': 470.0, 'yCenter': 455.0, 'zCenter': 506.0, 'xCenterConverted': 470.0, 'yCenterConverted': 455.0, 'width': 88.0, 'height': 170.0, 'classId': 'Screw_Desk_Clamp', 'className': 'Screw_Desk_Clamp', 'color': 'None'}, {'xCenter': 454.0, 'yCenter': 230.5, 'zCenter': 510.0, 'xCenterConverted': 454.0, 'yCenterConverted': 230.5, 'width': 60.0, 'height': 59.0, 'classId': 'Headplate', 'className': 'Headplate', 'color': 'None'}, {'xCenter': 845.5, 'yCenter': 448.5, 'zCenter': 487.0, 'xCenterConverted': 845.5, 'yCenterConverted': 448.5, 'width': 145.0, 'height': 171.0, 'classId': 'Base_Rev', 'className': 'Base_Rev', 'color': 'None'}, {'xCenter': 522.0, 'yCenter': 306.0, 'zCenter': 507.0, 'xCenterConverted': 522.0, 'yCenterConverted': 306.0, 'width': 72.0, 'height': 68.0, 'classId': 'Jaw2', 'className': 'Jaw2', 'color': 'None'}]

for elem in lego:
    if elem['className'] != 'person':
        print(elem)
        xCenter = int(elem['xCenter'])
        yCenter = int(elem['yCenter'])
        width = int(elem['width'])
        height = int(elem['height'])
        start_point = (int(xCenter - (width / 2)), int(yCenter - (height / 2)))
        print((yCenter - (height / 2), xCenter - (width / 2)), (yCenter + (height / 2), xCenter + (width / 2)))
        end_point = (int(xCenter + (width / 2)), int(yCenter + (height / 2)))
        color = (0, 0, 255)
        thickness = 3
        image_with_rectangle = cv2.rectangle(
            img=img,
            pt1=start_point,
            pt2=end_point,
            color=color,
            thickness=thickness
        )
        cv2.imshow("Image with a Rectangle", image_with_rectangle)
        cv2.rectangle(
            img=image_with_rectangle,
            pt1=end_point,
            pt2=(int(xCenter + (width / 2)) - 160, int(yCenter + (height / 2)) - 50),
            color=color,
            thickness=-1
        )
        print(elem['className'])
        image_with_rectangle = cv2.putText(image_with_rectangle, elem['className'],
                                           (end_point[0] - 150, end_point[1] - 15),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2,
                                           lineType=cv2.LINE_AA)


cv2.imwrite('image_with_rectangle.jpg', image_with_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()
