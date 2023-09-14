import numpy as np


def draw_bboxes(image: np.array, det_objects: dict):
    """Draw bounding boxes and predicted classes"""
    # COCO dataset class names
    classes = [
        'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack',
        'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush']

    import numpy as np
    # random color for each class

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # iterate over the network output for all boxes
    for box, box_class, score in zip(det_objects[0]['boxes'].detach().numpy().astype(int),
                                     det_objects[0]['labels'].detach().numpy(),
                                     det_objects[0]['scores'].detach().numpy()):

        # filter the boxes by objectness score
        if score > 0.5:
            # transform bounding box format
            box = [(box[0], box[1]), (box[2], box[3])]

            # draw the bounding box
            cv2.rectangle(img=image,
                          pt1=box[0],
                          pt2=box[1],
                          color=colors[box_class],
                          thickness=4)

            # display the box class label
            cv2.putText(img=image,
                        text=classes[box_class],
                        org=box[0],
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2,
                        color=colors[box_class],
                        thickness=4)


# Download object detection image
import requests
import os.path

image_file_path = 'source_2.png'
if not os.path.isfile(image_file_path):
    url = 'https://github.com/ivan-vasilev/Python-Deep-Learning-3rd-Edition/blob/main/Chapter05/source_2.png?raw=true'
    r = requests.get(url)
    with open(image_file_path, 'wb') as f:
        f.write(r.content)

# load the pytorch model
from torchvision.models.detection import \
    FasterRCNN_ResNet50_FPN_V2_Weights, \
    fasterrcnn_resnet50_fpn_v2

model = fasterrcnn_resnet50_fpn_v2(
    weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

# set the model in evaluation mode
model.eval()

# read the image file
import cv2

img = cv2.imread(image_file_path)

# transform the input to tensor
import torchvision.transforms as transforms

transform = transforms.ToTensor()

nn_input = transform(img)

# run the model
detected_objects = model([nn_input])

draw_bboxes(img, detected_objects)

cv2.imshow("Object detection", img)
cv2.waitKey()
