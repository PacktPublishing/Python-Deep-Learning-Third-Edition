from ultralytics import YOLO

# Load a YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")

# Detect objects on a Wikipedia image
results = model.predict('https://raw.githubusercontent.com/ivan-vasilev/Python-Deep-Learning-3rd-Edition/main/Chapter05/wikipedia-2011_FIA_GT1_Silverstone_2.jpg')

# convert results->numpy_array->Image and display it
from PIL import Image

Image.fromarray(results[0].plot()).show()
