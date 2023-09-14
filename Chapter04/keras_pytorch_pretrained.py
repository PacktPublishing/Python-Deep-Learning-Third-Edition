from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

# With pretrained weights:
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
model = mobilenet_v3_large(weights="IMAGENET1K_V1")

# Using no weights:
model = mobilenet_v3_large(weights=None)

from torchvision.models import list_models, get_model

# List available models
all_models = list_models()
model = get_model(all_models[0], weights="DEFAULT")

from keras.applications.mobilenet_v3 import MobileNetV3Large
model = MobileNetV3Large(weights='imagenet')
