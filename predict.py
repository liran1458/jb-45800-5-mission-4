import sys

import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms, models


MODEL_PATH = "model.pt"

# If no image path is provided, use test_image.jpg
IMAGE_PATH = sys.argv[1] if len(sys.argv) > 1 else "test_image.jpg"


# --------------------
# Load saved model data
# --------------------

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

classes = checkpoint["classes"]
image_size = checkpoint["image_size"]


# --------------------
# Image transform
# Must match train.py
# --------------------

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------
# Build ResNet18
# --------------------

# No pretrained download is needed here.
# The trained weights are loaded from model.pt.
model = models.resnet18(
    weights=None
)

number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    len(classes)
)


# --------------------
# Load trained weights
# --------------------

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# --------------------
# Load foreign image
# --------------------

image = Image.open(
    IMAGE_PATH
).convert("RGB")

image = transform(image)

# Add batch dimension
image = image.unsqueeze(0)


# --------------------
# Prediction
# --------------------

with torch.no_grad():

    output = model(image)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1
    )


predicted_class = classes[
    predicted_index.item()
]

confidence_percentage = (
    confidence.item() * 100
)


# --------------------
# Result
# --------------------

print("Image:", IMAGE_PATH)

print(
    "Prediction:",
    predicted_class
)

print(
    f"Confidence: "
    f"{confidence_percentage:.2f}%"
)