import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


DATASET_PATH = "dataset"
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 0.0001


# --------------------
# Image transforms
# --------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------
# Load dataset
# --------------------

full_dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

print("Classes:", full_dataset.classes)
print("Number of classes:", len(full_dataset.classes))
print("Number of images:", len(full_dataset))


# --------------------
# Train / validation split
# --------------------

torch.manual_seed(42)

indices = torch.randperm(
    len(full_dataset)
).tolist()

train_size = int(
    0.8 * len(indices)
)

train_indices = indices[:train_size]

validation_indices = indices[train_size:]


train_dataset = Subset(
    full_dataset,
    train_indices
)

validation_dataset = Subset(
    full_dataset,
    validation_indices
)


print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(validation_dataset)
)


# --------------------
# DataLoaders
# --------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------
# ResNet18 model
# --------------------

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# Freeze all pretrained layers
for parameter in model.parameters():
    parameter.requires_grad = False


# Unfreeze the last ResNet block
for parameter in model.layer4.parameters():
    parameter.requires_grad = True


# Replace the final classification layer
number_of_features = model.fc.in_features

model.fc = nn.Linear(
    number_of_features,
    len(full_dataset.classes)
)


# --------------------
# Loss + optimizer
# --------------------

loss_function = nn.CrossEntropyLoss()


optimizer = torch.optim.Adam(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


# --------------------
# Training
# --------------------

best_accuracy = 0


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0


    for images, labels in train_loader:

        optimizer.zero_grad()

        predictions = model(images)

        loss = loss_function(
            predictions,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    # --------------------
    # Validation
    # --------------------

    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in validation_loader:

            predictions = model(images)

            predicted_classes = predictions.argmax(
                dim=1
            )

            correct += (
                predicted_classes == labels
            ).sum().item()

            total += labels.size(0)


    accuracy = (
        correct / total
    ) * 100

    average_loss = (
        total_loss / len(train_loader)
    )


    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f} "
        f"- Validation Accuracy: {accuracy:.2f}%"
    )


    # --------------------
    # Save best model
    # --------------------

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": full_dataset.classes,
                "image_size": IMAGE_SIZE,
                "architecture": "resnet18"
            },
            "model.pt"
        )

        print(
            f"New best model saved - "
            f"Accuracy: {best_accuracy:.2f}%"
        )


print()

print(
    f"Best Validation Accuracy: "
    f"{best_accuracy:.2f}%"
)