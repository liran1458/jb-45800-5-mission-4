import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms



DATASET_PATH = "dataset"
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001



transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

print("Classes:", dataset.classes)
print("Number of classes:", len(dataset.classes))
print("Number of images:", len(dataset))



torch.manual_seed(42)

train_size = int(0.8 * len(dataset))
validation_size = len(dataset) - train_size

train_dataset, validation_dataset = random_split(
    dataset,
    [train_size, validation_size]
)

print("Training images:", len(train_dataset))
print("Validation images:", len(validation_dataset))



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



class ChessCNN(nn.Module):

    def __init__(self, number_of_classes):
        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),

            nn.Linear(128, number_of_classes)
        )

    def forward(self, x):
        return self.network(x)


model = ChessCNN(len(dataset.classes))



loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)



for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        optimizer.zero_grad()

        predictions = model(images)

        loss = loss_function(predictions, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()



    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            predictions = model(images)

            predicted_classes = predictions.argmax(dim=1)

            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f} "
        f"- Validation Accuracy: {accuracy:.2f}%"
    )


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes,
        "image_size": IMAGE_SIZE
    },
    "model.pt"
)

print("Model saved to model.pt")