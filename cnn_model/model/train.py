import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .cnn import SimpleCNN
from ..data.dataset import ImageClassificationDataset
from ..data.transforms import get_train_transforms
import os

def train_model(image_dir, num_epochs=10, batch_size=32, learning_rate=0.001, checkpoint_dir='checkpoints'):
    dataset = ImageClassificationDataset(image_dir=image_dir, transform=get_train_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleCNN(num_classes=2)  # Change num_classes based on your dataset
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

        # Save checkpoint
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f'cnn_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model checkpoint saved at {checkpoint_path}')

if __name__ == "__main__":
    # Example usage
    train_model(image_dir='path/to/training/data', num_epochs=10)
