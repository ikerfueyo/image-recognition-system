import torch
from torch.utils.data import DataLoader
from .cnn import SimpleCNN
from ..data.dataset import ImageClassificationDataset
from ..data.transforms import get_test_transforms

def test_model(image_dir, checkpoint_path, batch_size=32):
    dataset = ImageClassificationDataset(image_dir=image_dir, transform=get_test_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN(num_classes=2) # Change num_classes based on dataset
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy}%')

if __name__ == "__main__":
    # Example usage
    test_model(image_dir='path/to/test/data', checkpoint_path='path/to/model_checkpoint.pth')
