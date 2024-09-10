import os
from PIL import Image
from torch.utils.data import Dataset

class ImageClassificationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load the image and apply transformations."""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        label = self.get_label_from_path(img_path)
        
        return image, label

    def get_label_from_path(self, img_path):
        """Extract label from image file path."""
        # Assuming directory structure is image_dir/class_name/image.jpg
        return img_path.split(os.path.sep)[-2]
