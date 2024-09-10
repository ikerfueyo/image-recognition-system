import os
from PIL import Image
from cnn_model.data.transforms import get_train_transforms
import argparse

def preprocess_and_save(image_dir, output_dir):
    """Preprocess images and save them to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transform = get_train_transforms()

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        try:
            image = Image.open(image_path).convert('RGB')
            image = transform(image)

            save_path = os.path.join(output_dir, image_name)
            image.save(save_path)
            print(f"Processed and saved {image_name}")
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess images for image classification.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the raw images directory.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save preprocessed images.')

    args = parser.parse_args()

    preprocess_and_save(image_dir=args.data_dir, output_dir=args.output_dir)

if __name__ == "__main__":
    main()
