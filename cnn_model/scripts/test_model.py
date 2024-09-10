from cnn_model.model.test import test_model
import argparse

def main():
    parser = argparse.ArgumentParser(description='Test CNN model for image classification.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the test data directory.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint to load.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing.')

    args = parser.parse_args()

    test_model(
        image_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
