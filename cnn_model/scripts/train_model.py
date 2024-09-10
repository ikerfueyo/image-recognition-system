from cnn_model.model.train import train_model
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train CNN model for image classification.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the training data directory.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer.')
    parser.add_argument('--checkpoint_dir', type=str, default='cnn_model/checkpoints', help='Directory to save model checkpoints.')

    args = parser.parse_args()

    train_model(
        image_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )

if __name__ == "__main__":
    main()
