# k:\Coding\ML\MergeConflict_PVH_ML\main.py
import argparse
import os
from train import train_model
from test import generate_submission

def main():
    parser = argparse.ArgumentParser(description='Fake Image Detection - PVH ML Hackathon')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'], default='both', help='Operation mode')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weights_dir', type=str, default='weights', help='Directory to save/load model weights')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model weights (for testing)')
    parser.add_argument('--test_dir1', type=str, default=None, help='Path to clean test dataset')
    parser.add_argument('--test_dir2', type=str, default=None, help='Path to adversarial test dataset')
    parser.add_argument('--output_dir', type=str, default='submissions', help='Directory to save submission files')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        print("=== Training Model ===")
        model = train_model(
            data_dir=args.data_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            save_dir=args.weights_dir
        )
    
    if args.mode in ['test', 'both']:
        print("=== Generating Submissions ===")
        model_path = args.model_path
        if model_path is None:
            model_path = os.path.join(args.weights_dir, 'best_model.pth')
            
        test_dir1 = args.test_dir1 if args.test_dir1 else args.data_dir
        test_dir2 = args.test_dir2 if args.test_dir2 else args.data_dir
        
        generate_submission(
            model_path=model_path,
            test_data_dir1=test_dir1,
            test_data_dir2=test_dir2,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()