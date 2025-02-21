import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training and Evaluation for Co-Teaching Model")
    
    # Model and Training Parameters
    parser.add_argument('-m', '--model_name', type=str, default='bert-base-multilingual-cased',
                        help='Model name from HuggingFace')
    parser.add_argument('-l', '--local_dir', type=str, default='./Sent-en-coteaching-output',
                        help='Directory for saving the trained models')
    parser.add_argument('-p', '--project_name', type=str, default='General-model-English',
                        help='Project name for WandB logging')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable logging to WandB')
    
    # Data Handling
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='Path to the input dataset file')
    parser.add_argument('--sep', type=str, default='\t',
                        help='Field separator in the dataset file')
    parser.add_argument('-t', '--test_files', type=str, nargs='+',
                        help='List of test dataset files with language codes')
    
    # Training Parameters
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--eval_steps', type=int, default=1000, help='Number of steps between evaluations')
    parser.add_argument('--valsplit', type=float, default=0.2,
                        help='Proportion of the data used for validation split')
    parser.add_argument('--early_stopping', type=int, default=2,
                        help='Number of epochs with no improvement before stopping')
    
    # Co-Teaching Specific Parameters
    parser.add_argument('--rho_min', type=float, default=0.0,
                        help='Minimum forget rate for co-teaching')
    parser.add_argument('--rho_max', type=float, default=0.3,
                        help='Maximum forget rate for co-teaching')
    parser.add_argument('--confidence_threshold', type=float, default=0.6,
                        help='Threshold for identifying low-confidence samples')
    parser.add_argument('--checkpoint_interval', type=int, default=2,
                        help='Interval to save model checkpoints during training')
    
    return parser.parse_args()
