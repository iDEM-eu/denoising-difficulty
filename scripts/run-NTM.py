import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training and Evaluation for NTM with Noise Transition Matrix")
    
    # Data and Output Paths
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='Path to the input dataset file')
    parser.add_argument('-o', '--output_dir', type=str, default='./NTM-en-BERT',
                        help='Directory for saving output models and results')
    parser.add_argument('--sep', type=str, default='\t',
                        help='Field separator in the dataset file')
    
    # Model Parameters
    parser.add_argument('-m', '--model_name', type=str, default='bert-base-multilingual-cased',
                        help='BERT model name from HuggingFace')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and evaluation')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for the optimizer')
    
    # Training Parameters
    parser.add_argument('-e', '--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--valsplit', type=float, default=0.2,
                        help='Proportion of the data used for validation split')
    parser.add_argument('--early_stopping', type=int, default=2,
                        help='Number of epochs with no improvement before stopping')
    
    # Noise Transition Matrix Parameters
    parser.add_argument('--noise_threshold', type=float, default=0.8,
                        help='Threshold for detecting noisy sentences')
    
    # Evaluation and Test Parameters
    parser.add_argument('-t', '--test_files', type=str, nargs='+',
                        help='List of test dataset files with language codes')
    
    return parser.parse_args()
