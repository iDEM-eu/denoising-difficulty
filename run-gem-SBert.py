import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Optimization of Gaussian Mixture Model with SBERT embeddings")
    
    # Data and Output Paths
    parser.add_argument('-i', '--input_file', type=str, required=True,
                        help='Path to the input dataset file')
    parser.add_argument('-o', '--output_dir', type=str, default='./Sent-en-GMM-SBERT',
                        help='Directory for saving output models and results')
    parser.add_argument('--sep', type=str, default='\t',
                        help='Field separator in the dataset file')
    
    # Model Parameters
    parser.add_argument('-m', '--model_name', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='SBERT model name from HuggingFace')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for generating embeddings')
    
    # GMM Hyperparameters
    parser.add_argument('--n_components_min', type=int, default=2,
                        help='Minimum number of Gaussian components for GMM')
    parser.add_argument('--n_components_max', type=int, default=10,
                        help='Maximum number of Gaussian components for GMM')
    parser.add_argument('--covariance_types', type=str, nargs='+', default=['full', 'tied', 'diag', 'spherical'],
                        help='List of covariance types to try for GMM')
    parser.add_argument('--tol_min', type=float, default=1e-5,
                        help='Minimum tolerance for convergence of GMM')
    parser.add_argument('--tol_max', type=float, default=1e-2,
                        help='Maximum tolerance for convergence of GMM')
    parser.add_argument('--max_iter_min', type=int, default=50,
                        help='Minimum number of iterations for GMM')
    parser.add_argument('--max_iter_max', type=int, default=300,
                        help='Maximum number of iterations for GMM')
    
    # Optuna Optimization Parameters
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Number of trials for Optuna optimization')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='Number of parallel jobs for Optuna optimization')
    
    return parser.parse_args()
