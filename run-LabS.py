#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import LabS
import argparse

parser = argparse.ArgumentParser(description="A Transformer Model with Label Smoothing")
parser.add_argument('-m', '--mname', type=str, default='bert-base-multilingual-cased', help='Model name from HuggingFace')
parser.add_argument('-l', '--local', type=str, default=None, help='Directory for saving the local model')
parser.add_argument('-n', '--hubname', type=str, default=None, help='Model name to save to the hub')
parser.add_argument('-p', '--projectname', type=str, default=None, help='Project name for WandB')
parser.add_argument('--use_wandb', action='store_true', help='Flag to enable WandB logging')
parser.add_argument('-i', '--inputfile', type=str, default=None, help='Training dataset file')
parser.add_argument('-t', '--testfile', type=str, default=None, help='Test dataset file')
parser.add_argument('--sep', type=str, default='\t', help='Field separator')
parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('-e', '--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--eval_steps', type=int, default=1000)
parser.add_argument('--valsplit', type=float, default=0.2)
parser.add_argument('--early_stopping', type=int, default=2, help='Number of epochs with no improvement before stopping')
parser.add_argument('-v', '--verbosity', type=int, default=1)
parser.add_argument('--smoothing_factor', type=float, default=0.1)
args = parser.parse_args()

LabS.seed = args.seed

if args.use_wandb:
    import wandb
    wandb.init(project=args.projectname, name="Label_Smoothing_Transformer")

if args.inputfile:
    train_dataset, val_dataset = LabS.prepare_datasets(args.inputfile, args.sep, args.valsplit)
    LabS.train_model(
        model_name=args.mname,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.local,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        early_stopping=args.early_stopping,
        project_name=args.projectname if args.use_wandb else None
    )

if args.testfile:
    test_dataset, _ = LabS.prepare_datasets(args.testfile, args.sep)
    metrics = LabS.test_model(
        model_name=args.local,
        test_dataset=test_dataset,
    )
    print(args.testfile)
    print("\t".join([f"{k} {metrics[k]:.3f}" for k in metrics]))

if args.use_wandb:
    wandb.finish()
