#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import complexity
import argparse

parser = argparse.ArgumentParser(description="A Transformer Model for Complexity Classification")
parser.add_argument('-m', '--mname', type=str, default='xlm-roberta-base', help='Model name according to HuggingFace transformers')
parser.add_argument('-l', '--local', type=str, default=None, help='Directory for the local model')
parser.add_argument('-n', '--hubname', type=str, default=None, help='Model name to save to the hub')
parser.add_argument('-p', '--projectname', type=str, default=None, help='Project name to report at Wand')
parser.add_argument('-i', '--inputfile', type=str, default=None, help='one-doc-per-line training corpus')
parser.add_argument('-t', '--testfile', type=str, default=None, help='one-doc-per-line test only corpus')
parser.add_argument('--sep', type=str, default=',', help='Field separator')

parser.add_argument('-s', '--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('-e', '--epochs', type=int, default=4)
parser.add_argument( '--batch_size', type=int, default=16)
parser.add_argument( '--eval_steps', type=int, default=1000)
parser.add_argument( '--valsplit', type=float, default=0.2)

parser.add_argument('-v', '--verbosity', type=int, default=1)

args = parser.parse_args()

complexity.seed = args.seed

if args.inputfile:
    train_dataset, val_dataset = complexity.prepare_datasets(args.inputfile, args.sep, args.valsplit)

    complexity.train_model(
        model_name=args.mname,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.local,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        project_name=args.projectname
    )

if args.testfile:
    test_dataset, _ = complexity.prepare_datasets(args.testfile, args.sep)
    metrics=complexity.test_model(
        model_name=args.local,
        test_dataset=test_dataset,
        )
    print(args.testfile)
    print("\t".join([f"{k} {metrics[k]:.3f}" for k in metrics]))
