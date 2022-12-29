# This script trains and evaluates flat BERT/SciBERT models -- with/without keywords.

#Example command:

#With keywords:
#python Train_and_test_flat_models_transformers.py --model_type 'SciBERT' --prepared_data_dir '/home/msadat3/HTC/SciHTC_data/SciBERT_with_keywords/' --prepared_labels_dir '/home/msadat3/HTC/SciHTC_data/SciHTC_preprocessed/' --checkpoint_location '/home/msadat3/HTC/SciHTC_data/SciBERT_with_keywords/flat_model/' --test_output_location '/home/msadat3/HTC/SciHTC_data/SciBERT_with_keywords/flat_model/'

#Without keywords:
#python Train_and_test_flat_models_transformers.py --model_type 'SciBERT' --prepared_data_dir '/home/msadat3/HTC/SciHTC_data/SciBERT_without_keywords/' --prepared_labels_dir '/home/msadat3/HTC/SciHTC_data/SciHTC_preprocessed/' --checkpoint_location '/home/msadat3/HTC/SciHTC_data/SciBERT_without_keywords/flat_model/' --test_output_location '/home/msadat3/HTC/SciHTC_data/SciBERT_without_keywords/flat_model/'

import os
from Train_and_test_utils import *
import argparse

parser = argparse.ArgumentParser(description='Train and test flat BERT/SciBERT models.')
parser.add_argument("--model_type", type=str, help="Type of the model: BERT/SciBERT.")
parser.add_argument("--prepared_data_dir", type=str, help="Directory containing the data prepared using PrepareData_transformers.py script.")
parser.add_argument("--prepared_labels_dir", type=str, help="Directory containing the prepared labels.")
parser.add_argument("--device", type=str, default='cuda', help="Device to run the experiment.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing the models.")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for model training")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs.")
parser.add_argument("--checkpoint_location", type=str, help="Directory to save best checkpoint.")
parser.add_argument("--test_output_location", type=str, help="Directory for saving the test results.")

args = parser.parse_args()


X_train = load_data(args.prepared_data_dir+"X_train.pkl")
X_test = load_data(args.prepared_data_dir+"X_test.pkl")
X_dev = load_data(args.prepared_data_dir+"X_dev.pkl")

att_mask_train = load_data(args.prepared_data_dir+'att_mask_train.pkl')
att_mask_test = load_data(args.prepared_data_dir+'att_mask_test.pkl')
att_mask_dev = load_data(args.prepared_data_dir+'att_mask_dev.pkl')

y_train = load_data(args.prepared_labels_dir+"y_train_flat.pkl")
y_test = load_data(args.prepared_labels_dir+"y_test_flat.pkl")
y_dev = load_data(args.prepared_labels_dir+"y_dev_flat.pkl")

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_dev = np.asarray(X_dev)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
y_dev = np.asarray(y_dev)


train_flat_model_transformers(X_train, att_mask_train, y_train, X_dev, att_mask_dev, y_dev, args.model_type, args.device, args.batch_size, args.learning_rate, args.num_epochs, args.gradient_accumulation_steps, args.checkpoint_location)
test_flat_model_transformers(X_test, y_test, att_mask_test, args.model_type, args.device, args.batch_size, args.checkpoint_location, args.test_output_location)

