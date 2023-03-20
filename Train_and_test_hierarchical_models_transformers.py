# This script trains and evaluates hierarchical BERT/SciBERT models -- with/without keywords.

#Example command:

#With keywords:
#python Train_and_test_hierarchical_models_transformers.py --model_type 'SciBERT' --prepared_data_dir '/home/msadat3/HTC/SciHTC_data/SciBERT_with_keywords/' --prepared_labels_dir '/home/msadat3/HTC/SciHTC_data/SciHTC_preprocessed/' --models_base_directory '/home/msadat3/HTC/SciHTC_data/SciBERT_with_keywords/HR_models/'

#Without keywords:
#python Train_and_test_hierarchical_models_transformers.py --model_type 'SciBERT' --prepared_data_dir '/home/msadat3/HTC/SciHTC_data/SciBERT_without_keywords/' --prepared_labels_dir '/home/msadat3/HTC/SciHTC_data/SciHTC_preprocessed/' --models_base_directory '/home/msadat3/HTC/SciHTC_data/SciBERT_without_keywords/HR_models/' 

import os
from Train_and_test_utils import *
import argparse
from HierarchyTree.HierarchyTreeFile import HierarchyTree
import os.path as p
import pandas

parser = argparse.ArgumentParser(description='Train and test hierarchical BERT/SciBERT models.')
parser.add_argument("--model_type", type=str, help="Type of the model: BERT/SciBERT.")
parser.add_argument("--prepared_data_dir", type=str, help="Directory containing the data prepared using PrepareData_transformers.py script.")
parser.add_argument("--prepared_labels_dir", type=str, help="Directory containing the prepared labels.")
parser.add_argument("--device", type=str, default='cuda', help="Device to run the experiment.")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing the models.")
parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for model training")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
parser.add_argument("--models_base_directory", type=str, help="Directory to save the binary models.")
parser.add_argument("--highest_level", type=int, help="Deepest level of the hierarchy tree for the topics to experiment with.")

args = parser.parse_args()


if p.exists(args.models_base_directory) == False:
	os.mkdir(args.models_base_directory)

X_train = load_data(args.prepared_data_dir+"X_train.pkl")
X_test = load_data(args.prepared_data_dir+"X_test.pkl")
X_dev = load_data(args.prepared_data_dir+"X_dev.pkl")

att_mask_train = load_data(args.prepared_data_dir+'att_mask_train.pkl')
att_mask_test = load_data(args.prepared_data_dir+'att_mask_test.pkl')
att_mask_dev = load_data(args.prepared_data_dir+'att_mask_dev.pkl')

y_train = pandas.read_csv(args.prepared_labels_dir+"y_train.csv")
y_test = pandas.read_csv(args.prepared_labels_dir+"y_test.csv")
y_dev = pandas.read_csv(args.prepared_labels_dir+"y_dev.csv")

X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
X_dev = np.asarray(X_dev)

topics_considered = list(y_train.columns)


HT = HierarchyTree('ACM Tree')
HT.buildTree('./HierarchyTree/CCS 2012.html')
HT.root.findChildren(HT.TreeNodes)

level = 1

#Training
for child in HT.root.children:
	train_binary_hierarchical_models_for_all_children_transformers(child, level, args.highest_level, HT, X_train, att_mask_train, y_train, X_dev, att_mask_dev, y_dev, args.model_type, args.device, args.batch_size, args.learning_rate, args.num_epochs, args.gradient_accumulation_steps, args.models_base_directory, topics_considered)

#Testing

f1_test = {}
precision_test = {}
recall_test = {}
TP_test = {}
FP_test = {}
TN_test = {}
FN_test = {}
outputs_binary_all_class_test = {}

level = 1
for child in HT.root.children:
	test_binary_hierarchical_models_for_all_children_transformers(child, X_test, y_test, att_mask_test, level, args.highest_level, HT, args.models_base_directory, args.model_type, args.device, args.batch_size, f1_test, precision_test, recall_test, TP_test, FP_test, TN_test, FN_test, outputs_binary_all_class_test, topics_considered)

get_score_reports_average(f1_test, precision_test, recall_test, TP_test, FP_test, TN_test, FN_test, outputs_binary_all_class_test, args.models_base_directory + '/testing/')

