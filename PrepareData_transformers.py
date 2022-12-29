#Example command
# python PrepareData_transformers.py --model_type 'SciBERT' --preprocessed_data_loc '/home/msadat3/HTC/SciHTC_data/SciHTC_preprocessed/' --output_dir '/home/msadat3/HTC/SciHTC_data/SciBERT_without_keywords/' --use_keywords 'no'


import numpy as np
import pandas
from transformers import *
import string
from Utils import *
import os.path as p
import os
import nltk
import argparse
from DataPreparation.Data_Subset import Data_Subset

def Tokenize_Input(text, add_special_tokens):
	text = str(text)
	encoded = tokenizer.encode(text,add_special_tokens=add_special_tokens)
	return encoded

def pad_seq(seq,max_len,pad_idx):
	if len(seq)>max_len:
		sep = seq[-1]
		seq = seq[0:max_len-1]
		seq.append(sep)
	while len(seq) != max_len:
		seq.append(pad_idx)
	return seq

def get_attention_masks(X):
	attention_masks = []

	# For each sentence...
	for sent in X:
		att_mask = [int(token_id != tokenizer.pad_token_id) for token_id in sent]

		# Store the attention mask for this sentence.
		att_mask = np.asarray(att_mask)
		attention_masks.append(att_mask)
	attention_masks = np.asarray(attention_masks)
	return attention_masks

def tag_keywords_or_not(input_tokens, keywords_tokens):
	punctuations = ' '.join(string.punctuation)
	punctuations = tokenizer.encode(punctuations)

	Keyword_labels = []
	for token in input_tokens:
		if (token in keywords_tokens) and (token not in punctuations):
			Keyword_labels.append(1)
		else:
			Keyword_labels.append(0)
	return Keyword_labels



def prepare_data_for_transformers(output_location, trainingSet, testingSet, devSet, use_keywords=False):
	if p.exists(output_location) == False:
		os.mkdir(output_location)
	column_name = ""
	if use_keywords == 'yes':
		column_name = "Preprocessed_title_abstract_keywords"
	elif use_keywords == 'no':
		column_name = "Preprocessed_title_abstract"

	# print(use_keywords, column_name)
	# quit()
	X_train  = trainingSet.apply(lambda x: Tokenize_Input(x[column_name], add_special_tokens=True), axis=1)
	X_test  = testingSet.apply(lambda x: Tokenize_Input(x[column_name], add_special_tokens=True), axis=1)
	X_dev = devSet.apply(lambda x: Tokenize_Input(x[column_name], add_special_tokens=True), axis=1)

	X_train = pandas.Series(X_train)
	X_test = pandas.Series(X_test)
	X_dev = pandas.Series(X_dev)

	trainingSet['BERT_tokenized'] = X_train
	testingSet['BERT_tokenized'] = X_test
	devSet['BERT_tokenized'] = X_dev

	max_len = 0
	for x in X_train:
		if len(x) > max_len:
			max_len = len(x)
	for x in X_test:
		if len(x) > max_len:
			max_len = len(x)
	for x in X_dev:
		if len(x) > max_len:
			max_len = len(x)
	#print(max_len)

	X_train = X_train.apply(pad_seq, max_len=max_len, pad_idx=tokenizer.pad_token_id)
	X_test = X_test.apply(pad_seq, max_len=max_len, pad_idx=tokenizer.pad_token_id)
	X_dev = X_dev.apply(pad_seq, max_len=max_len, pad_idx=tokenizer.pad_token_id)

	X_train = np.array(X_train.values.tolist())
	X_test = np.array(X_test.values.tolist())
	X_dev = np.array(X_dev.values.tolist())

	att_mask_train = get_attention_masks(X_train)
	att_mask_test = get_attention_masks(X_test)
	att_mask_dev = get_attention_masks(X_dev)

	save_data(X_train, output_location+'X_train.pkl')
	save_data(X_test, output_location + 'X_test.pkl')
	save_data(X_dev, output_location + 'X_dev.pkl')

	save_data(att_mask_train, output_location + 'att_mask_train.pkl')
	save_data(att_mask_test, output_location + 'att_mask_test.pkl')
	save_data(att_mask_dev, output_location + 'att_mask_dev.pkl')

   
	print('Prepared data shape:')
	print(X_train.shape, att_mask_train.shape)
	print(X_test.shape, att_mask_test.shape)
	print(X_dev.shape, att_mask_dev.shape)
  

	if use_keywords == False:
		X_train_keywords = trainingSet.apply(lambda x: Tokenize_Input(x['Preprocessed_keywords'], add_special_tokens=False), axis=1)
		X_test_keywords = testingSet.apply(lambda x: Tokenize_Input(x['Preprocessed_keywords'], add_special_tokens=False), axis=1)
		X_dev_keywords = devSet.apply(lambda x: Tokenize_Input(x['Preprocessed_keywords'], add_special_tokens=False), axis=1)

		X_train_keywords = pandas.Series(X_train_keywords)
		X_test_keywords = pandas.Series(X_test_keywords)
		X_dev_keywords = pandas.Series(X_dev_keywords)

		trainingSet['BERT_tokenized_keywords'] = X_train_keywords
		testingSet['BERT_tokenized_keywords'] = X_test_keywords
		devSet['BERT_tokenized_keywords'] = X_dev_keywords

		y_train_keywords = trainingSet.apply(lambda x: tag_keywords_or_not(x['BERT_tokenized'], x['BERT_tokenized_keywords']), axis=1)
		y_test_keywords = testingSet.apply(
			lambda x: tag_keywords_or_not(x['BERT_tokenized'], x['BERT_tokenized_keywords']), axis=1)
		y_dev_keywords = devSet.apply(
			lambda x: tag_keywords_or_not(x['BERT_tokenized'], x['BERT_tokenized_keywords']), axis=1)

		y_train_keywords = pandas.Series(y_train_keywords)
		y_test_keywords = pandas.Series(y_test_keywords)
		y_dev_keywords = pandas.Series(y_dev_keywords)

		y_train_keywords = y_train_keywords.apply(pad_seq, max_len=max_len, pad_idx=tokenizer.pad_token_id)
		y_test_keywords = y_test_keywords.apply(pad_seq, max_len=max_len, pad_idx=tokenizer.pad_token_id)
		y_dev_keywords = y_dev_keywords.apply(pad_seq, max_len=max_len, pad_idx=tokenizer.pad_token_id)

		y_train_keywords = np.array(y_train_keywords.values.tolist())
		y_test_keywords = np.array(y_test_keywords.values.tolist())
		y_dev_keywords = np.array(y_dev_keywords.values.tolist())

		save_data(y_train_keywords, output_location + 'y_train_keywords.pkl')
		save_data(y_test_keywords, output_location + 'y_test_keywords.pkl')
		save_data(y_dev_keywords, output_location + 'y_dev_keywords.pkl')

		print('Keyword labels shapes:')
		print(y_train_keywords.shape)
		print(y_test_keywords.shape)
		print(y_dev_keywords.shape)


parser = argparse.ArgumentParser(description='Prepare data for BERT/SciBERT models.')
parser.add_argument("--model_type", type=str, help="Type of the model: BERT/SciBERT.")
parser.add_argument("--use_keywords", type=str, help="Include keywords with input or not.")
parser.add_argument("--preprocessed_data_loc", type=str, help="Directory containing the pre-processed data.")
parser.add_argument("--output_dir", type=str, help="Output directory for saving the prepared data for experiments.")

args = parser.parse_args()

trainingSet = pandas.read_csv(args.preprocessed_data_loc+'train_preprocessed.csv')
testingSet = pandas.read_csv(args.preprocessed_data_loc+'test_preprocessed.csv')
devSet = pandas.read_csv(args.preprocessed_data_loc+'dev_preprocessed.csv')


if args.model_type == 'SciBERT':
	tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased', do_lower_case=True)
elif args.model_type == 'BERT':
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

prepare_data_for_transformers(args.output_dir, trainingSet, testingSet, devSet, use_keywords=args.use_keywords)




