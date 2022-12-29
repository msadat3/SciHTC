#This script reads the data from CSV files, pre-processes the inputs (title, abstract, keywords) and prepares the labels for both flat and hierarchical multi-label classification.
#Example command:
#python Preprocess_data.py --title_abstract_keywords_loc '/home/msadat3/HTC/SciHTC_data/IDs_title_abstract_keywords/' --category_loc '/home/msadat3/HTC/SciHTC_data/IDs_and_categories/' --output_dir '/home/msadat3/HTC/SciHTC_data/SciHTC_preprocessed/' --title_abstract_max_token 100 --keywords_max_token 15 

import pandas
from Utils import *
import argparse
from Data_Subset import Data_Subset
import os.path as p
import os

def convert_tokenized_to_single_string(tokens):
	return ' '.join(tokens)

def prepare_labels(output_location, dataframe, data_suffix, all_categories):
	y = dataframe.loc[:,all_categories]
	y.to_csv(output_location+'y_'+data_suffix+'.csv')

	y_flat = []
	for i in range(y.shape[0]):
		temp = []
		for category in all_categories: 
			temp.append(y.iloc[i][category])
		y_flat.append(temp)
	print(len(y_flat[0]))
	save_data(y_flat, output_location+'y_'+data_suffix+'_flat.pkl')

parser = argparse.ArgumentParser(description='Preprocess SciHTC data.')
parser.add_argument("--title_abstract_keywords_loc", type=str, help="Directory containing the csv files with the titles, abstract and keywords.")
parser.add_argument("--category_loc", type=str, help="Directory containing the csv files with the categories.")
parser.add_argument("--output_dir", type=str, help="Output directory for saving the pre-processed data.")
parser.add_argument("--title_abstract_max_token", type=int,default=100, help="Maximum token count allowed in title+abstract.")
parser.add_argument("--keywords_max_token", type=int,default=15, help="Maximum token count allowed in keywords.")


args = parser.parse_args()

trainingSet_features = pandas.read_csv(args.title_abstract_keywords_loc+'train_title_abstract_keywords.csv')
testingSet_features = pandas.read_csv(args.title_abstract_keywords_loc+'test_title_abstract_keywords.csv')
devSet_features = pandas.read_csv(args.title_abstract_keywords_loc+'dev_title_abstract_keywords.csv')

trainingSet_categories = pandas.read_csv(args.category_loc+'train_ids_83_categories.csv')
testingSet_categories = pandas.read_csv(args.category_loc+'test_ids_83_categories.csv')
devSet_categories = pandas.read_csv(args.category_loc+'dev_ids_83_categories.csv')

trainingSet = Data_Subset('',trainingSet_features.merge(trainingSet_categories,on='id'))
testingSet = Data_Subset('',testingSet_features.merge(testingSet_categories,on='id'))
devSet = Data_Subset('',devSet_features.merge(devSet_categories,on='id'))


trainingSet.preProcessAll()
testingSet.preProcessAll()
devSet.preProcessAll()

trainingSet.combine_titles_abstracts(args.title_abstract_max_token)
testingSet.combine_titles_abstracts(args.title_abstract_max_token)
devSet.combine_titles_abstracts(args.title_abstract_max_token)
trainingSet.combine_titles_abstracts_keywords(args.title_abstract_max_token, args.keywords_max_token)
testingSet.combine_titles_abstracts_keywords(args.title_abstract_max_token, args.keywords_max_token)
devSet.combine_titles_abstracts_keywords(args.title_abstract_max_token, args.keywords_max_token)


trainingSet.dataframe['Preprocessed_title_abstract'] = trainingSet.dataframe['Preprocessed_title_abstract'].apply(convert_tokenized_to_single_string)
testingSet.dataframe['Preprocessed_title_abstract'] = testingSet.dataframe['Preprocessed_title_abstract'].apply(convert_tokenized_to_single_string)
devSet.dataframe['Preprocessed_title_abstract'] = devSet.dataframe['Preprocessed_title_abstract'].apply(convert_tokenized_to_single_string)
trainingSet.dataframe['Preprocessed_title_abstract_keywords'] = trainingSet.dataframe['Preprocessed_title_abstract_keywords'].apply(convert_tokenized_to_single_string)
testingSet.dataframe['Preprocessed_title_abstract_keywords'] = testingSet.dataframe['Preprocessed_title_abstract_keywords'].apply(convert_tokenized_to_single_string)
devSet.dataframe['Preprocessed_title_abstract_keywords'] = devSet.dataframe['Preprocessed_title_abstract_keywords'].apply(convert_tokenized_to_single_string)


if p.exists(args.output_dir) == False:
	os.mkdir(args.output_dir)

trainingSet.dataframe.to_csv(args.output_dir+'train_preprocessed.csv')
testingSet.dataframe.to_csv(args.output_dir+'test_preprocessed.csv')
devSet.dataframe.to_csv(args.output_dir+'dev_preprocessed.csv')


categories = []

for category in trainingSet.dataframe['Category'].tolist():
	category_split = category.split('->')
	categories+=category_split[1:]
categories = list(set(categories))
print('Total number of categories: ', len(categories))

trainingSet.multiLabelData(categories)
testingSet.multiLabelData(categories)
devSet.multiLabelData(categories)

prepare_labels(args.output_dir, trainingSet.dataframe, 'train', categories)
prepare_labels(args.output_dir, testingSet.dataframe, 'test', categories)
prepare_labels(args.output_dir, devSet.dataframe, 'dev', categories)

