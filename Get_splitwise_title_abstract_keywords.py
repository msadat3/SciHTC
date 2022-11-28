# This script is for obtaining the title, subtitle, abstract and keywords from the extracted data for SciHTC for train, test and dev splits given the csv files containing the ids and categories belonging to each of the splits. Use the script named 'ExtractData.py' to extract the data from ACM proceedings before running this script.
#Example command: python Get_splitwise_title_abstract_keywords.py --extracted_data_loc '/home/msadat3/SciHTC_data/ExtractedData.csv' --directory_for_splitwise_ids_and_categories '/home/msadat3/SciHTC_data/IDs_and_categories/' --output_directory '/home/msadat3/SciHTC_data/Titles_abstracts_keywords_included/'

import pandas
import argparse

parser = argparse.ArgumentParser(description='.')
parser.add_argument("--extracted_data_loc", type=str, help="Location of the CSV file containing the data extracted from ACM proceedings.")
parser.add_argument("--directory_for_splitwise_ids_and_categories", type=str, help="Directory containing the split-wise ids and categories for the papers in SciHTC.")
parser.add_argument("--output_directory", type=str, help="Directory to save the files containing all relevant data for each split.")

args = parser.parse_args()

ExtractedData = pandas.read_csv(args.extracted_data_loc)

train = pandas.read_csv(args.directory_for_splitwise_ids_and_categories+'/train_ids_all_categories.csv')
test = pandas.read_csv(args.directory_for_splitwise_ids_and_categories+'/test_ids_all_categories.csv')
dev = pandas.read_csv(args.directory_for_splitwise_ids_and_categories+'/dev_ids_all_categories.csv')

print(train.shape, test.shape, dev.shape)

train_with_all_columns = train.merge(ExtractedData[['id','Title','Subtitle', 'Abstract', 'Keywords']], on=['id'], how='left')
test_with_all_columns = test.merge(ExtractedData[['id','Title','Subtitle', 'Abstract', 'Keywords']], on=['id'], how='left')
dev_with_all_columns = dev.merge(ExtractedData[['id','Title','Subtitle', 'Abstract', 'Keywords']], on=['id'], how='left')

print(train_with_all_columns.shape)
print(test_with_all_columns.shape)
print(dev_with_all_columns.shape)

train_with_all_columns.to_csv(args.output_directory+'/train.csv')
test_with_all_columns.to_csv(args.output_directory+'/test.csv')
dev_with_all_columns.to_csv(args.output_directory+'/dev.csv')
