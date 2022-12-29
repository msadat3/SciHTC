# Hierarchical Multi-Label Classification of Scientific Documents
This repository contains the dataset named "SciHTC" introduced in the EMNLP 2022 paper "Hierarchical Multi-Label Classification of Scientific Documents." The code for training and testing the baselines and proposed models will be released soon.

## Abstract
Automatic topic classification has been studied extensively to assist managing and indexing scientific documents in a digital collection. With the large number of topics being available in recent years, it has become necessary to arrange them in a hierarchy. Therefore, the automatic classification systems need to be able to classify the documents hierarchically. In addition, each paper is often assigned to more than one relevant topic. For example, a paper can be assigned to several topics in a hierarchy tree. In this paper, we introduce a new dataset for hierarchical multi-label text classification (HMLTC) of scientific papers called SciHTC, which contains 186,160 papers and 1,233 categories from the ACM CCS tree. We establish strong baselines for HMLTC and propose a multi-task learning approach for topic classification with keyword labeling as an auxiliary task. Our best model achieves a Macro-F1 score of 34.57% which shows that this dataset provides significant research opportunities on hierarchical scientific topic classification.

## Dataset
We derive SciHTC from papers available in the ACM digital library. The category information of the papers in our dataset were defined based on the category hierarchy tree named ['CCS'](https://dl.acm.org/ccs) created by ACM. Specifically, each paper is assigned to a sub-branch of the hierarchy tree which was specified by their respective authors. In addition to the category information, SciHTC also contains the title, abstract and author-specified keywords of each paper. For a more detailed description of our dataset construction process and its properties, we refer the reader to our [paper](https://arxiv.org/pdf/2211.02810.pdf). 

### Dataset Size
SciHTC contains 184,160 papers in total. We split the dataset in a 80:10:10 ratio for train, test and development sets, respectively. The number of papers in each split are as follows:
  * Train: 148,928 

  * Test: 18,616

  * Dev: 18,616 

  * Total: 184,160.


### Dataset Access
We make the ACM paper IDs and the labels for the train, test and dev splits publicly available [here](https://drive.google.com/drive/folders/1uRh5A-GpFRxA_QLzgN_D-y8G5j6JpZPJ?usp=sharing). Please email us at msadat3@uic.edu/sadat.mobashir@gmail.com/cornelia@uic.edu for getting access to the full dataset.

Alternatively, you can reconstruct the SciHTC dataset based on the released paper IDs for each split and extracting the titles, abstracts and keywords using our released scripts from ACM proceedings as follows.

### Dataset Reconstruction

ACM makes the proceedings data available on request for research purposes. After obtaining the proceedings data from ACM, follow the steps below to reconstruct SciHTC.

#### Step 1:
Extract the ids, titles, abstracts and keywords of the papers used to create SciHTC by using the script named 'ExtractData.py.' Given the location of the directory containing ACM proceedings, this script with extract the necessary information for each paper and put them into a specified location in CSV format. An example command for using this script can be seen below.

```
python ExtractData.py --proceedings_loc '/home/msadat3/HTC/proceedings/' --output_csv_loc '/home/msadat3/SciHTC_data/ExtractedData.csv' 
```
#### Step 2:
After extracting the data, the titles, abstracts and keywords for the train, test and dev splits can be obtained using the script named 'Get_splitwise_title_abstract_keywords.py.' Given the location of the CSV file created in the previous step and the location of the directory containing the split-wise ids and categories (made publicly available in this repository), this script extracts the titles, abstracts and keywords and saves them in CSV files along with their respective ids and categories for each of the three splits in a specified output directory. An example command for this step can be seen below.

```
python Get_splitwise_title_abstract_keywords.py --extracted_data_loc '/home/msadat3/SciHTC_data/ExtractedData.csv' --directory_for_splitwise_ids_and_categories '/home/msadat3/SciHTC_data/IDs_and_categories/' --output_directory '/home/msadat3/SciHTC_data/Titles_abstracts_keywords_included/'
```

## Data pre-processing
After obtaining the CSV files for both category information and the input texts (title, abstract, keywords), use the script named "Preprocess_data.py" to pre-process the dataset. In addition to pre-processing the titles, abstracts and keywords, this file will also prepare the labels for both flat and hierarchical multi-label experiments. An example command can be seen below:

```
python Preprocess_data.py --title_abstract_keywords_loc '/home/msadat3/HTC/SciHTC_data/IDs_title_abstract_keywords/' --category_loc '/home/msadat3/HTC/SciHTC_data/IDs_and_categories/' --output_dir '/home/msadat3/HTC/SciHTC_data/SciHTC_preprocessed/' --title_abstract_max_token 100 --keywords_max_token 15 
```

## Model Training and Testing

### BERT/SciBERT models

#### Prepare Data
To prepare the data for BERT/SciBERT models, use the script named 'PrepareData_transformers.py.' This script reads the pre-processed data and prepares encoded inputs for BERT/SciBERT models. Example command:

```
python PrepareData_transformers.py --model_type 'SciBERT' --preprocessed_data_loc '/home/msadat3/HTC/SciHTC_data/SciHTC_preprocessed/' --output_dir '/home/msadat3/HTC/SciHTC_data/SciBERT_without_keywords/' --use_keywords 'no'
```
When the --use_keywords parameter is set to 'no', both the encoded versions of the input text (title+abstract) and the keyword labels will be created. When it is set to 'yes', only the encoded versions of the title+abstract+keywords will be created.

#### With/without keywords
##### Flat models
Use the script named 'Train_and_test_flat_models_transformers.py' to train and evaluate the flat models. Based on the type of model you want to train (with/without keywords), make sure to specify the correct data directory (with/without keywords) created using the script from the previous step. Example command:

```
python Train_and_test_flat_models_transformers.py --model_type 'SciBERT' --prepared_data_dir '/home/msadat3/HTC/SciHTC_data/SciBERT_with_keywords/' --prepared_labels_dir '/home/msadat3/HTC/SciHTC_data/SciHTC_preprocessed/' --checkpoint_location '/home/msadat3/HTC/SciHTC_data/SciBERT_with_keywords/flat_model/' --test_output_location '/home/msadat3/HTC/SciHTC_data/SciBERT_with_keywords/flat_model/'
```
##### Hierarchical models
Coming soon!

#### Multi-tasking models
##### Flat models
Use the script named 'Train_and_test_flat_models_transformers_multi_tasking.py' to train and evaluate the flat multi-tasking models. Make sure to specify the correct data directory (with or without keywords) created using the script from the previous step. Example command:

```
python Train_and_test_flat_models_transformers_multi_tasking.py --model_type 'SciBERT' --prepared_data_dir '/home/msadat3/HTC/SciHTC_data/SciBERT_without_keywords/' --prepared_labels_dir '/home/msadat3/HTC/SciHTC_data/SciHTC_preprocessed/' --checkpoint_location '/home/msadat3/HTC/SciHTC_data/SciBERT_without_keywords/flat_model_multitasking/' --test_output_location '/home/msadat3/HTC/SciHTC_data/SciBERT_without_keywords/flat_model_multitasking/' --classification_loss_weight 1 --keyword_labeling_loss_weight 1
```
##### Hierarchical models
Coming soon!

### BiLSTM/CNN models
Coming soon!

## Citation
If you use this dataset, please cite our paper:

```
@inproceedings{sadat-caragea-2022-scihtc,
    title = "Hierarchical Multi-Label Classification of Scientific Documents",
    author = "Sadat, Mobashir  and Caragea, Cornelia",
    booktitle = "Proceedings of The 2022 Conference on Empirical Methods in Natural Language Processing (Volume 1: Long Papers)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
}
```

## Contact
Please contact us at msadat3@uic.edu, sadat.mobashir@gmail.com, cornelia@uic.edu with any questions.
