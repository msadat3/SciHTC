# Hierarchical Multi-Label Classification of Scientific Documents
This repository contains the dataset named "SciHTC" introduced in the EMNLP 2022 paper "Hierarchical Multi-Label Classification of Scientific Documents." The code for training and testing the baselines and proposed models will be released soon.

## Abstract
Automatic topic classification has been studied extensively to assist managing and indexing scientific documents in a digital collection. With the large number of topics being available in recent years, it has become necessary to arrange them in a hierarchy. Therefore, the automatic classification systems need to be able to classify the documents hierarchically. In addition, each paper is often assigned to more than one relevant topic. For example, a paper can be assigned to several topics in a hierarchy tree. In this paper, we introduce a new dataset for hierarchical multi-label text classification (HMLTC) of scientific papers called SciHTC, which contains 186,160 papers and 1,233 categories from the ACM CCS tree. We establish strong baselines for HMLTC and propose a multi-task learning approach for topic classification with keyword labeling as an auxiliary task. Our best model achieves a Macro-F1 score of 34.57% which shows that this dataset provides significant research opportunities on hierarchical scientific topic classification.

## Dataset
We derive SciHTC from papers available in the ACM digital library. The category information of the papers in our dataset were defined based on the category hierarchy tree named ['CCS'](https://dl.acm.org/ccs) created by ACM. Specifically, each paper is assigned to a sub-branch of the hierarchy tree which was specified by their respective authors. In addition to the category information, SciHTC also contains the title, abstract and author-specified keywords of each paper. For a more detailed description of our dataset construction process and its properties, we refer the reader to our [paper](paper_url). 

### Dataset Size
SciHTC contains 184,160 papers in total. We split the dataset in a 80:10:10 ratio for train, test and development sets, respectively. The number of papers in each split are as follows:
  * Train: 148,928 

  * Test: 18,616

  * Dev: 18,616 

  * Total: 184,160.


### Dataset Access
We make the ACM paper IDs and the labels for the train, test and dev splits publicly available [here](https://drive.google.com/drive/folders/1uRh5A-GpFRxA_QLzgN_D-y8G5j6JpZPJ?usp=sharing). Please email us at msadat3@uic.edu for getting access to the full dataset.


## Model Training and Testing
Coming soon!

## Citation
If you use this dataset, please cite our paper:

```
@inproceedings{sadat-caragea-2022-scihtc,
    title = "Hierarchical Multi-Label Classification of Scientific Documents",
    author = "Sadat, Mobashir  and
      Caragea, Cornelia",
    booktitle = "Proceedings of The 2022 Conference on Empirical Methods in Natural Language Processing (Volume 1: Long Papers)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
}
```

## Contact
Please contact us at msadat3@uic.edu with any questions.
