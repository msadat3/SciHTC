from _ast import keyword
import pandas as pd
import nltk
import re
import string
from nltk.stem import PorterStemmer
import numpy as np
import statistics
from Utils import *


class Data_Subset:
    def __init__(self, location, dataframe):
        self.location = location
        self.dataframe = dataframe
        try:
            if self.dataframe == None:
                self.dataframe = pd.read_csv(location)
        except Exception as e:
            pass
        self.dataframe = literal_all_cols(self.dataframe)

    def convert_keywords_to_one_string(self,keywords_list):
        keywords_list = [k for k in keywords_list if k != ',']
        joined = ' '.join(keywords_list)
        return joined

    

    def preProcessor(self, text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
        text = str(text)
        text = text.strip()
        text = text.lower()
        regex = re.compile(r'<.*?>')
        text = re.sub(regex, '', text)
        text = re.sub(r"http\S+", "", text)
        regex = re.compile(r'&#.*?;')
        text = re.sub(regex, ' ', text)
        text = re.sub('([!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~])', r' \1 ', text)
        text = text.replace('\\',' \\ ')
        text = re.sub('\s{2,}', ' ', text)

        text = nltk.word_tokenize(text)
        tokens = []
        unstemmed_tokens = []
        stemmer = PorterStemmer()

        s = nltk.corpus.stopwords.words('english')
        stopwords = []
        for word in s:
            stopwords.append(stemmer.stem(word))
        stopwords = set(stopwords)

        for token in text:
            try:
                if any(i.isdigit() for i in token) == False:
                    stemmed = stemmer.stem(token)
                    if stemmed not in stopwords:
                        tokens.append(stemmed)#For BiLSTM/CNN
                        unstemmed_tokens.append(token)#For transformers
                else:
                    tokens.append('[NUMBER]')
                    unstemmed_tokens.append('[NUMBER]')
            except Exception as e:
                print(e, token)
                pass
        return tokens, unstemmed_tokens

    def preProcessAll(self):
        self.dataframe['Preprocessed_title'] = [[] for _ in range(len(self.dataframe))]
        self.dataframe['Preprocessed_abstract'] = [[] for _ in range(len(self.dataframe))]
        self.dataframe['Preprocessed_keywords'] = [[] for _ in range(len(self.dataframe))]
        self.dataframe['Preprocessed_title_stemmed'] = [[] for _ in range(len(self.dataframe))]
        self.dataframe['Preprocessed_abstract_stemmed'] = [[] for _ in range(len(self.dataframe))]
        self.dataframe['Preprocessed_keywords_stemmed'] = [[] for _ in range(len(self.dataframe))]

        temp_stemmed, temp = zip(*self.dataframe['Title'].apply(self.preProcessor))
        self.dataframe['Preprocessed_title'] += temp
        self.dataframe['Preprocessed_title_stemmed'] += temp_stemmed

        temp_stemmed, temp = zip(*self.dataframe['Abstract'].apply(self.preProcessor))
        self.dataframe['Preprocessed_abstract'] += temp
        self.dataframe['Preprocessed_abstract_stemmed'] += temp_stemmed


        self.dataframe['Keywords'] = self.dataframe['Keywords'].apply(self.convert_keywords_to_one_string)
        temp_stemmed, temp = zip(*self.dataframe['Keywords'].apply(self.preProcessor))
        self.dataframe['Preprocessed_keywords'] += temp
        self.dataframe['Preprocessed_keywords_stemmed'] += temp_stemmed


    def mask_text_with_unk(self, text, vocabulary):
        text = [x if x in vocabulary else '[UNK]' for x in text]
        return text

    def mask_all_text_with_unk(self, vocabulary):
        self.dataframe['Preprocessed_stemmed_title_abstract'] = self.dataframe['Preprocessed_stemmed_title_abstract'].apply(self.mask_text_with_unk, vocabulary=vocabulary)
        self.dataframe['Preprocessed_stemmed_title_abstract_keywords'] = self.dataframe['Preprocessed_stemmed_title_abstract_keywords'].apply(self.mask_text_with_unk, vocabulary=vocabulary)
        self.dataframe['Preprocessed_keywords_stemmed'] = self.dataframe['Preprocessed_keywords_stemmed'].apply(self.mask_text_with_unk, vocabulary=vocabulary)

    
    def combine_title_abstract(self, title, abstract, max_len):
        combined = []
        combined+=title
        combined+=['.']
        combined+=abstract
        if len(combined) > max_len:
            combined = combined[0:max_len]
        return combined


    def combine_titles_abstracts(self, max_len):
        self.dataframe['Preprocessed_stemmed_title_abstract'] = self.dataframe.apply(lambda x: self.combine_title_abstract(x['Preprocessed_title_stemmed'], x['Preprocessed_abstract_stemmed'], max_len=max_len), axis=1)
        self.dataframe['Preprocessed_title_abstract'] = self.dataframe.apply(lambda x: self.combine_title_abstract(
                                                                                     x['Preprocessed_title'],
                                                                                     x['Preprocessed_abstract'],
                                                                                     max_len=max_len), axis=1)

    def combine_title_abstract_keywords(self, title, abstract, keywords, title_abstract_max, keywords_max):
        if len(keywords)>keywords_max:
            keywords = keywords[0:keywords_max]
        combined = []
        combined+=title
        combined += ['.']
        combined+=abstract
        combined += ['.']
        if len(combined) > title_abstract_max:
            combined = combined[0:title_abstract_max]

        combined+=keywords
        return combined

    def combine_titles_abstracts_keywords(self, title_abstract_max, keywords_max): #column dict should contain a key 'name' and 'max_len'
        self.dataframe['Preprocessed_stemmed_title_abstract_keywords'] = self.dataframe.apply(lambda x: self.combine_title_abstract_keywords(x['Preprocessed_title_stemmed'], x['Preprocessed_abstract_stemmed'], x['Preprocessed_keywords_stemmed'], title_abstract_max = title_abstract_max, keywords_max=keywords_max), axis=1)
        self.dataframe['Preprocessed_title_abstract_keywords'] = self.dataframe.apply(lambda x: self.combine_title_abstract_keywords(x['Preprocessed_title'], x['Preprocessed_abstract'], x['Preprocessed_keywords'], title_abstract_max = title_abstract_max, keywords_max = keywords_max), axis=1)


    def tagKeywordOrNot(self, keywords, tokens):
        seq = []
        punctuations = string.punctuation
        for token in tokens:
            if (token in keywords) and (token not in punctuations) and (token!='[UNK]') and (token!='[NUMBER]'):
                seq.append(1)
            else:
                seq.append(0)
        return seq
    def tagKeywordOrNotAll(self):
        self.dataframe['KeywordLabel'] = [[] for _ in range(len(self.dataframe))]
        self.dataframe['KeywordLabel']  = self.dataframe.apply(lambda x: self.tagKeywordOrNot(x['Preprocessed_keywords_stemmed'], x['Preprocessed_stemmed_title_abstract']), axis=1)


    def multiLabelData(self, cateogries):
        for category in cateogries:
            self.dataframe[category] = self.dataframe['Category'].apply(self.EncodeLabel, label = category)


    def EncodeLabel(self, categoryHirerarchy, label):
        splitted = categoryHirerarchy.split('->')
        splitted = set(splitted)
        if label in splitted:
            return 1
        else:
            return 0

    def convertToSeqAndPad(self, vector, feature_to_idx_dict, paddingLength):
        X = []
        for x in vector:
            if len(X) <= paddingLength:
                try:
                    idx = feature_to_idx_dict[x] 
                    X.append(idx)
                except:
                    X.append(feature_to_idx_dict['[UNK]'])
        if len(X)>paddingLength:
            X = X[0:paddingLength]
        else:
            while(len(X) < paddingLength):
                X.append(feature_to_idx_dict['[PAD]'])
        return X

    def convertAlltOSeqAndPad(self, matrix, feature_to_idx_dict, paddinLength):
        X_mat = []
        for x in matrix:
            #print('bal')
            x_vect = self.convertToSeqAndPad(x, feature_to_idx_dict, paddinLength)
            X_mat.append(x_vect)
        return X_mat

    def PadKeyordLabelALL(self, matrix, X_matrix, paddingLength):
        X = []
        for x, x_seq in zip(matrix, X_matrix):
            x = np.asarray(x)
            x = x.astype(int)
            x = x.tolist()
            if len(x) >= paddingLength:
                x = x[0:paddingLength]
            for i in range(0, len(x)):
                if x_seq[i] == 0:
                    x[i] = 0
            for i in range(len(x), paddingLength):
                x.append(0)
            X.append(x)
        return X


    def get_lengths_statistics(self, column_name):
        lengths = []
        for item in self.dataframe[column_name]:
            lengths.append(len(item))
        print('Max:', max(lengths), ', Min:', min(lengths), ', Mean:',statistics.mean(lengths), ', Median:', statistics.median(lengths), ', Stdv:',statistics.stdev(lengths))

