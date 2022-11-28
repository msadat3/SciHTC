##This script extracts the data for SciHTC from ACM proceedings and saves it in a CSV file.
#Example command: python ExtractData.py --proceedings_loc '/home/msadat3/HTC/proceedings/' --output_csv_loc '/home/msadat3/HTC/ExtractedData.csv'

from DataExtraction.XMLFile import *
import os
import pandas
import pickle
from HierarchyTree.HierarchyTreeFile import HierarchyTree
import argparse

parser = argparse.ArgumentParser(description='Extract data for SciHTC from ACM proceedings.')
parser.add_argument("--proceedings_loc", type=str, help="Location of a directory containing the ACM proceedings in XML format.")
parser.add_argument("--output_csv_loc", type=str, help="Location of the output CSV file to save the extracted data.")
args = parser.parse_args()

HT = HierarchyTree('ACM Tree')
HT.buildTree('./HierarchyTree/CCS 2012.html')


def post_process_category_info(category_info):
    category_info = str(category_info)
    if 'CCS' in category_info:
        return category_info
    if '~' in category_info:
        categories = category_info.split('~')
    else:
        categories = [category_info]

    curr_parent = HT.root
    consistent_labels = "CCS"
    curr_sign = '->'
    for category in categories:
        category_node = HT.findNodeInTree(category)
        if category_node == None:
            break

        if category_node.parent != curr_parent:
            curr_sign = '~'
        consistent_labels+=curr_sign+category
        curr_parent = category_node
        
    return consistent_labels


proceedings_folder_location = args.proceedings_loc
output_csv_location = args.output_csv_loc


ids = []
titles = []
subtitles = []
abstracts = []
keywords = []
categories = []

paper_count = 0

for root, dirs, files in os.walk(proceedings_folder_location):
    for file_name in files:
        if file_name.endswith('.xml'):
            try:
                path = os.path.join(root, file_name)
                file = XMLFile(path)

                papers1 = file.getPapers('./content/section/article_rec')
                file.papers = []
                papers2 = file.getPapers('./content/article_rec')
                papers = papers1 + papers2
                
                for paper in papers:
                    # print(paper_count)
                    paper_count+=1
                    if paper.categorywithHighestSignificance != None:
                        if paper.title !="" and paper.abstract!= "" and paper.categorywithHighestSignificance.description!="" and len(paper.keywords) > 0:
                            ids.append(paper.id)
                            titles.append(paper.title)
                            subtitles.append(paper.subtitle)
                            abstracts.append(paper.abstract)
                            keywords.append(paper.keywords)
                            categories.append(paper.categorywithHighestSignificance.description)
            
            except Exception as e:
                print(e, file_name)
                pass


ExtractedData = pandas.DataFrame({'id':ids, 'Title':titles, 'Subtitle':subtitles, 'Abstract': abstracts, 'Keywords': keywords, 'Category': categories})
ExtractedData = ExtractedData.drop_duplicates('id')
ExtractedData['Category'] = ExtractedData['Category'].apply(post_process_category_info)
ExtractedData = ExtractedData.loc[~(ExtractedData['Category']=='CCS')]
ExtractedData.to_csv(output_csv_location)



