import xml.etree.cElementTree as ET
from DataExtraction.Paper import *


class XMLFile:
    def __init__(self, fileLocation):
        try:
            self.tree = ET.parse(fileLocation)
            self.location = fileLocation
        except:
            pass
        self.papers = []

    def getPapers(self, structure):#structure corresponds to xml file tree structure
        root = self.tree.getroot()

        for article in root.findall(structure):
            paper = Paper()
            for child in article:
                if child.tag == 'article_id':
                    paper.id = child.text
                elif child.tag == "title":
                    paper.title = child.text
                elif child.tag == "subtitle":
                    paper.subtitle = child.text
                elif child.tag == "abstract":
                    paper.abstract = child[0].text
                elif child.tag == "keywords":
                    for keyword in child:
                        paper.keywords.append(keyword.text)
                elif child.tag == "ccs2012":
                    for concept in child:
                        concept_id = ""
                        desc = ""
                        sig = ""
                        for conceptItem in concept:
                            if conceptItem.tag == "concept_desc":
                                desc = conceptItem.text
                            elif conceptItem.tag == "concept_significance":
                                sig = int(conceptItem.text)
                            elif conceptItem.tag == "concept_id":
                                id = conceptItem.text
                        paper.categoriesWithSignificance.append(Category(id,desc,sig))

            if len(paper.categoriesWithSignificance)>0:
                paper.getCategorywithHighestSignificance()
            paper.clean_XML_tags()
            self.papers.append(paper)
        return self.papers

class Category:
    def __init__(self,id, description, significance):
        self.id = id
        self.description = description
        self.significance = significance

