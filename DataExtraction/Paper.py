import string
import re


class Paper:
    def __init__(self):
        self.id = ""
        self.title = ""
        self.subtitle = ""
        self.abstract = ""
        self.keywords = []
        self.categoriesWithSignificance = []
        self.categorywithHighestSignificance = None
        

    def clean_XML_tags(self):
        self.title = self.preProcessor(self.title)
        self.abstract = self.preProcessor(self.abstract)
        for i in range(0,len(self.keywords)):
            self.keywords[i] = self.preProcessor(self.keywords[i])


    def getCategorywithHighestSignificance(self):
        self.categorywithHighestSignificance = self.categoriesWithSignificance[0]
        for category in self.categoriesWithSignificance:
            if int(self.categorywithHighestSignificance.significance) < int(category.significance):
                self.categorywithHighestSignificance = category

    def preProcessor(self, text):
        text = str(text)
        regex = re.compile(r'<.*?>')
        text = re.sub(regex, '', text)
        return text

    def get_author_ids(self):
        authors_ids = []
        for author in self.authors:
            authors_ids.append(author.id)
        return authors_ids
