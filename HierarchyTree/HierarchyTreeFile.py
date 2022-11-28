import json
from bs4 import BeautifulSoup

class HierarchyTree:
    
    def __init__(self,name):
        self.name = name
        self.root = None
        self.TreeNodes = []

    def printName(self):
        print(self.name)

    def getAllNames(self):
        categories = []
        for node in self.TreeNodes:
            categories.append(node.name)
        return categories
    
    def buildTree(self, inputLocation): #call this to build the tree
        parsedHTML = BeautifulSoup(open(inputLocation), "html.parser")
        divForTree = parsedHTML.find("div",{"id":"holdflat"})
        HTMLTree = divForTree.find("ul")
        self.root = CategoryNode('0',"CCS", None,0)
        self.TreeNodes.append(self.root)
        for category in HTMLTree.find_all("li",recursive = False):
            headerCategory = category.find("div").find("a",{"class":"boxedlinkh"}).text
            headerCategoryid = category.find("div").find("a",{"class":"boxedlink"}).attrs['name']
            self.TreeNodes.append(CategoryNode(headerCategoryid,headerCategory,self.root, 1))
            self.getNodesFromHTML(category.find("ul"),headerCategory,2)
    
    def printTree(self):
        for node in self.TreeNodes:
            try:
                print("id: ",node.id,"Name: ",node.name," Parent: ",node.parent.name)
            except:
                print("id: ",node.id,"Name: ",node.name," Parent: ",None)


    def getNodesFromHTML(self, classHTML, categoryName,level):
        if classHTML.find("li") == None:
            return
        for subclass in classHTML.find_all("li", recursive= False):
            name = subclass.find("a").text

            id = subclass.find("a").attrs['name']
            self.TreeNodes.append(CategoryNode(id, name, self.findNodeInTree(categoryName),level))
            self.getNodesFromHTML(subclass.find("ul"), name,level+1)


    def findNodeInTree(self,name):
        nodes = []
        for node in self.TreeNodes:
            if node.name == name:
                nodes.append(node)
        if len(nodes)>=1:
            return nodes[0]
        else:
            return None

    def get_names_upto_level(self, node, highestLevel, names):
        if node.level == highestLevel:
            names.append(node.name)
            return
        else:
            names.append(node.name)
            node.findChildren(self.TreeNodes)
            for child in node.children:
                self.get_names_upto_level(child,highestLevel, names)

    def get_highest_level(self):
        categories = []
        levels = []
        CCS_level = -1
        for node in self.TreeNodes:
            if node.name == 'CCS':
                CCS_level = node.level
            categories.append(node.name)
            levels.append(node.level)

            if node.level == 6:
                node.findParentalHistory()
                print(node.name)
                print([n.name for n in node.parentalHistory])

        return max(levels), CCS_level
        



class CategoryNode:
    
    def __init__(self,id,name,parent,level):
        self.id = id
        self.name = name
        self.parent = parent
        self.children = []
        self.parentalHistory = []
        self.featureMLEs = []
        self.mixtureWeights = []
        self.level = level
              
    
    def isLeaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False
    def findChildren(self,TreeNodes):
        for node in TreeNodes:
            try:
                if node.parent.name == self.name:
                    if node not in self.children:
                        self.children.append(node)
            except:
                pass
        return self.children
        
    def findParentalHistory(self):
        parent = self.parent
        while parent != None:
            self.parentalHistory.append(parent)
            parent = parent.parent
    
