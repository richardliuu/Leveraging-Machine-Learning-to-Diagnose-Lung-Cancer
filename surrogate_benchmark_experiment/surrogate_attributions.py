"""
This file will get local surrogate attributions 

Decision tree: naturally brought by feautre importance module from sklearn
"""

class TreeAttributions:
    def __init__(self):
        self.tree = None

    def local_attributions(self, X):
        """
        Need to get each node of the decision tree's path 

        Then get how much that node reduced impurity 

        Correspond this to the right feature 
        """

        #return attributions