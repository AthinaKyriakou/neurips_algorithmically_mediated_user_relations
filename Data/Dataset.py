import scipy.sparse as sp
import numpy as np

# BASED ON: RecSys2019_DeepLearning_Evaluation/Conferences/WWW/NeuMF_github/Dataset.py

class Dataset(object):

    def __init__(self, num_users, num_items, isHeader, trainPath, testPath, validationPath = None, testNegativePath = None):
        
        print("Dataset: Creating dataset object")
        self.isHeader = isHeader
        self.num_users = num_users
        self.num_items = num_items
        
        self.trainMatrix = self.load_rating_file_as_matrix(trainPath)
        self.testRatings = self.load_rating_file_as_matrix(testPath)
        #self.testRatings = self.load_rating_file_as_list(testPath) # why as list
        if testNegativePath != None:
            self.testNegatives = self.load_negative_file(testNegativePath)
            assert len(self.testRatings) == len(self.testNegatives)
        
        #self.num_users, self.num_items = self.trainMatrix.shape

        
    def load_rating_file_as_list(self, filepath): #tofix
        ratingList = []
        with open(filepath, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename): #tofix
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    
    def load_rating_file_as_matrix(self, filename):
        mat = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        with open(filename, "r") as f:
            if self.isHeader: # if the first line of the file contains a header
                line = f.readline() 
            line = f.readline()
            while line != None and line != "":
                arr = line.split(",")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                mat[user-1, item-1] = rating
                line = f.readline()    
        return mat

        '''
        # Get number of users and items - seem to be the max userID + itemID found
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()    
        return mat
        '''
