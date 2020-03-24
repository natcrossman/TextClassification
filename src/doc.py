## @package Doc.py
# Document structure:
# This package contains a document class which is designed to contain the raw text document information. 
# This purpose is to break down the wrong document information into accessible formats.  
# Index class contains a list of IndexItems, stored in a dictionary type for easier access
#@copyright     All rights are reserved, this code/project is not Open Source or Free
#@bug           None Documented     
#@author        Nathaniel Crossman & Adam
#



##
# @brief This is our Document class and its purpose is to break down the wrong document information into accessible formats.
# 
#
#
# @bug       None documented yet   
#
class Document:
    def __init__(self, docid, subject, message, body):
        self.docID = docid
        self.title = subject
        self.author = message
    


    # add more methods if needed


class Collection:
    ''' a collection of documents'''

    def __init__(self):
        self.docs = {} # documents are indexed by docID

    def find(self, docID):
        ''' 
        return a document object
        fixed to python 3.7
        '''
        key = self.docs.keys() 
        if docID in key:
            return self.docs[docID]
        else:
            return None

        # if self.docs.has_key(docID):
        #     return self.docs[docID]
        # else:
        #     return None


