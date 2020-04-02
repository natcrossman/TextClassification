## @package index.py
# Index structure:
# The Index class contains a list of IndexItems, stored in a dictionary type for easier access
# each IndexItem contains the term and a set of PostingItems
# each PostingItem contains a document ID and a list of positions that the term occurs
#

#Within a document collection, we assume that each document has a unique
#serial number, known as the document identifier (docID ). During index construction,
#we can simply assign successive integers to each new document
#when it is first encountered. 
#
#The input to indexing is a list of normalized
#tokens for each document, which we can equally think of as a list of pairs of
#SORTING term and docID (NOTE Sort BOTH). The core indexing step is sorting this list
#so that the terms are alphabetical. Multiple occurrences of the same term from the same
#document are then merged And we increase TF. Instances of the same term are then grouped,
#and the result is split into a dictionary and postings.
#Since a term generally occurs in a number of documents,
#this data organization already reduces the storage requirements of
#the index. 

#The dictionary also records some statistics, such as the number of
#documents which contain each term (the document frequency, which is here
#also the length of each postings list). This information is not vital for a basic
#Boolean search engine, but it allows us to improve the efficiency of the search engine at query time, 
#and it is a statistic later used in many ranked retrieval
#models. 
#
#The postings are secondarily sorted by docID. This provides
#the basis for efficient query processing. This inverted index structure is essentially
#without rivals as the most efficient structure for supporting ad hoc
#text search
#
#@copyright     All rights are reserved, this code/project is not Open Source or Free
#@bug           None Documented     
#@author        Nathaniel Crossman & Adam
#


"""Internal libraries"""
#import doc
from util import Tokenizer


"""Outside libraries"""
import sys
import math
import json
import operator
import collections
import re
import numpy as np
import os.path
from os import path
import pickle

##
#This is our posting clas. 
# @brief The job of this class is to  store the document ID, 
# the position where the term occurred in the document, 
# and the frequency of each term in each document.
# 
#In Short, Each postings list stores the list of documents
# in which a term occurs, and may store other information such as the term frequency
# or the position(s) of the term in each document.
# We don't technically need to store the frequency as we can calculate it by looking at the positions
#
# @bug       None documented yet   
#
class Posting:
    ##
    #    @param         self
    #    @param         docID
    #    @return        None
    #    @brief         The constructor. 
    #    @exception     None documented yet
    ##
    def __init__(self, docID):
        self.__docID      = docID
        self.__positions  = []
        #  self.termFrequency = 0 not need
    
    def get_docID(self):
        return self.__docID 

    ##
    #   @brief         This method append a positions to our array
    #   @param         self
    #   @param         pos
    #   @return        None
    #   @exception     None
    ## 
    def append(self, pos):
        self.__positions.append(pos)

    ##
    #   @brief         This method sorts the positions array
    #   @param         self
    #   @return        None
    #   @bug           This need to be tested
    #   @exception     None
    ## 
    def sort(self):
        ''' sort positions'''
        self.__positions.sort()

    ##
    #   @brief         This method combines/merges two conditional array.
    #   @param         self
    #   @param         positions
    #   @return        None
    #   @exception     None
    ## 
    def merge(self, positions):
        self.__positions.extend(positions)
    ##
    #   @brief         This method returns the term freq by count the
    #                  positions this term appeared in a given document.
    #    Why?: A Boolean model only records term presence or absence, but often we
    #    would like to accumulate evidence, givingmoreweight to documents that
    #    have a term several times as opposed to ones that contain it only once. To
    #    be able to do this we need term frequency information TERM FREQUENCY (the number of times
    #    a term occurs in a document) in postings lists
    #   @param         self
    #   @param         positions
    #   @return        tf:int
    #   @exception     None
    ## 
    def term_freq(self):
      return  len(self.__positions)

    ##
    #   @brief         This method used for Json
    #   @param         self
    #   @return        (self.__docID, self.__positions)
    #   @exception     None
    ## 
    def get_info(self):
        return (self.__docID, self.__positions)
  

##
# @brief     Tested
#
# @bug       None documented yet   
#
class IndexItem:
     ##
    #    @param         self
    #    @param         term
    #    @return        None
    #    @brief         The constructor. 
    #    @exception     None documented yet
    ##
    def __init__(self, term):
        self.__term             = term
        self.__posting          = {} #postings are stored in a python dict for easier index building
        self.__sorted_postings  = [] # may sort them by docID for easier query processing
        self.__sorted_dict      = {} #not sure if need

    ##
    #   @brief         This method sets the posting list
    #   @param         self
    #   @return        None
    #   @exception     None
    ## 
    def set_posting_list(self, docID, posting):
        self.__posting[docID] = posting
    
    ##
    #   @brief         This method return the posting list
    #   @param         self
    #   @return        None
    #   @exception     None
    ## 
    def get_posting_list(self):
        return self.__posting

    ##
    #   @brief         This method adds a term position, for a Document to the postings list.
    # If this is the first time a document has been added to the posting list,
    # the method creates a new posting (with docID) 
    # and then adds this the position the term was in the document.
    # Otherwise, This method just adds the new position.
    #
    #   @param         self
    #   @param         docid
    #   @param         pos
    #   @return        None
    #   @exception     None
    ## 
    def add(self, docid, pos):
        key = self.__posting.keys() #list of all keys
        if docid not in key:
            self.__posting[docid] = Posting(docid)
        self.__posting[docid].append(pos)
        #Removed old code, as python 3 does not have has_key.
        # if not self.posting.has_key(docid):
        #     self.posting[docid] = Posting(docid)
        # self.posting[docid].append(pos)

    ##
    #   @brief         This method sort the posting list by document ID for more efficient merging. 
    #                  And also sort each posting positions
    # 
    #
    #   @param         self
    #   @return        sorted posting list and sorted dict posting list
    #   @exception     None
    ## 
    def return_sorted_posting(self):
        for _, postingTemp in self.__posting.items():
            postingTemp.sort()

        self.__sorted_postings    = sorted(self.__posting.items(), key=operator.itemgetter(0))
        self.__sorted_dict        = collections.OrderedDict(self.__sorted_postings)
        return self.__sorted_postings , self.__sorted_dict

    ##
    #   @brief         This method sort the posting list by document ID for more efficient merging. 
    #                  And also sort each posting positions.
    #     sort by document ID for more efficient merging. For each document also sort the positions
    #     Firt sort all posting positions
    #     then sort doc id 
    #     also creat new sorted dict. // not sure if need but why not
    # 
    #
    #   @param         self
    #   @return        Noe
    #   @exception     None
    ## 
    def sort(self):
        for _, postingTemp in self.__posting.items():
            postingTemp.sort()

        self.__sorted_postings    = sorted(self.__posting.items(), key=operator.itemgetter(0))
        self.__posting            = collections.OrderedDict(self.__sorted_postings)
    
    ##
    #   @brief         This Method transforms the postings data into a dictionary format to be converted to Json
    #   @param         self
    #   @return        posting: dict
    #   @exception     None
    ## 
    def posting_list_to_string(self):
        docID       = int
        positions   = []
        listOfShit  = {}
        posting  = {}
        numberOfTimeTermIsInDoc = 0
        for docID, post in  self.__posting.items():
            docID, positions = post.get_info()
            listOfShit[docID] = positions
            numberOfTimeTermIsInDoc += 1

        posting["df"]       = numberOfTimeTermIsInDoc 
        posting["posting"]  = listOfShit
        return posting
##
# @brief     
#
# @bug       None documented yet   
#
class Index:
    ##
    #    @param         self
    #    @param         topicName
    #    @return        None
    #    @brief         The constructor. 
    #    @exception     None documented yet
    ##
    def __init__(self,index_type=1):
        # 0 is forward 1 in inverted
        self.__items     = {} # list of IndexItems
        self.__tokenizer = Tokenizer()
        self.__index_type = index_type
        self.__docIDs = set()
        self.__terms = set()
    ##
    #   @brief     This method return the total number of doc in our data set
    #
    #   @param         self
    #   @param         Doc
    #   @return        int
    #   @exception     None
    ## 
    def get_total_number_Doc(self):
        return len(self.__docIDs)

    ##
    #   @brief     This method return the total number of doc in our data set
    #
    #   @param         self
    #   @param         Doc
    #   @return        int
    #   @exception     None
    ## 
    def get_total_number_terms(self):
        return len(self.__terms)

    ##
    #   @brief     This method returns a sorted list of docIDS
    #
    #   @param         self
    #   @param         Doc
    #   @return        int
    #   @exception     None
    ## 
    def get_docIDs(self):
        return sorted(self.__docIDs)
    
    ##
    #   @brief     This method return a sorted list of terms 
    #
    #   @param         self
    #   @param         Doc
    #   @return        int
    #   @exception     None
    ## 
    def get_terms(self):
        return sorted(self.__terms)

    ##
    #   @brief     This method return the total number of doc in our data set
    #
    #   @param         self
    #   @param         Doc
    #   @return        items: dict
    #   @exception     None
    ## 
    def get_items(self):
        return self.__items

    ##
    #   @brief     This method is designed to index a docuemnt, using the simple SPIMI algorithm, 
    #              but no need to store blocks due to the small collection we are handling. 
    #              Using save/load the whole index instead
    # 
    #       ToDo: indexing only title and body; use some functions defined in util.py
    #       (1) convert to lower cases,
    #       (2) remove stopwords,
    #       (3) stemming
    #
    #   @param         self
    #   @param         Doc
    #   @return        None
    #   @exception     None
    ## 
    def index_doc(self, doc): # indexing a Document object
        newDoc              = doc['indexed_content']
        docID               = doc['docID']
        full_stemmed_list   = self.__tokenizer.transpose_document_tokenized_stemmed(newDoc)
        self.__docIDs.add(docID)

        for position, term in enumerate(full_stemmed_list):
            #may be more efficient to split into a forward and backward method for larger datasets
            # 0: forward index
            # 1: inverted_index
            index_key,index_value = {   
                0:(docID,term),
                1:(term,docID)
            }[self.__index_type]

            if term not in self.__terms:
                self.__terms.add(term)

            if self.__items.get(index_key) != None:
                self.__items[index_key].add(index_value, position)
            else:
                #key does not exists in dict
                newPosting                          = Posting(index_value)
                newPosting.append(position)
                self.__items[index_key]                  = IndexItem(index_key)
                self.__items[index_key].set_posting_list(index_value, newPosting)


    def single_doc_file(self,filepath): # indexing a file

        ## NOT DONE NEED TO INCLUDE PARSING FOR THE SUBJECT WHICH SHOULD BE EASY. REMEMBER TO CHECK IF SUBJECT EXISTS SO THAT WE CAN HANDLE EXCEPTIONS
        with open(filepath,'rb') as f:
            lines = [str.strip(line.decode("utf-8", "ignore")) for line in f]
        num_lines=-1
        for line in lines:
            if re.match("Lines:[\\s]*[0-9]+$" , line):
                num_lines = int(str.split(line)[1])
        doc =   {   
                    'indexed_content': ' '.join(lines[-num_lines:]),
                    'docID':filepath
                }
        self.index_doc(doc)


    def indexDir(self,dirpath, file_format=0, recursive = True): # indexing files in a directory
        
        file_format_map =   {
                                0:self.single_doc_file
                            }
        
        if file_format not in file_format_map:
            raise ValueError('fileformat not supported')

        file_processor = file_format_map[file_format]

        if recursive==False:
            for filename in os.listdir(dirpath):
                if os.path.isfile(os.path.join(dirpath,filename)):
                    file_processor(os.path.join(dirpath,filename))
        else:
            for directory in os.walk(dirpath):
                for filename in directory[2]: 
                    file_processor(os.path.join(directory[0],filename))

    ##
    #   @brief     This method Sorts all posting list by document ID. 
    #              NOTE: This method seems redundant as by default all postings list document IDs will be in order. 
    #                    Since documents are read in in a particular order. 
    #   @param         self
    #   @return        None
    #   @exception     None
    ## 
    def sort(self):
        ''' sort all posting lists'''
        for _, posting in self.__items.items():
            posting.sort()

    ##
    #   @brief     This method sorts all indexing primary key values in our index 
    #
    #   @param         self
    #   @return        OrderedDict
    #   @exception     None
    ## 
    def sort_postings(self):
        ''' sort IndexItems according to associated index key'''
        return collections.OrderedDict(sorted(self.__items.items(), key=operator.itemgetter(0)))
  
    # ##
    # #   @brief     This method to dumper for json
    # #
    # #   @param         self
    # #   @param         obj
    # #   @return        toJSON or dict
    # #   @exception     None
    # ## 
    # def dumper(self, obj):
    #     try:
    #         return obj.toJSON()
    #     except:
    #         return obj.__dict__
    # 
    # ##
    # #   @brief     This method Serializes the inverted index to a json format and 
    # #              clears the Memory that holds this dictionary
    # #
    # #   @param         self
    # #   @param         filename
    # #   @return        ValueError
    # #   @exception     None
    # ## 
    # def save(self, filename):
    #     write_stream = open(filename, 'w')
    #     listTerm = self.sort_postings()
    #     dictMain = {}
    #     listInfo = {}

    #     for term, postingList in listTerm.items():
    #         dictTemp = postingList.posting_list_to_string()
    #         dictTemp["idf"] = self.idf(term)
    #         dictMain[term] = dictTemp

    #     listInfo["nDoc"] = self.get_total_number_Doc()
    #     listInfo["Data"] = dictMain
    #     try: 
    #         write_stream.write(json.dumps(listInfo, indent=3))
    #     except ValueError: 
    #          print ("Is not valid json")
    #     write_stream.close()

    # ##
    # #   @brief     This method deserializes a json file in a object by reallocating the self.__items
    # #
    # #   @param         self
    # #   @param         filename
    # #   @return        json: dict
    # #   @exception     ValueError
    # ## 
    # def load(self, filename):
    #     try: 
    #         with open(filename) as json_file:
    #             return json.load(json_file)
    #     except ValueError: 
    #         print ("Is not valid json") 


    ##
    #   @brief     This method get IDF for  term by compute the inverted document frequency for a given term.
    #               We used this IDF = (Total number of (documents))/(Number of  (documents) containing the word)
    #
    #   @param         self
    #   @param         term
    #   @return        idf:int
    #   @exception     None
    ## 
    def idf(self, term):
        ''' '''
        if term not in self.__terms:
            return 0

        N = self.get_total_number_Doc()

        if self.__index_type == 0:
            doc_frequency = 0
            for d in self.get_items():
                if term in d.get_posting_list():
                    doc_frequency+=1

        elif self.__index_type == 1:
            termData = self.__items[term]
            doc_frequency = len(termData.get_posting_list())
            
        #inverse document frequency 
        inverse_doc_frequency = round(math.log10(N/(float(doc_frequency))), 4)
        #probabilistic inverse document frequency from  
        #idf = round(math.log10(N - df /(float(df))), 4)
        return inverse_doc_frequency
        
    ##
    #   @brief     This method create IDF for doc
    #
    #   @param         self
    #   @return        idf: {term:idf, term2:idf}
    #   @exception     None
    ## 
    def all_idfs(self):
        if self.__index_type == 0:
            # technically no faster in big-O time complexity but is better in theta as there is no for terms in documents they dont occur in
            #import pdb; pdb.set_trace()
            idf_dictionary = collections.OrderedDict(zip(self.get_terms(),[0]*self.get_total_number_terms()))
            N = self.get_total_number_Doc()
            for doc in self.get_items():
                for term in self.get_items()[doc].get_posting_list():
                    idf_dictionary[term]+=1
            for term in idf_dictionary:
                idf_dictionary[term] = round(math.log10(N/(float(idf_dictionary[term]))), 4)

        elif self.__index_type == 1:
            idf_dictionary = collections.OrderedDict()
            for term in self.get_terms():
                idf_dictionary[term] = self.idf(term)

        return idf_dictionary 

    ##
    #   @brief     This method create IDF for doc
    #
    #   @param         self
    #   @return        idf: {doc1:{term:idf, term2:idf}, doc2:{term2:idf, term3:idf} }
    #   @exception     None
    ## 
    def idf_Dict(self):
        tf_dict = self.tf_Dict()
        all_idfs = self.all_idfs()

        for doc in tf_dict:
            for term in tf_dict[doc]:
                tf_dict[doc][term] = all_idfs[term]

        return tf_dict

    ##
    #   @brief     This method create TF for add doc
    #   There are different ways to represent TF we used tf = log(1+tf) 
    #   Another way is TF = (Frequency of the word in the sentence) / (Total number of words in the sentence)
    #   
    #   @param         self
    #   @return        word_tf_values: {term: tf, term2: tf2, ... }
    #   @exception     None
    ##   
    def doc_tf(self,docID):

        if self.__index_type == 0:
            if docID not in self.get_items():
                return {}
            posting = self.get_items()[docID].get_posting_list()
            return{term: round(math.log10(1 + posting[term].term_freq()), 4) for term in posting}

        elif self.__index_type == 1:
            items = self.get_items()
            return{term: round(math.log10(1 + items[term].get_posting_list()[docID].term_freq()), 4) for term in items if docID in items[term].get_posting_list()}

        return None

    ##
    #   @brief     This method create TF for add doc
    #   There are different ways to represent TF we used tf = log(1+tf) 
    #   Another way is TF = (Frequency of the word in the sentence) / (Total number of words in the sentence)
    #   
    #   @param         self
    #   @return        word_tf_values: {docID: tf, docID: tf}  
    #   @exception     None
    ##   
    def term_tf(self,term):

        if self.__index_type == 0:
            items = self.get_items()
            return{doc: round(math.log10(1 + items[doc].get_posting_list()[term].term_freq()), 4) for doc in items if term in items[doc].get_posting_list()}
        elif self.__index_type == 1:
            posting = self.get_items()[term].get_posting_list()
            return{doc : round(math.log10(1 + posting[doc].term_freq()), 4) for doc in posting}


        return None

    ##
    #   @brief     This method create IDF for doc
    #
    #   @param         self
    #   @return        idf: {term: {docID:idf}}
    #   @exception     None
    ## 
    def tf_Dict(self):

        if self.__index_type == 0:
            tf_dictionary = collections.OrderedDict()
            for docID in self.get_docIDs():
                tf_dictionary.update({docID:self.doc_tf(docID)})

        if self.__index_type == 1:
            tf_dictionary = collections.OrderedDict({docid:{} for docid in self.get_docIDs()})
            for term in self.get_terms():
                for docID,tf in self.term_tf(term).items():
                    tf_dictionary[docID].update({term:tf})

        return tf_dictionary

    ##
    #   @brief     This method create tfidf for all doc. 
    #              It structure is of the form {docID: {term: tf-idf,term: tf-idf }}    
    #   @param         self
    #   @param         self
    #   @param         idfDict    
    #   @return        TFIDF_dict:{docID: {term: tf-idf,term: tf-idf },docID2: {term: tf-idf,term: tf-idf }}  
    #   @exception     None
    ##  
    def tfidf_Dict(self):
        #import pdb;pdb.set_trace()
        tf_dict = self.tf_Dict()
        all_idfs = self.all_idfs()

        for doc in tf_dict:
            for term in tf_dict[doc]:
                tf_dict[doc][term] = tf_dict[doc][term] * all_idfs[term]

        return tf_dict

    ##
    #   @brief     This method create tfidf for all doc. 
    #              It structure is of the form {docID: {term: tf-idf,term: tf-idf }}    
    #   @param         self
    #   @param         self
    #   @param         idfDict    
    #   @return        TFIDF_dict:{docID: {term: tf-idf,term: tf-idf },docID2: {term: tf-idf,term: tf-idf }}  
    #   @exception     None
    ##  
    def tfidf(self,docs):
        tf_dict = {doc:self.doc_tf(doc) for doc in docs}
        terms = set.union(*[set(tf_dict[doc].keys()) for doc in tf_dict])
        doc_idfs = {term : self.idf(term) for term in terms}

        for doc in tf_dict:
            for term in tf_dict[doc]:
                tf_dict[doc][term] = tf_dict[doc][term] * doc_idfs[term]

        return tf_dict  

    
    ##
    
    ##
    #   @brief     This method Saves the current state of the InvertedIndex
    #
    #   @param         self
    #   @param         filename
    #   @return        None
    #   @exception     AttributeError,  pickle.PickleError
    ##          
    def storeData(self, filename):
        
        try: 
            fileP = open(filename, "wb") 
            pickle.dump(self, fileP) # serialize class object
        except (AttributeError, pickle.PickleError):
            print("Error pickle.dump InvertedIndex ")
        fileP.close()
    
    ##
    #   @brief     This method Loads the saved InvertedIndex
    #
    #   @param         self
    #   @param         filename
    #   @return        invertedIndexer
    #   @exception     (pickle.UnpicklingError, ImportError, EOFError, IndexError, TypeError)
    ##  
    def loadData(self, filename): 
        try:
            fileP = open(filename ,"rb")
            invertedIndexer = pickle.load(fileP)
        except (pickle.UnpicklingError, ImportError, EOFError, IndexError, TypeError) as err:
            print(err)
            print("Error pickle.load InvertedIndex ")
        fileP.close()
        return invertedIndexer


# def indexingCranfield():
#     #ToDo: indexing the Cranfield dataset and save the index to a file
#     # command line usage: "python index.py cran.all index_file"
#     # the index is saved to index_file

#     filePath = sys.argv[1]
#     fileName = sys.argv[2]

#     #filePath = "src/CranfieldDataset/cran.all"
#     #fileName = "src/Data/tempFile"
#     #filePath = "./CranfieldDataset/cran.all"
#     #fileName = "./Data/tempFile"
   
#     invertedIndexer = InvertedIndex()
#     #data = CranFile(filePath)
#     #for doc in data.docs:
#     #   invertedIndexer.indexDoc(doc)

#    #invertedIndexer.storeData(fileName)
#     print("Done")
   
#python index.py CranfieldDataset/cran.all Data/tempFile
if __name__ == '__main__':
    #test()
    pass


