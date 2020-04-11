# stl modules
import os
import json
import sys
import re
# external modules

# project modules
import index

def main():
    # reads command line arguments
    directory_of_newsgroups_data = sys.argv[1]
    class_definition_filename = sys.argv[3]
    feature_definition_filename = sys.argv[2]
    training_data_filename = sys.argv[4]

    # maps filenames to classes
    groups_to_class = [['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x'],
    ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
    ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
    ['misc.forsale'],
    ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
    ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']]

    # create class definition
    class_definition = {j:inx+1 for inx, i  in enumerate(groups_to_class) for j in i}

    # create index object, can do forward or reverse though 
    # we used reverse in this case as all of the functionality for this project was tested in the previous project
    # forward and reverse give the same results though feature orders and other cosmetic things are different
    my_index = index.Index(index_type=1)
    my_index.indexDir(directory_of_newsgroups_data) 

    # create mappings between featuresids and the terms
    terms = my_index.get_terms()
    feature_definition = dict([(inx+1,i)for inx,i in enumerate(terms)])
    inverted_feature_definition = dict((v, k) for k, v in feature_definition.items())

    # calls index functions for calculating tf, idf, and tfidf for all documents
    # these were extended from preexisting code that was tested in the last project for reverse indexes
    tf = my_index.tf_Dict()
    idf = my_index.idf_Dict()
    tfidf = my_index.tfidf_Dict()

    output_data_and_names = [(tf,'TF'),(idf,'IDF'),(tfidf,'TFIDF')]

    #write out features
    for dataset in output_data_and_names:
        with open(training_data_filename+'.'+dataset[1],'w') as outfile:
            for docid,data in dataset[0].items():
                
                ## have to check for platform since windows uses \ and linux uses / it should work for either
                if sys.platform == "win32" or sys.platform == "win64" or sys.platform == "windows" :
                    stop = docid[::-1].find("\\")
                    start = docid[::-1].find('\\', docid[::-1].find('\\') + 1)
                    directory = docid[-start:-stop-1]

                else:
                    stop = docid[::-1].find('/')
                    start = docid[::-1].find('/', docid[::-1].find('/') + 1)
                    directory = docid[-start:-stop-1]
                   

                category = class_definition[directory]

                outstring = [str(category)] + [str(inverted_feature_definition[term])+':'+str(termvalue) for term,termvalue in data.items()]
                outfile.write(' '.join(outstring)+'\n')
    
    # write out feature definition file in expected format, though json writeout is also supported in commented code
    with open(feature_definition_filename, 'w') as feature_file:
        for k,v in feature_definition.items():
            feature_file.write(str(k)+', '+str(v)+'\n')
        # can do it with json too 
        #json.dump(feature_definition, feature_file)

    # write out feature definition file in expected format, though json writeout is also supported in commented code
    with open(class_definition_filename, 'w') as class_file:
        for k,v in class_definition.items():
            class_file.write(str(k)+', '+str(v)+'\n')
        # can do it with json too if that is easier
        #json.dump(class_definition, class_file)



if __name__ == '__main__':
    print("Starting Feature Extraction")
    main()
    print("Finished Feature Extraction")
    
'''
equivalancy between feature_definition and class_definition is trivial using diff

equivalency between training_data files is done with the following code in a python interpreter
it effectively reads in each line (document) and checks that the same features are present and that they have the same value
it does this by creating dictionaries of each document with term:value and then compares files created with a forward and reverse index

file1 = 'training_data_file.IDF'
file2 = 'training_data_file.idf'

with open(file1) as f1:
    lines1 = [line for line in f1]
with open(file2) as f2:
    lines2 = [line for line in f2]

if not (len(lines1) == len(lines2)):
    print('Failure files are not the same length')

for linx,l1 in enumerate(lines1):
    l2 = lines2[linx]

    d1 = dict([tuple(str.split(t,':')) for t in str.split(l2)[1:]])
    d2 = dict([tuple(str.split(t,':')) for t in str.split(l2)[1:]])
    if not (all([d1[k]==d2[k] for k in d1]) and (str.split(l1)[0] == str.split(l2)[0])):
        print('Failure: '+str(linx)+' do not match')

'''