# stl modules
import os
import json
import sys
import re
# external modules

# project modules
import index

reverse = True

def feature_extract():
    directory_of_newsgroups_data = sys.argv[1]
    class_definition_filename = sys.argv[3]
    feature_definition_filename = sys.argv[2]
    training_data_filename = sys.argv[4]

    groups_to_class = [['comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x'],
    ['rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'],
    ['sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'],
    ['misc.forsale'],
    ['talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast'],
    ['talk.religion.misc', 'alt.atheism', 'soc.religion.christian']]

    class_definition = {j:inx+1 for inx, i  in enumerate(groups_to_class) for j in i}

    my_index = index.Index(index_type=int(reverse))
    my_index.indexDir(directory_of_newsgroups_data) 

    terms = my_index.get_terms()
    feature_definition = dict([(i,inx+1)for inx,i in enumerate(terms)])

    tf = my_index.tf_Dict()
    idf = my_index.idf_Dict()
    tfidf = my_index.tfidf_Dict()

    # my_index2 = index.Index(index_type=int(not reverse))
    # my_index2.indexDir(directory_of_newsgroups_data)
    # terms2 = my_index2.get_terms()
    # tf2 = my_index2.tf_Dict()
    # idf2 = my_index2.idf_Dict()
    # tfidf2 = my_index2.tfidf_Dict()

    # write out tf features
    with open(training_data_filename+'.tf','w') as outfiletf:
        for docid,data in tf.items():
            directory = re.search('/(.*)/', docid)
            category = class_definition[directory.group(1)]
            outstring = [str(category)] + [str(feature_definition[term])+':'+str(tfvalue) for term,tfvalue in data.items()]
            outfiletf.write(' '.join(outstring)+'\n')

    # write out idf features
    with open(training_data_filename+'.idf','w') as outfileidf:
        for docid,data in idf.items():
            directory = re.search('/(.*)/', docid)
            category = class_definition[directory.group(1)]
            outstring = [str(category)] + [str(feature_definition[term])+':'+str(idfvalue) for term,idfvalue in data.items()]
            outfileidf.write(' '.join(outstring)+'\n')

    # write out tfidf feature
    with open(training_data_filename+'.tfidf','w') as outfiletfidf:
        for docid,data in tfidf.items():
            directory = re.search('/(.*)/', docid)
            category = class_definition[directory.group(1)]
            outstring = [str(category)] + [str(feature_definition[term])+':'+str(tfidfvalue) for term,tfidfvalue in data.items()]
            outfiletfidf.write(' '.join(outstring)+'\n')

    with open(feature_definition_filename, 'w') as feature_file:
        json.dump(feature_definition, feature_file)

    with open(class_definition_filename, 'w') as class_file:
        json.dump(class_definition, class_file)

def test():
    return


if __name__ == '__main__':
    test()
    feature_extract()
