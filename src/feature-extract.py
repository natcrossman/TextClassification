# stl modules
import os
import json
import sys
import re
# external modules

# project modules
import index

def main():
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

    my_index = index.Index(index_type=1)
    my_index.indexDir(directory_of_newsgroups_data) 

    terms = my_index.get_terms()
    feature_definition = dict([(i,inx+1)for inx,i in enumerate(terms)])

    tf = my_index.tf_Dict()
    idf = my_index.idf_Dict()
    tfidf = my_index.tfidf_Dict()

    output_data_and_names = [(tf,'tf'),(idf,'idf'),(tfidf,'tfidf')]

    #write out features and stuff
    for dataset in output_data_and_names:
        with open(training_data_filename+'.'+dataset[1],'w') as outfile:
            for docid,data in dataset[0].items():
                directory = re.search('/(.*)/', docid)
                category = class_definition[directory.group(1)]
                outstring = [str(category)] + [str(feature_definition[term])+':'+str(termvalue) for term,termvalue in data.items()]
                outfile.write(' '.join(outstring)+'\n')
    
    with open(feature_definition_filename, 'w') as feature_file:
        json.dump(feature_definition, feature_file)

    with open(class_definition_filename, 'w') as class_file:
        json.dump(class_definition, class_file)



if __name__ == '__main__':
    main()
    