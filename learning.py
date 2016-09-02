import glob, os
import nltk
from nltk.stem import PorterStemmer
from nltk import word_tokenize,sent_tokenize
from scipy import spatial
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk.data
category_folders=os.listdir(r"C:\\Users\ashish\AppData\Roaming\nltk_data\corpora\final_dataset")#contains folders 01 to 22

path= 'C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset\\'
path1='C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_refined\\'
#print(os.listdir(path+str(category_folders[0])))#it prints all the file that is contained in that folder
#print(len(os.listdir(path+str(category_folders[0]))))#it prints the count of all the files in that path
files_in_category=[]

for category in range(len(category_folders)):
    files_in_category.append(os.listdir(path+str(category_folders[category])))
#files_in_category is a list of list that contains all the files divided in different list category

print("feature extraction begins now:")
#now the idea is to traverse all files in all categories and do feature extraction(training)
feature_list=[]
f_list=[]#list of list of list to store feature vector of each file in a category, in all categories
for cat_no in range(len(files_in_category)):#cat_no contains category number of list of files of each category
    feature_list_each_category=[] #feature list in each category
    
    for file_no in range(len(files_in_category[cat_no])):#file_no contains file number for each category
        cat=category_folders[cat_no]#cat contains the particular category
        file=files_in_category[cat_no][file_no]#file is the name of the file that is being accessed
        address=str(cat)+"/"+str(file)
        dat=str(nltk.data.load('corpora/final_dataset/'+str(address),format='raw')).lower()#dat contains data of the current text files
        tokenizer=list(sent_tokenize(dat))#all the data being tokenized into sentences and made alist of that
        features=[]  #for all sent tokenizer it will have data as list of list containing noun phrases 
        for i in range(len(tokenizer)):
            words=nltk.word_tokenize(tokenizer[i])#for each sentence all words have been tokenized
            tagged=nltk.pos_tag(words)#pos tags given
            chunkgram=r"""Chunk: {<JJ.?>*<NN.?>*<NNPS>*<NN.?>}"""#now extracting only noun phrases i.e,extracting medical terms
            chunkparser=nltk.RegexpParser(chunkgram)
            chunked=chunkparser.parse(tagged)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                features.append(list(subtree))
        featuresetnounphrase=[] #it contains noun phrases features in appropriate format
        for featureset in features:#here featureset is an individual list
            stri=''
            for members in featureset:
                if '.\\n' in members[0]:
                    stri=stri+str(members[0].replace('.\\n',''))#to remove .\\n characters from features that was automatically coming with chunking
                elif '.' in members[0]:
                    stri=stri+str(members[0].replace('.',''))#to remove . character from the feature that was automatically coming with chunking
                else:
                     stri=stri+str(members[0])
            featuresetnounphrase.append(stri)
        featuresetnounphrase=list(set(featuresetnounphrase))
        #print(len(featuresetnounphrase))
        #now stemming all the words and having them as features    
        tokenizer1=RegexpTokenizer(r'\w+')
        tokenized_data=tokenizer1.tokenize(dat)
        #print(tokenized_data)
        single_letter_data=list(string.ascii_lowercase)
        stop_words = set(stopwords.words('english'))
        filtered_words = [w for w in tokenized_data if not w in stop_words]
        filtered_words = [w for w in filtered_words if not w in single_letter_data]
        filtered_words=nltk.pos_tag(filtered_words)
        #print(filtered_words)

        ps=PorterStemmer()
        refiltered_words=[]#stemming performed
        for word_count in range(len(filtered_words)):
            if filtered_words[word_count][1]!='CD':   #not taking numbers as 'CD' represents numbers,NOTE: word_count[1]represents pos tag
                refiltered_words.append(ps.stem(filtered_words[word_count][0]))
        refiltered_words=list(set(refiltered_words))

    
        for nounphrases in featuresetnounphrase:
            refiltered_words.append(nounphrases)
        refiltered_words=list(set(refiltered_words))#refilteres_words contains all the features for a particular file.
        c_number=category_folders[cat_no]
        filewrite = open(str(path1)+str(c_number)+"\\"+str(file),'w')
        for all_items in refiltered_words:
            filewrite.write("%s," % all_items)
        filewrite.close()
    print("done for another category:")


