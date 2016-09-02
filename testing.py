import glob, os
import nltk
from nltk.stem import PorterStemmer
from nltk import word_tokenize,sent_tokenize
from scipy import spatial
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk.data
category_folders=os.listdir(r"C:\\Users\ashish\AppData\Roaming\nltk_data\corpora\final_dataset")#contains folders 01 to 23
path= 'C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset\\'
#print(os.listdir(path+str(category_folders[0])))#it prints all the file that is contained in that folder
#print(len(os.listdir(path+str(category_folders[0]))))#it prints the count of all the files in that path
files_in_category=[]

for category in range(len(category_folders)):
    files_in_category.append(os.listdir(path+str(category_folders[category])))
#files_in_category is a list of list that contains all the files divided in different list category
    
textdata=files_in_category[3][1879]
#print(textdata)
text=str(nltk.data.load('corpora/final_dataset/C04/'+str(textdata),format='raw')).lower()
#print(text)
tokenizer=list(sent_tokenize(text))
#print(tokenizer[0])
#print(len(tokenizer))


features=[]
for i in range(len(tokenizer)):
    words=nltk.word_tokenize(tokenizer[i])
    tagged=nltk.pos_tag(words)
    chunkgram=r"""Chunk: {<JJ.?>*<NN.?>*<NNPS>*<NN.?>}"""
    chunkparser=nltk.RegexpParser(chunkgram)
    chunked=(chunkparser.parse(tagged))
    for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
        features.append(list(subtree))
    #chunked.draw()
    #print(tagged)

##for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
##    features.append(list(subtree))
#print(features)
#here features have list of list
#trying to get a list of noun phrase features
featuresetnounphrase=[]
for featureset in features:#here featureset is an individual list
    stri=''
    for members in featureset:
        if '.\\n' in members[0]:
            stri=stri+str(members[0].replace('.\\n',''))
        elif '.' in members[0]:
            stri=stri+str(members[0].replace('.',''))
        else:
             stri=stri+str(members[0])
    featuresetnounphrase.append(stri)
featuresetnounphrase=list(set(featuresetnounphrase))
#print(len(featuresetnounphrase))
            
            
    
tokenizer1=RegexpTokenizer(r'\w+')
#single_letter=RegexpTokenizer(r'[a-z A-Z]')
tokenized_data=tokenizer1.tokenize(text)
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
    if filtered_words[word_count][1]!='CD':
        refiltered_words.append(ps.stem(filtered_words[word_count][0]))
refiltered_words=list(set(refiltered_words))

    
for nounphrases in featuresetnounphrase:
    refiltered_words.append(nounphrases)
refiltered_words=list(set(refiltered_words))
#print(refiltered_words)#refilteres_words contains all the features for a particular file.

    
