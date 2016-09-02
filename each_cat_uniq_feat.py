import glob, os
import nltk
from nltk.stem import PorterStemmer
from nltk import word_tokenize,sent_tokenize
from scipy import spatial
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk.data
category_folders=os.listdir(r"C:\\Users\ashish\AppData\Roaming\nltk_data\corpora\final_dataset_refined")#contains folders 01 to 22

path= 'C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_refined\\'
path1='C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_each_cat_feat\\'
#print(os.listdir(path+str(category_folders[0])))#it prints all the file that is contained in that folder
#print(len(os.listdir(path+str(category_folders[0]))))#it prints the count of all the files in that path
files_in_category=[]

for category in range(len(category_folders)):
    files_in_category.append(os.listdir(path+str(category_folders[category])))
#files_in_category is a list of list that contains all the files divided in different list category
final_files_for_training=[]
for each_cat in range(len(files_in_category)):
    total_files=len(files_in_category[each_cat])#count of total files in a category
    training_files=int(0.8*total_files)#count of training files
    
    training_set=files_in_category[each_cat][0:training_files]
    
    final_files_for_training.append(training_set)#list of list that contains all the all the training files of all the categories
    

#idea is to write unique features of all the categories at the specified destination.

    
print("feature extraction begins now:")



for cat_no in range(len(final_files_for_training)):#cat_no contains category number of list of files of each category
    feature_list_each_category=[] #feature list in each category
    
    for file_no in range(len(final_files_for_training[cat_no])):#file_no contains file number for each category
        feature_list_each_file=[]
        cat=category_folders[cat_no]#cat contains the particular category
        file=final_files_for_training[cat_no][file_no]#file is the name of the file that is being accessed
        address=str(cat)+"/"+str(file)
        dat=str(nltk.data.load('corpora/final_dataset_refined/'+str(address),format='raw')).lower().split(',')#dat contains data of the current text files
        for data_length in range(len(dat)-1):
            feature_list_each_file.append(dat[data_length])
        
        feature_list_each_category.append(feature_list_each_file)        
    



    each_category_features_list=[]
    each_category_features_set=[]
    each_category_features=[]#it is a list which contains all the feaures found for a category having threshold of occurence>=3
    for fea in range(len(feature_list_each_category)):
        for feas in range(len(feature_list_each_category[fea])):
            each_category_features_list.append(feature_list_each_category[fea][feas])
            
    each_category_features_set=list(set(each_category_features_list))
    for each_item in each_category_features_set: #taking into consideration only those features which have a count greater than 2 in complete cat.
        if each_category_features_list.count(each_item)>2:
            each_category_features.append(each_item)
    print("feature extraction has been done for category number: ",cat_no+1)
    print("total features= ",len(each_category_features))

    


    c_number=category_folders[cat_no]   
    filewrite = open(str(path1)+str(c_number)+"\\"+str(c_number),'w')
    for all_items in each_category_features:
        filewrite.write("%s," % all_items)
    filewrite.close()
print("done for every category:")
