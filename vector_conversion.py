import glob, os
import nltk
from nltk.stem import PorterStemmer
from nltk import word_tokenize,sent_tokenize
from scipy import spatial
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk.data
category_folders=os.listdir(r"C:\\Users\ashish\AppData\Roaming\nltk_data\corpora\final_dataset_each_cat_feat")#contains folders 01 to 22

path= 'C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_each_cat_feat\\'
path1='C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_feature_vector\\'
path2='C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_refined\\'

files_in_category=[]#to store all files present in final_dataset_each_cat_feat in each category.
files_in_category1=[]#to store all files that are there in final_dataset_refined

for category in range(len(category_folders)):
    files_in_category.append(os.listdir(path+str(category_folders[category])))
#files_in_category is a list of list that contains all the files divided in different list category

for category1 in range(len(category_folders)):
    files_in_category1.append(os.listdir(path2+str(category_folders[category1])))




#now the idea is to traverse all files in all categories and collect all the feature and create a final feature vector
feat_list=[]
f_list=[]
for cat_no in range(len(files_in_category)):#cat_no contains category number of list of files of each category
    #feature list in each category
    
    for file_no in range(len(files_in_category[cat_no])):#file_no contains file number for each category
        feature_list_each_file=[]
        cat=category_folders[cat_no]#cat contains the particular category
        file=files_in_category[cat_no][file_no]#file is the name of the file that is being accessed
        address=str(cat)+"/"+str(file)
        dat=str(nltk.data.load('corpora/final_dataset_each_cat_feat/'+str(address),format='raw')).lower().split(',')#dat contains data of the current text files
        for data_length in range(len(dat)-1):
            feature_list_each_file.append(dat[data_length])
        
                
    f_list.append(feature_list_each_file)

    
for i in range(len(f_list)):
    for j in range(len(f_list[i])):
        feat_list.append(f_list[i][j])
feat_list=list(set(feat_list))
feature_list=[] 
for every_item in feat_list:
    if 'b\\\'' in every_item:
        feature_list.append(every_item[3:])
    else:
        feature_list.append(every_item)
feature_list=list(set(feature_list))
featu_list=[]
for item in feature_list:
    if 'b\'' in item:
        featu_list.append(item[2:])
    else:
        featu_list.append(item)
featu_list=list(set(featu_list))# consists of 16832 items, which are unique features from all categories.



#now trying to make feature vector for each file in all categories

        
for cat_no in range(len(files_in_category1)):
    for file_no in range(len(files_in_category1[cat_no])):
        feature_vector_file=[]   #it will contain vector representation of features of a file in this list
        feature_list_each_file=[]  #contains all the features for a particular file.
        feature_list_each_file2=[]
        feature_list_each_file1=[]
        feature_list_each_file3=[]
        cat=category_folders[cat_no]
        file=files_in_category1[cat_no][file_no]
        
        address=str(cat)+"/"+str(file)
        dat=str(nltk.data.load('corpora/final_dataset_refined/'+str(address),format='raw')).lower().split(',')
        for data_length in range(len(dat)-1):
            feature_list_each_file1.append(dat[data_length])
        for entry1 in feature_list_each_file1:
            if 'b\'' in entry1:
                feature_list_each_file2.append(entry1[2:])
            else:
                feature_list_each_file2.append(entry1)

        for entry2 in feature_list_each_file2:
            if 'b\"' in entry2:
                feature_list_each_file3.append(entry2[2:])
            else:
                feature_list_each_file3.append(entry2)

        for entry in feature_list_each_file3:
            if 'b\\\'' in entry:
                feature_list_each_file.append(entry[3:])
            else:
                feature_list_each_file.append(entry)
        for recipient in featu_list:
            if recipient in feature_list_each_file:
                feature_vector_file.append(1)
            else:
                feature_vector_file.append(0)
        
                
        c_number=category_folders[cat_no]
        filewrite = open(str(path1)+str(c_number)+"\\"+str(file),'w')
        
        for all_item in feature_vector_file:
            filewrite.write("%s," % all_item)
        filewrite.close()
    print("Done for another category:")
        
        
        


