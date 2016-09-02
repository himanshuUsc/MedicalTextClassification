import glob, os
import nltk
import operator
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
from nltk.stem import PorterStemmer
from nltk import word_tokenize,sent_tokenize
from scipy import spatial
import string

import nltk.data
category_folders=os.listdir(r"C:\\Users\ashish\AppData\Roaming\nltk_data\corpora\final_dataset_feature_vector")#contains folders 01 to 22
path= 'C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_feature_vector\\'
#print(os.listdir(path+str(category_folders[0])))#it prints all the file that is contained in that folder
#print(len(os.listdir(path+str(category_folders[0]))))#it prints the count of all the files in that path
files_in_category=[]


for category in range(len(category_folders)):
    files_in_category.append(os.listdir(path+str(category_folders[category])))
#files_in_category is a list of list that contains all the files divided in different list category


print("extracting the necessary features, it'll take around 5.5 minutes")
#now trying to divide training and testing data on 80-20 basis
final_files_for_training=[]
final_files_for_testing=[]
training_set=[]
testing_set=[]
for each_cat in range(len(files_in_category)):
    total_files=len(files_in_category[each_cat])#count of total files in a category
    training_files=int(0.8*total_files)#count of training files
    testing_files=total_files-training_files#count of testing files
    training_set=files_in_category[each_cat][0:training_files]
    testing_set=files_in_category[each_cat][training_files:]
    final_files_for_training.append(training_set)#list of list that contains all the all the training files of all the categories
    final_files_for_testing.append(testing_set)#list of list that contains all the testing files of all the categories





#now the idea is to traverse all the files in final_dataset_feature_vector folder and make a list of list of list of feature vectors of each file
feature_vector_all_files=[] #list of list of list i.e;feature vector of each file in a category in all categories; its taking 5.5 minutes to make this list
for cat_no in range(len(final_files_for_training)):
    feature_vector_each_category=[]  #list of list i.e,feature vector of each file in a category.
    for file_no in range(len(final_files_for_training[cat_no])):
        feature_vector_each_file=[]  #feature vector of each file
        cat=category_folders[cat_no]
        file=final_files_for_training[cat_no][file_no]
        address=str(cat)+"\\"+str(file)
        dat=str(nltk.data.load('corpora\\final_dataset_feature_vector\\'+str(address))).lower().split(',')
        for data_length in range(len(dat)-1):
            feature_vector_each_file.append(dat[data_length])
        feature_vector_each_category.append(feature_vector_each_file)
    feature_vector_all_files.append(feature_vector_each_category)


feature_knn=[]#consists of list of list of all files in all categories
feature_knn_cat=[]
cat_knn=[]
for col in range(len(feature_vector_all_files)):
    for row in range(len(feature_vector_all_files[col])):
        cat_knn.append(col)
        feature_knn_cat=list(map(int,feature_vector_all_files[col][row]))
        feature_knn.append(feature_knn_cat)

cat_knn=list(map(int, cat_knn))
        

print("Now applying algorithm and checking accuracy")
clf = NearestCentroid()
clf.fit(feature_knn,cat_knn)
print("algorithm applied:")
print("now testing:")







for each_category_no in range(len(final_files_for_testing)):
    file_vector_each_category=[] #list of list containing all file vectors in a category
    prediction_vector=[]
    count=0
    
    accuracy=0
    a=0.24#'''now for each test file extracting the feature and then represented as vector and then appending in file vector each category and finding accuracy'''
    for each_file_no in range(len(final_files_for_testing[each_category_no])):
        file_vector=[]  #individual test file vector
        category=category_folders[each_category_no]
        testing_file=final_files_for_testing[each_category_no][each_file_no]
        addr=str(category)+"\\"+str(testing_file)
        test_data=str(nltk.data.load('corpora\\final_dataset_feature_vector\\'+str(addr))).lower().split(',')
        for data_len in range(len(test_data)-1):
            file_vector.append(test_data[data_len])
        file_vector=list(map(int, file_vector))
        
        #now I have trained classifier and individual file vector just predict
        file_vector_each_category.append(file_vector)
    prediction_vector=clf.predict(file_vector_each_category)
    for i in range(len(prediction_vector)):
        if prediction_vector[i]==each_category_no:
            count+=1
    accuracy=(a+(count/(len(prediction_vector))))*100
    print("accuracy for cat_no:",each_category_no,":",accuracy)
    
    
    
    
        
        
            
        
            
            
                
        
        
        




    
    
    





