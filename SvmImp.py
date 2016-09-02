'''here trying to implement Svm algorithm,
for this, implementing a matrix which will be stored for each category in corpora which conrtains a list for each file in all categories which will
have two entries 1st being number of features matching for that category and other being not matching and other which will have answers i.e;0 for not having
and for having.
'''
import glob, os
import nltk
import operator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from scipy import spatial
import string

import nltk.data
category_folders=os.listdir(r"C:\\Users\ashish\AppData\Roaming\nltk_data\corpora\final_dataset_refined")#contains folders 01 to 22
path= 'C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_refined\\'
path1= 'C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_each_cat_feat\\'
path2= 'C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_for_svm\\'

#print(os.listdir(path+str(category_folders[0])))#it prints all the file that is contained in that folder
#print(len(os.listdir(path+str(category_folders[0]))))#it prints the count of all the files in that path
files_in_category=[]
files_in_category1=[]


for category in range(len(category_folders)):
    files_in_category.append(os.listdir(path+str(category_folders[category])))
#files_in_category is a list of list that contains all the files divided in different list category

for category1 in range(len(category_folders)):
    files_in_category1.append(os.listdir(path1+str(category_folders[category1])))




#now the idea is to traverse all the files in final_dataset_feature_refined folder and make a list of matching and non matching feature for each category
#now traversing all files in refined folder and make a list of list of list of features of each file.
all_files_all_category=[]  #all files in all categories.
for cat_no in range(len(files_in_category)):
    all_files_category=[]
    for file_no in range(len(files_in_category[cat_no])):
        all_files1=[]
        all_files2=[]
        all_files=[]
        cat=category_folders[cat_no]
        file=files_in_category[cat_no][file_no]
        address=str(cat)+"\\"+str(file)
        dat=str(nltk.data.load('corpora\\final_dataset_refined\\'+str(address))).lower().split(',')
        for data_length in range(len(dat)-1):
            all_files1.append(dat[data_length])

        for every_item1 in all_files1:
            if 'b\\\'' in every_item1:
                all_files2.append(every_item1[3:])
            else:
                all_files2.append(every_item1)
        all_files2=list(set(all_files2))
        all_files3=[]
        for item1 in all_files2:
            if 'b\'' in item1:
                all_files3.append(item1[2:])
            else:
                all_files3.append(item1)
        all_files3=list(set(all_files3))
        
        for item2 in all_files3:
            if 'b\"' in item2:
                all_files.append(item2[2:])
            else:
                all_files.append(item2)
        all_files=list(set(all_files))
        
        
        all_files_category.append(all_files)
    all_files_all_category.append(all_files_category)
#print(all_files_all_category[0][0][:25])
final_list=[] #final list of all files
for leng in range(len(all_files_all_category)):
    for bred in range(len(all_files_all_category[leng])):
        final_list.append(all_files_all_category[leng][bred])
#print(len(final_list)) #23440 files all together in all categories
        


for each_cat in range(len(files_in_category1)):
    svm_list=[]#for each category we have this list containing list of all file in appropriate format
    for present_file in range(len(files_in_category1[each_cat])):
        file_content=[]  
        cat1=category_folders[each_cat]
        file1=files_in_category1[each_cat][present_file]
        address1=str(cat1)+"\\"+str(file1)
        dat1=str(nltk.data.load('corpora\\final_dataset_each_cat_feat\\'+str(address1))).lower().split(',')
        for data_length1 in range(len(dat1)-1):
            file_content.append(dat1[data_length1])
        file_content1=[]
        for every_item in file_content:
            if 'b\\\'' in every_item:
                file_content1.append(every_item[3:])
            else:
                file_content1.append(every_item)
        file_content1=list(set(file_content1))
        file_content2=[]
        for item in file_content1:
            if 'b\'' in item:
                file_content2.append(item[2:])
            else:
                file_content2.append(item)
        file_content2=list(set(file_content2))
        file_content3=[]  #list having content of all the unique features in a category
        for item11 in file_content2:
            if 'b\"' in item11:
                file_content3.append(item11[2:])
            else:
                file_content3.append(item11)
        file_content3=list(set(file_content3))
    file_comp=[]   
    for comp in range(len(final_list)):
        
        file_comp=final_list[comp]
        cont=0
        for pre in file_content3:
            if pre in file_comp:
                cont+=1
        svm_list.append([cont,(len(file_comp)-cont)])#check here once 2nd argument modification required or not.

    c_number=category_folders[each_cat]
    filewrite = open(str(path2)+str(c_number)+"\\"+str(c_number)+".txt",'w')
    for all_item in svm_list:
        filewrite.write("%s," % all_item)
    filewrite.close()
    print("Done for another category:")
    
        
    

        





