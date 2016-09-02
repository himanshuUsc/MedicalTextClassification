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
path1= 'C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_for_svm\\'
files_in_category=[]
files_in_category1=[]
for category in range(len(category_folders)):
    files_in_category.append(os.listdir(path+str(category_folders[category])))
for category1 in range(len(category_folders)):
    files_in_category1.append(os.listdir(path1+str(category_folders[category1])))
add=0
for ca in range(len(files_in_category)):
    add=add+len(files_in_category[ca])#add stores the count of all the files in all the categories





file_count_each_category=[] #represented as a list of list where each list contains starting file no and ending file no. of each category in order
start=0
for cat_no in range(len(files_in_category)):
    end=start+(len(files_in_category[cat_no])-1)
    file_count_each_category.append([start,end])
    start=end+1


#now applying SVM for each category:

for each_cat in range(len(files_in_category1)):
    training_vector=[0]*add  #here training vector is initialized with zero values
    accuracy=0
    for file_no in range(len(files_in_category1[each_cat])):
        file_content=[]
        file_content_final=[]
        file_content_finally=[]#to store the content of file in a category from final_dataset_for_svm folder
        start_address=0
        end_address=0
        cat=category_folders[each_cat]
        file_present=files_in_category1[each_cat][file_no]
        address=str(cat)+"/"+str(file_present)
        dat=str(nltk.data.load('corpora/final_dataset_for_svm/'+str(address))).split(',')
        for data_length in range(len(dat)-1):
            file_content.append(dat[data_length])
        for items in file_content:
            if '[' in items:
                file_content_final.append(int(items.replace('[','')))
            elif ']'in items:
                file_content_final.append(int(items.replace(']','').strip()))
                
            else:
                file_content_final.append(int(items))
        for count in range(0,(len(file_content_final)-1),2):
            file_content_finally.append([file_content_final[count],file_content_final[count+1]])
        file_count=file_count_each_category[each_cat]#it is a list which contains starting file no and ending file no. of the category(each_cat)
        
        start_address=file_count[0]
        end_address=file_count[1]
        training_vector[start_address:(end_address+1)]=[1]*((end_address+1)-start_address)
        #now applying svm algorithm
        #splitting training and testing files
        #train_count=int(0.8*add)
        if (end_address+(end_address-start_address))<(file_count_each_category[len(file_count_each_category)-1][1]):
            ending=end_address+(end_address-start_address)#till here training
            ran=(ending+1-(start_address))
            end=int(ending+(0.2*ran))
            #start=int(start_address-(0.2*ran))
            train_data_mat=file_content_finally[start_address:ending]
            test_data_mat=file_content_finally[ending:end]
            train_vector=training_vector[start_address:ending]
            test_vector=training_vector[ending:end]
        else:
            ending=end_address-(2*(end_address-start_address))
            ran=start_address+1-(ending)
            end=int(ending-(0.2*ran))
            train_data_mat=file_content_finally[ending:end_address]
            test_data_mat=file_content_finally[end:ending]
            train_vector=training_vector[ending:end_address]
            test_vector=training_vector[end:ending]
        
        
        training_matrix=np.array(train_data_mat)
        clf = svm.SVC(kernel='linear', C = 1.0)
        clf.fit(training_matrix,train_vector)
        total_match=0
        
        for data in range(len(test_data_mat)):
            prediction = clf.predict(test_data_mat[data])
            
            if prediction==test_vector[data]:
                total_match+=1
        
        accuracy = (float(total_match)/len(test_data_mat))
        print("Accuracy for category "+str(each_cat+1)+" = "+"%.2f" % (accuracy*100) + "%")

        w = clf.coef_[0]
        #print(w)

        a = -w[0] / w[1]

        xx = np.linspace(0,12)
        yy = a * xx - clf.intercept_[0] / w[1]

        h0 = plt.plot(xx, yy, 'k-', label="non weighted div")

        plt.scatter(training_matrix[:, 0], training_matrix[:, 1], c = train_vector)
        plt.legend()
        plt.show()        
        
        
        
                
                
    
        
    
    
    
        
    
