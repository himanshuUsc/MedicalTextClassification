import glob, os
import nltk
import operator
from nltk.stem import PorterStemmer
from nltk import word_tokenize,sent_tokenize
from scipy import spatial
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk.data
category_folders=os.listdir(r"C:\\Users\ashish\AppData\Roaming\nltk_data\corpora\final_dataset_refined")#contains folders 01 to 22
path= 'C:\\Users\\ashish\\AppData\\Roaming\\nltk_data\\corpora\\final_dataset_refined\\'
#print(os.listdir(path+str(category_folders[0])))#it prints all the file that is contained in that folder
#print(len(os.listdir(path+str(category_folders[0]))))#it prints the count of all the files in that path
files_in_category=[]


for category in range(len(category_folders)):
    files_in_category.append(os.listdir(path+str(category_folders[category])))
#files_in_category is a list of list that contains all the files divided in different list category


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
print(final_files_for_training[21])


print("feature extraction begins now:")
#now the idea is to traverse all files in all categories and do feature extraction(training)
feature_list=[]
f_list=[]#list of list of list to store feature vector of each file in a category, in all categories
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
    f_list.append(feature_list_each_category)



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

    feature_list.append(each_category_features)#list of list containing features from each category having threshold>=3
final_feature_vector=[]#all the features found from all files in all categories i.e.;around a lakh(82515)
for eachh in range(len(feature_list)):
    for eachhh in range(len(feature_list[eachh])):
        final_feature_vector.append(feature_list[eachh][eachhh])
final_feature_vector=list(set(final_feature_vector))
#print(len(final_feature_vector))

#note here f_list is a list of list of list to store feature vector of each file in a category, in all categories
#now trying to make a feature vector for each file in a category in terms of final_feature_vector
feature_vector_each_file_list=[]#to store feature vector of each file in all categories

for category_number in range(len(f_list)):
    f_vector_file_category=[]
    for file_number in range(len(f_list[category_number])):
        file=f_list[category_number][file_number]
        f_vector_file=[]
        for item in final_feature_vector:
            if item in file:
                f_vector_file.append(1)
            else:
                f_vector_file.append(0)
        f_vector_file_category.append(f_vector_file)
    feature_vector_each_file_list.append(f_vector_file_category)
print("Now applying algorithm and checking accuracy")

#now doing the accuracy check for k-nearest neighbour algorithm
#for this we have to do feature extraction for testing data file and then we'll try to find its k-nearest neighbour
for each_category_no in range(len(final_files_for_testing)):
    count=0
    for each_file_no in range(len(final_files_for_testing[each_category_no])):
        cosine_product_list=[]
        cosproduct_list=[]
        frequent_cat=[]
        
        
        cosproduct_list_cat=[]
        index_list_dec=[]
        each_file=[]
        freq_dict={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0,11:0,12:0,13:0,14:0,15:0,16:0,17:0,18:0,19:0,20:0,21:0}
        file_vector=[]
        testing_file=final_files_for_testing[each_category_no][each_file_no]
        category=category_folders[cat_no]
        addr=str(category)+"/"+str(testing_file)
        data=str(nltk.data.load('corpora/final_dataset_refined/'+str(addr),format='raw')).lower().split(',')
        for data_len in range(len(data)-1):
            each_file.append(data[data_len])
        
        for items in final_feature_vector:
            if items in each_file:
                file_vector.append(1)
            else:
                file_vector.append(0)
        for x in range(len(feature_vector_each_file_list)):
            cos_prod_cat=[]
            for y in range(len(feature_vector_each_file_list[x])):
                cosine_value=sum(map( operator.mul, file_vector,feature_vector_each_file_list[x][y] ))
                cos_prod_cat.append(cosine_value)
            cosine_product_list.append(cos_prod_cat)
        for xin in range(len(cosine_product_list)):
            for yin in range(len(cosine_product_list[xin])):
                cosproduct_list.append(cosine_product_list[xin][yin])
                cosproduct_list_cat.append(xin)
        index_list_dec=sorted(range(len(cosproduct_list)), key=lambda i: cosproduct_list[i], reverse=True)[:20]#this list consists of top 20element index
        for indices in index_list_dec:
            frequent_cat.append(cosproduct_list_cat[indices])
        frequent_cat=sorted(frequent_cat)
        for l in frequent_cat:
            freq_dict[l]=freq_dict[l]+1
        maximum=0
        for i in freq_dict:
            if(freq_dict[i]>maximum):
                maximum=freq_dict[i]
                key=i
        if(key==each_category_no):
            count+=1
    Accuracy=count/len(final_files_for_testing[each_category_no])
    print("Accuracy: ",Accuracy)
                
        
            
        
            
            
                
        
        
        




    
    
    





    
    





