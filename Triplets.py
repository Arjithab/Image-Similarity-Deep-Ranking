import os
import pandas as pd
from random import randint, choice

rootDir = "tiny-imagenet-200/train/"
file_table = pd.DataFrame(columns = ['classid','class','image','inclass','outclass'])
fileSet = []
i = 0
img_class = 0

for dir_, _, files in os.walk(rootDir): #{n02504458/image,[],imagenames.jpeg}
    #dir_ = 'tiny-imagenet-200/train/n02504458/images/'
    #files = [.jpeg,.jpeg,.... 500 images]
    for fileName in files:#for every jpeg file within files
        relDir = os.path.relpath(dir_, rootDir)#Gets the ralative path => relDir = n02504458/images/
        relFile = os.path.join(relDir, fileName)#Joins with the filename
        if '.JPEG' in relFile:
            if i%500 == 0:
                img_class+=1
                print(img_class)
            file_table.loc[i,'image'] = relFile
            file_table.loc[i, 'class'] = relFile[0:9]
            file_table.loc[i,'classid'] = img_class
            i+=1 #All the 500 image names will be stored in the file_table and their corresponding class and classid are stored as well from 1 to 500.

file_table.columns = ['classid','class','query','inclass','outclass']
file_table['inclass'] = 0
file_table['outclass'] = 0

for index,row in file_table.iterrows(): #index will be the row number starting from 0 and row will become a series containing call the columns of the row. row = {'classid':1,'class':1,'query':'n02504458/images/img.jpeg,'inclass':0,'outclass':0}
    print(index)
    current_class = row[0]
    gen_start = (current_class-1)*500
    incls_index = index
    
    while incls_index == index:
        incls_index = randint(gen_start,gen_start+499) #incls_index will be a random number belonging to the class the image belongs to
    
    if current_class == 1:
        outcls_index = randint(gen_start+500,99999)
    elif current_class == 200:
        outcls_index = randint(0,gen_start-1)
    else:
        first_seg = randint(0,gen_start-1)
        second_seg = randint(gen_start+500,99999)
        outcls_index = choice([first_seg,second_seg]) 
    #outcls_index will be a random number belonging to the classes the image doesn''t belong to
    
    file_table.loc[index,'inclass'] = file_table.loc[incls_index,'query']
    file_table.loc[index,'outclass'] = file_table.loc[outcls_index,'query']
    
file_table.to_csv('file_table.csv',index = False)
#Finally we'll save it as a csv file consiting of roots to the triplets

#RANDOMLY Printing the generated triplets
w=10
h=10
fig=plt.figure(figsize=(12, 12))
columns = 3
rows = 4
for i in range(1,columns+1):
    row = randint(0,99999)
    query = file_table.loc[row,'query']
    inclass = file_table.loc[row,'inclass']
    outclass = file_table.loc[row,'outclass']
    query = os.path.join(rootDir,query)
    inclass = os.path.join(rootDir,inclass)
    outclass = os.path.join(rootDir,outclass)
    query_img = Image.open(query)
    inclass_img = Image.open(inclass)
    outclass_img = Image.open(outclass)
    fig.add_subplot(rows, columns, i)
    plt.imshow(query_img)
    fig.add_subplot(rows, columns, i+3)
    plt.imshow(inclass_img)
    fig.add_subplot(rows, columns, i+6)
    plt.imshow(outclass_img)
plt.show()