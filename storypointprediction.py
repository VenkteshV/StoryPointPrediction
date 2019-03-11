
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df = pd.read_csv('appceleratorstudio.csv')
df


# In[2]:


import json
import requests
response  = requests.get('use your vsts token and hit the endpoint with the pbi numbers to get pbi details')


# In[3]:


responseDeserialized = response.json()
# data = pd.read_json(json.dumps(responseDeserialized))
# data
workItems = responseDeserialized.get('value')
accumulator = []
for workItem in workItems:
    accumulator.append((workItem.get('id'),workItem.get('fields').get('System.Title'), (workItem.get('fields').get('System.Description')), (workItem.get('fields').get('Microsoft.VSTS.Scheduling.Effort'))))

newDf = pd.DataFrame(accumulator, columns = ['issuekey' , 'title', 'description' , 'storypoint'])
df = df.append(newDf,ignore_index=True)
df


# In[4]:



df.isnull().sum()

df = df.dropna(how='any')
df


# In[5]:


df.storypoint.describe()


# In[6]:


import matplotlib.pyplot as plt
 
plt.hist(df.storypoint, bins=20, alpha=0.6, color='b')
plt.title("#Items per Point")
plt.xlabel("Points")
plt.ylabel("Count")
 
plt.show()


# In[7]:


df.groupby('storypoint').size()


# In[8]:


df.loc[df.storypoint <= 2, 'storypoint'] = 0 #small
df.loc[(df.storypoint > 2) & (df.storypoint <= 5), 'storypoint'] = 1 #medium
df.loc[df.storypoint > 5, 'storypoint'] = 2 #big


# In[9]:


df.groupby('storypoint').size()


# In[10]:


import numpy as np
import csv
# from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#Define some known html tokens that appear in the data to be removed later
htmltokens = ['{html}','<div>','<pre>','<p>','<b>','<br>','</br>','</b>','<span>','</span>', '</div>','</pre>','</p>']

#Clean operation
#Remove english stop words and html tokens
def cleanData(text):
    
    result = ''
    
    for w in htmltokens:
        text = text.replace(w, '')
    
    text_words = text.split()    
    
    resultwords  = [word for word in text_words if word not in stopwords.words('english')]
    
    if len(resultwords) > 0:
        result = ' '.join(resultwords)
    else:
        print('Empty transformation for: ' + text)
        
    return result

def formatFastTextClassifier(label):
    return "__label__" + str(label) + " "


# In[11]:


df['title_desc'] = df['title'].str.lower() + ' - ' + df['description'].str.lower()
df['label_title_desc'] = df['storypoint'].apply(lambda x: formatFastTextClassifier(x)) + df['title_desc'].apply(lambda x: cleanData(str(x)))


# In[12]:


df = df.reset_index(drop=True)
df


# In[13]:


from collections import Counter

def SimpleOverSample(_xtrain, _ytrain):
    xtrain = list(_xtrain)
    ytrain = list(_ytrain)

    samples_counter = Counter(ytrain)
    max_samples = sorted(samples_counter.values(), reverse=True)[0]
    for sc in samples_counter:
        init_samples = samples_counter[sc]
        samples_to_add = max_samples - init_samples
        if samples_to_add > 0:
            #collect indices to oversample for the current class
            index = list()
            for i in range(len(ytrain)):
                if(ytrain[i] == sc):
                    index.append(i)
            #select samples to copy for the current class    
            copy_from = [xtrain[i] for i in index]
            index_copy = 0
            for i in range(samples_to_add):
                xtrain.append(copy_from[index_copy % len(copy_from)])
                ytrain.append(sc)
                index_copy += 1
    return xtrain, ytrain


# In[18]:


import uuid
import subprocess,shlex
import os
class FastTextClassifier:

    rand = ""
    inputFileName = ""
    outputFileName = ""
    testFileName = ""
    
    def __init__(self):
        self.rand = str(uuid.uuid4())
        self.inputFileName = "issues_train_" + self.rand + ".txt"
        self.outputFileName = "supervised_classifier_model_" + self.rand
        self.testFileName = "issues_test_" + self.rand + ".txt"
    
    def fit(self, xtrain, ytrain):
        outfile=open(self.inputFileName, mode="w", encoding="utf-8")
        for i in range(len(xtrain)):
            #line = "__label__" + str(ytrain[i]) + " " + xtrain[i]
            line = xtrain[i]
            outfile.write(line + '\n')
        outfile.close()     
        command_line = "./fasttext supervised -input " +self.inputFileName +" -output " + self.outputFileName + " -epoch 500 -wordNgrams 4 -dim  300 -minn 4 -maxn 6 -pretrainedVectors pretrain_model.vec"
        args = shlex.split(command_line)
        p1 = subprocess.Popen(args, stdout=subprocess.PIPE )
        p1.communicate()[0].decode("utf-8").split("\r\n")
        
        
    def predict(self, xtest):
        #save test file
        outfile=open(self.testFileName, mode="w", encoding="utf-8")
        for i in range(len(xtest)):
            outfile.write(xtest[i] + '\n')
        outfile.close()
        #get predictions
        command2 = "./fasttext predict "  + self.outputFileName +  ".bin " + self.testFileName
        args1 = shlex.split(command2)
        p1 = subprocess.Popen(args1, stdout=subprocess.PIPE)
        output_lines = p1.communicate()[0].decode("utf-8").split("\n")
        test_pred = [int(float(p.replace('__label__',''))) for p in output_lines if p != '']
        return test_pred


# In[21]:


import pandas as pd
import numpy as np

pretrain_files = ['apache_pretrain.csv',
                 'jira_pretrain.csv',
                 'spring_pretrain.csv',
                 'talendforge_pretrain.csv',
                 'moodle_pretrain.csv',
                 'appcelerator_pretrain.csv',
                 'duraspace_pretrain.csv',
                 'mulesoft_pretrain.csv',
                 'lsstcorp_pretrain.csv']

pretrained = None

for file in pretrain_files:
        df_pretrain = pd.read_csv('PretrainData/' + file, usecols=['issuekey', 'title', 'description'])
        if(pretrained is not None):
            pretrained = pd.concat([pretrained, df_pretrain])
        else:
            pretrained = df_pretrain

pretrained = pretrained.dropna(how='any')


# In[22]:


pretrained['title_desc'] = (pretrained['title'].str.lower() + ' - ' + pretrained['description'].str.lower()).apply(lambda x: cleanData(str(x)))

outfile=open("issues_pretrain.txt", mode="w", encoding="utf-8")
for line in pretrained.title_desc.values:
    outfile.write(line + '\n')
outfile.close()


# In[19]:



def rebuild_kfold_sets(folds, k, i):
    training_set = None
    testing_set = None
 
    for j in range(k):
        if(i==j):
            testing_set = folds[i]
        elif(training_set is not None):
            training_set = pd.concat([training_set, folds[j]])
        else:
            training_set = folds[j]
    
    return training_set, testing_set


# In[20]:


import itertools
import os
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_confusion_matrix_with_accuracy(classes, y_true, y_pred, title, sum_overall_accuracy, total_predictions):
    cm = ConfusionMatrix(y_true, y_pred) 
    print('Current Overall accuracy: ' + str(cm.stats()['overall']['Accuracy']))
    if total_predictions != 0:
        print('Total Overall Accuracy: ' + str(sum_overall_accuracy/total_predictions))
    else:
        print('Total Overall Accuracy: ' + str(cm.stats()['overall']['Accuracy']))

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=classes, title=title)
    plt.show()


# In[21]:


from sklearn.metrics import confusion_matrix
from pandas_ml import ConfusionMatrix
import os
# K-folds cross validation 
# K=5 or K=10 are generally used. 
# Note that the overall execution time increases linearly with k
k = 5

# Define the classes for the classifier
classes = ['0','1','2']

# Make Dataset random before start
df_rand = df.sample(df.storypoint.count(), random_state=99)
# Number of examples in each fold
fsamples =  int(df_rand.storypoint.count() / k)

# Fill folds (obs: last folder could contain less than fsamples datapoints)
folds = list()
for i in range(k):
    folds.append(df_rand.iloc[i * fsamples : (i + 1) * fsamples])
        
# Init
sum_overall_accuracy = 0
total_predictions = 0

# Repeat k times and average results
for i in range(k):
    
    #1 - Build new training and testing set for iteration i
    training_set, testing_set  = rebuild_kfold_sets(folds, k, i)
    y_true = testing_set.storypoint.tolist()

    #2 - Oversample (ONLY TRAINING DATA)
    X_resampled, y_resampled =  SimpleOverSample(training_set.label_title_desc.values.tolist(), training_set.storypoint.values.tolist())
    
    #3 - train
    clf = FastTextClassifier()
    clf.fit(X_resampled, y_resampled)
    
    #4 - Predict
    y_pred = clf.predict(testing_set.label_title_desc.values.tolist())
 #   print(y_pred)     
    #3 - Update Overall Accuracy
    for num_pred in range(len(y_pred)):
        if(y_pred[num_pred] == y_true[num_pred]):
            sum_overall_accuracy += 1
        total_predictions += 1

    #4 - Plot Confusion Matrix and accuracy 

    plot_confusion_matrix_with_accuracy(classes, y_true, y_pred, 'Confusion matrix (testing-set folder = ' + str(i) + ')', sum_overall_accuracy, total_predictions)

