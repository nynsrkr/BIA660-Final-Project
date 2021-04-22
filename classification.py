import os
# print(os.path.abspath(os.curdir))
os.chdir(".")
path = os.path.abspath(os.curdir)
current_path = path.replace('\\', '/')

# Utility Functions

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

"""
A  script that demonstrates how we classify three different profession (data engineer, data scientist and software engineer)indeed data and train model
with sklearn.

"""
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")
import time


# A List of Items
items = list(range(0, 100))
l = len(items)

# Data scraped from indeed
df = pd.read_excel(current_path + '/Training_Data.xls', encoding = 'utf8')

######Preprocessing steps#######

# To assign 0,1 and 2 to "Data Engineer job", "Data Scientist" and "Software Engineer" job
# list(le.inverse_transform([0, 1, 2]))-- ['Data Engineer', 'Data Scientist', 'Software Engineer'] ##

le = LabelEncoder()
df['Label_Encoded'] = le.fit_transform(df[['Label']])
# Dropped Title and Label column
df.drop(['Title','Label'],axis =1,inplace = True)

print('\n')
print('Loading Data for Pre Processing ... Please Wait ...')
print('\n')
printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

# Removed the most obvious words from job description
df['Description']=df['Description'].apply(lambda x: re.sub('data sci[a-z]+', ' ', x, re.I))
df['Description']=df['Description'].apply(lambda x: re.sub('data eng[a-z]+', ' ', x, re.I))
df['Description']=df['Description'].apply(lambda x: re.sub('software eng[a-z]+', ' ', x, re.I))

# Assigned X which is independent variable and y which is label that is dependent variable
X_train= df['Description']
y_train= df['Label_Encoded']

df_test = pd.read_excel(current_path + '/X_Test.xls',encoding = 'utf8')
df_test['Description']=df_test['Description'].apply(lambda x: re.sub('data sci[a-z]+', ' ', x, re.I))
df_test['Description']=df_test['Description'].apply(lambda x: re.sub('data eng[a-z]+', ' ', x, re.I))
df_test['Description']=df_test['Description'].apply(lambda x: re.sub('software eng[a-z]+', ' ', x, re.I))

X_test = df_test['Description']
df_test1 = pd.read_excel(current_path + '/y_test.xls',encoding = 'utf8')
df_test1['Label_Encoded'] = le.fit_transform(df_test1[['Label']])
y_test = df_test1['Label_Encoded']

time.sleep(4)
print('\n')
print('Pre-processing completed !!')
print('\n')
printProgressBar(100, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

time.sleep(1)
print('\n')
print('Training Model ... Please Wait ...')
print('\n')
printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

#Build a counter based on the training dataset
counter = CountVectorizer()
counter.fit(X_train)

#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(X_train)#transform the training data
counts_test = counter.transform(X_test)#transform the testing data
#train classifier
clf1 = LogisticRegression(solver='lbfgs', max_iter = 2500)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GradientBoostingClassifier()
clf4 = KNeighborsClassifier(n_neighbors=15)
clf5 = DecisionTreeClassifier(max_depth = 10, random_state = 101, max_features = None, min_samples_leaf = 15)

print('\n')
print('Model Trainng completed !!')
print('\n')
printProgressBar(100, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

time.sleep(1)
print('\n')
print('Running Voting classifier ... Please Wait ...')
print('\n')
printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

#voting classifier to determine the best classifier
eclf1 = VotingClassifier(estimators=[('lg',clf1),
                                    ('rf', clf2), ('gnb', clf3), ('knc', clf4),('dtc',clf5)], voting='soft')
eclf1 = eclf1.fit(counts_train, y_train)
pred = (eclf1.predict(counts_test))
printProgressBar(85, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

print('\n')
print('Classification completed !!')
print('\n')
printProgressBar(100, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

time.sleep(1)
print('\n')
print('Calculating Prediction Accuracy and writing output file ...')
print('\n')
printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
#Saving all the predicted data in csv file name predictions.csv
L1 = le.inverse_transform(pred)
pred_df = pd.DataFrame(L1)
pred_df.to_csv(current_path + '/predictions.csv', header = False, index = False)

print('\n')
print('Classification completed !!')
print('\n')
printProgressBar(100, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

#prints the accuracy
print('\n')
print("Prediction Accuracy = " + str(accuracy_score(pred,y_test)*100))
print("\n Final Predicted labels are saved in predictions.csv file under " + path)
print("\n Thank you Professor for an Awesome Web Mining Course !!!")