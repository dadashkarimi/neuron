import json
import os
import sys
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import argparse
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-file', '-f', nargs='+' ,help='data file for train and test using 5-fold')

parser.add_argument('--model', '-m', help='[lrg(linear regression),svr(support vector regression),dt (decision tree),rf(random forest)]')
parser.add_argument('--agg', '-agg', help='[nmf(non-negative matrix factorization), avg(average), max(maximum), min(minimum)]')
args = parser.parse_args()

train = np.zeros((len(args.file),268,268,713))
ttrain = np.zeros((len(args.file),713,268,268)) # transpose
X = np.zeros((len(args.file),713,268*268)) # reshape
y= np.array([0]*713)
kf = KFold(n_splits=5)

######################### Data Loading ############################
with open('all_behav_713.csv') as f:
	y = [float(line) for line in f.readlines()]

y=np.array(y)
dataList=[]

for file_ in args.file:
	with open(file_) as f:
	    dataList.append(json.load(f))

for i in range(268):
	for j in range(268):
		for k in range(713):
			for z in range(len(dataList)):
				train[z][i][j][k] = float(dataList[z][i][j][k])
				ttrain[z][k][i][j] = float(dataList[z][i][j][k])

for k in range(713):
	for z in range(len(dataList)):
		X[z][k] = np.reshape(ttrain[z][k],-1)


################### Normalization ##############################
for z in tqdm(range(len(dataList))):
	scalar = MinMaxScaler()
	X[z] = scalar.fit_transform(X[z])

############################ NMF ################################
if len(dataList)==1:
	X= X[0]
elif len(dataList)==2 and args.agg=='nmf': # NMF
	print('nmf starting ..')
	Z = np.zeros((713,268*268)) # reshape
	for i in range(713):
		W=[]
		H=[]
		mf = NMF(n_components=10, init='random', random_state=1)
		for j in range(len(args.file)):
			W.append(mf.fit_transform(X[j][i][:].reshape(268,268)))
			H.append(mf.components_)
		Z[i] = np.matmul(W[0],H[1]).reshape(268*268)
	X = Z
	print('nmf done!')
elif len(dataList)==2 and args.agg=='avg': # average
	Z = np.zeros((713,268*268)) # reshape
	for i in range(len(args.file)):
		Z = np.add(Z,X[i])
	X = Z/len(args.file)	
elif len(dataList)==2 and args.agg=='max': # average
	Z = np.zeros((713,268*268)) # reshape
	for i in range(len(args.file)):
		Z = np.maximum(Z,X[i])
	X = Z
elif len(dataList)==2 and args.agg=='min': # average
	Z = 10*np.ones((713,268*268)) # reshape
	for i in range(len(args.file)):
		Z = np.minimum(Z,X[i])
	X = Z
########################### K-Fold ###############################
try:
    os.remove('.'.join(args.file)+'.'+args.model+'.predict.xml')
except OSError:
    pass

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
#clf =SVR(kernel='rbf', C=1e3, gamma=0.1) 
svr = SVR(kernel='poly', C=1e3, degree=2)
#svr = svm.SVR(C=1000, epsilon=0.0001)
reg = linear_model.Lasso(alpha = 0.1)
rf = RandomForestClassifier(max_depth=2, random_state=0)
ab= AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=2),
    n_estimators=600,
    learning_rate=1.5,
    algorithm="SAMME")
gnb = MultinomialNB() #GaussianNB()
lrg=LogisticRegression()
model =dt
if args.model=='lrg':
	model= lrg
elif args.model=='svr':
	model=svr
elif args.model=='rf':
	model=rf
elif args.model=='av':
	model==ab
elif args.model=='gnb':
	model=gnb

for train_index, test_index in kf.split(X):
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	model.fit(X_train,y_train) # learning the model
	with open('.'.join(args.file)+'.'+args.model+'.predict.xml','a') as f:
		for i in range(len(X_test)):
			f.write(str(model.predict([X_test[i]])[0])+'\n')

