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
from numpy import linalg
from scipy.linalg import eigh

class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def writelines(self, datas):
       self.stream.writelines(datas)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)

import sys
sys.stdout = Unbuffered(sys.stdout)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-file', '-f', nargs='+' ,help='data file for train and test using 5-fold')

parser.add_argument('--model', '-m', help='[lrg(linear regression),svr(support vector regression),dt (decision tree),rf(random forest)]')
parser.add_argument('--agg', '-agg', help='[nmf(non-negative matrix factorization), avg(average), max(maximum), min(minimum)]')
parser.add_argument('--semi', '-semi', help='[semi supervised: sr=spectral regression, svr = two stage svr, harmonic = harmonic regressionm, isotonic = isotonic regression]')
args = parser.parse_args()

train = np.zeros((len(args.file),268,268,713))
ttrain = np.zeros((len(args.file),713,268,268)) # transpose
X = np.zeros((len(args.file),713,268*268)) # reshape
Xp = np.zeros((1000,268*268)) # 1000 is arbitrary number
y = np.array([0]*713)
yp = np.array([0]*1000)
kf = KFold(n_splits=5)

######################### Data Loading ############################
with open('all_behav_713.csv') as f:
	y = [float(line) for line in f.readlines()]

y=np.array(y)
dataList=[]
for file_ in args.file:
	if file_=='unlabled.csv':
		with open(file_) as f:
			a = json.load(f)
			Xp = np.asarray(a).reshape(268*268,len(a[0][0]))
			Xp = np.transpose(Xp)
	else:
		with open(file_) as f:
	    		dataList.append(json.load(f))

if args.semi:
	a = np.asarray(np.asarray(dataList[0])).reshape(268*268,713)
	X = np.transpose(a)
else:
	for z in range(len(dataList)):
		a = np.asarray(np.asarray(dataList[z])).reshape(268*268,len(dataList[z][0][0]))
		X[z] = np.transpose(a)

print('** Data loaded. X:{} and Xp:{} **'.format(np.shape(X),np.shape(Xp)))

################### Normalization ##############################
scalar = MinMaxScaler()
Xp = scalar.fit_transform(Xp)
if args.semi: # 1 labled and 1 unlabled data
	X= scalar.fit_transform(X)
else: # we have multiple labled data
	for z in tqdm(range(len(dataList))):
		X[z] = scalar.fit_transform(X[z])
print('** Data normalized **')



def nmf(k,mode):
	if mode ==0: # data level
		print('nmf starting ..')
		W=[]
		H=[]
		mf = NMF(n_components=40, init='random', random_state=1)
		for j in range(len(args.file)):
			W.append(mf.fit_transform(X[j]))
			H.append(mf.components_)
		X = np.matmul(W[0],H[1])
	else: # subject level
		for i in range(713):
			W=[]
			H=[]
			mf = NMF(n_components=5, init='random', random_state=1)
			for j in range(len(args.file)):
				W.append(mf.fit_transform(X[j][i][:].reshape(268,268)))
				H.append(mf.components_)
			Z[i] = np.matmul(W[0],H[1]).reshape(268*268)
		X = Z
	print('nmf done!')
	return X

def isotoic_reg(X,y,Xp,only_knn=False):
	print(only_knn)
	from sklearn.isotonic import IsotonicRegression
	ir = IsotonicRegression()
	idx= y.argsort()
	X = X[idx]
	y = y[idx]
	yp = np.zeros(len(Xp))
	y_hat = ir.fit_transform(np.arange(len(idx)),y)

	def semi(X,y,Xp,yp):
		pred = []
		mse = 0.0
		for train_index, test_index in tqdm(kf.split(X)):
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			X_train = np.concatenate((X_train,Xp))
			y_train = np.concatenate((y_train,yp))
			model.fit(X_train,y_train) # learning the model
			for i in range(len(X_test)):
				pred_i = model.predict([X_test[i]])[0]
				pred.append(pred_i)
				mse += (pred_i - y[i])**2
		print('length of y  = {}'.format(len(y)))
		return np.array(pred), mse/len(y)


	def find_nearest(X,y,xpi):
	    idx = np.sum((X-xpi)**2,axis=1).argsort() 
	    return y[idx[0]]
	
	from sklearn.neighbors import KNeighborsClassifier
	neigh = KNeighborsClassifier(n_neighbors=1)
	pred , mse = [] , 0.0
	if only_knn:
		print('1 nearest neighbour .. ')
		neigh.fit(X,y)
		yp = [find_nearest(X,y,xpi) for xpi in tqdm(Xp)]
		pred, mse= semi(X,y,Xp,yp)
	else:
		neigh.fit(X,y_hat)
		yp = [find_nearest(X,y_hat,xpi) for xpi in tqdm(Xp)]
		pred, mse= semi(X,y_hat,Xp,yp)
	with open('isotonic.predict.xml','w') as f:
		for label in pred:
			f.write(str(label)+'\n')
		f.write(str(mse)+'\n')
	print('isotonic regression done with mse={}'.format(mse))
	
def two_stage_svr_curve(X,y,Xp):
	model.fit(X,y) # learning the model
	yp= np.array([0]*len(Xp))
	for i in range(len(Xp)):
		yp[i] = model.predict([Xp[i]])[0]
	data = []
	for i in tqdm(range(5)):
		idx = np.arange(700)
		np.random.shuffle(idx)
	  	
	
		idxp = np.arange(len(Xp))
		np.random.shuffle(idxp)
		
		X_test = X[idx[601:700]]
		y_test  = y[idx[601:700]]
		X_train = np.concatenate((X[idx[1:100]],Xp[idxp[1:200]]))
		y_train = np.concatenate((y[idx[1:100]],yp[idxp[1:200]]))
		model.fit(X_train,y_train) # learning the model
		sum_ = 0
		for i in range(len(X_test)):
			y_ = model.predict([X_test[i]])[0]
			sum_ = sum_+ (y_test[i]-y_)**2
		data.append(sum_/len(y_test))
	with open('100.200.txt','w') as f:
		f.write(str(np.mean(data))+'\t'+str(np.std(data)))


def two_stage_svr(X,y,Xp):
	try:
		os.remove('two_stage.svr.predict.xml')
    	except OSError:
  		pass

	sum_ = 0
	idx = np.arange(len(X))
	#np.random.shuffle(idx)
	#X = X[idx]
	#y=  y[idx]
	for train_index, test_index in tqdm(kf.split(X)):
		#mu, sigma = 0.0 , 0.0
		#epsilon = np.random.normal(mu, sigma, len(train_index))
		
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
			
		yp= np.array([0]*len(Xp))
		model = SVR(kernel='poly', C=1e3, degree=3)
		#model = SVR(kernel='rbf')
		model.fit(X_train,y_train) # learning the model
		for i in range(len(Xp)):
			yp[i] = model.predict([Xp[i]])[0]
		#model_  = SVR(kernel='poly', C=1e3, degree=2)
		#y_train = y_train #+ epsilon
		X_train = np.concatenate((X_train,Xp))
		y_train = np.concatenate((y_train,yp))
		model.fit(X_train,y_train) # learning the model
		#model_.fit(Xp,yp) # learning the model
		y_= np.zeros(len(X_test))
		alpha = 0.0
		for i in range(len(X_test)):
			#y_[i] = (1-alpha)*model.predict([X_test[i]])[0] + alpha *model_.predict([X_test[i]])[0]
			y_[i] = model.predict([X_test[i]])[0] 
		with open('two_stage.svr.predict.xml','a') as f:
			for i in range(len(X_test)):
				f.write(str(y_[i])+'\n')
				sum_ = sum_+ (y_test[i]-y_[i])**2
	with open('two_stage.svr.predict.xml','a') as f:
		f.write('mse='+str(sum_/len(y)))
		print('mse={}'.format(str(sum_/len(y))))
	print('svr > svr is done!')

def spectral_reg(X,y,Xp):
	X_ = np.vstack([X,Xp]) # R^m*(268*268)
	yp= np.array([0]*len(Xp))
	l = len(X) # number of labled data
	m = len(X_) # total number of examples (m-l = # unlabled data)
	print('l:{},m:{}'.format(np.shape(X),m))
	delta = 0.005
	lambda_ = 1.0
	etha= 4*(m-np.count_nonzero(y==y[0]))/lambda_*delta
	W = np.zeros((m,m))
	D = np.zeros((m,m))
	print('*** starting spectral regression .. !')
	for i in tqdm(range(m)):
		for j in range(m):
			if i < len(y) and j < len(y): # labled data
				if y[i] == y[j]:
					W[i][j] = 1.0/np.count_nonzero(y==y[i])
			else:	
				W[i][j] = delta*linalg.norm(X_[i]-X_[j])
		D[i][i] = np.sum(W[i])
	eigvals, eigvecs = eigh(W, D, eigvals_only=False)
	idx = eigvals.argsort()[::-1] # decreasing sort eig values 
	eigvals = eigvals[idx]
	eigvecs = eigvecs[idx]
	vec_0 = eigvecs[len(eigvecs)-1]
	print('etha={}'.format(etha))
	for i in range(len(X)):
		for j in range(len(X),len(X_)):
			#if abs(vec_0[j]-vec_0[i])<etha:	
			if abs(eigvals[j]-eigvals[i])<etha:	
				yp[j-len(X)] = y[i]
	
	try:
		os.remove('sr.predict.xml')
    	except OSError:
  		pass

	sum_ = 0.0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		X_train = np.concatenate((X_train,Xp))
		y_train = np.concatenate((y_train,yp))
		model.fit(X_train,y_train) # learning the model
		with open('sr.predict.xml','a') as f:
			for i in range(len(X_test)):
				t = model.predict([X_test[i]])[0]
				f.write(str(t)+'\n')
				sum_ = sum_+ (y_test[i]-t)**2
	with open('sr.predict.xml','a') as f:
		f.write('mse='+str(sum_/len(y)))
	print('.. spectral regression completed. mse = {} ***'.format(sum_/len(y_test)))

def harmonic_reg(X,y,Xp):
	X_ = np.vstack([X,Xp]) # R^m*(268*268)
	yp= np.array([0]*len(Xp))
	l = len(X) # number of labled data
	m = len(X_) # total number of examples (m-l = # unlabled data)
	W = np.zeros((m,m))
	D = np.zeros((m,m))
	f = np.zeros((m,1))
	
	print('*** starting random walk regression .. !')
	from scipy.spatial.distance import pdist, squareform
	import random
	W = squareform(pdist(X_,'minkowski', p=2.))
	for i in tqdm(range(m)):
		D[i][i] = np.sum(W[i])
	#for i in tqdm(range(m)):
	#	for j in range(m):
	#		W[i][j] = linalg.norm(X_[i]-X_[j])
			#W[i][j] = random.randint(1,10)#linalg.norm(X_[i]-X_[j])
	#	D[i][i] = np.sum(W[i])
	Duu = D[l:,l:]	
	Wuu = W[l:,l:]
	Wul = W[l:,:l]
	f = np.concatenate([y,yp])
	fl = f[:l]
	from numpy.linalg import inv
	fu = np.matmul(np.matmul(inv(Duu-Wuu),Wul),fl)
	f[l:] = fu
		
	try:
		os.remove('harmonic.predict.xml')
    	except OSError:
  		pass

	sum_ = 0.0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		X_train = np.concatenate((X_train,Xp))
		y_train = np.concatenate((y_train,fu))
		model.fit(X_train,y_train) # learning the model
		with open('harmonic.predict.xml','a') as f:
			for i in range(len(X_test)):
				t = model.predict([X_test[i]])[0]
				f.write(str(t)+'\n')
				sum_ = sum_+ (y_test[i]-t)**2
	with open('harmonic.predict.xml','a') as f:
		f.write('mse='+str(sum_/len(y)))
	print('.. radom walk regression completed. mse = {} ***'.format(sum_/len(y)))


def svr_reg(X,y):
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		model.fit(X_train,y_train) # learning the model
		with open('.'.join(args.file)+'.'+args.model+'.predict.xml','a') as f:
			for i in range(len(X_test)):
				f.write(str(model.predict([X_test[i]])[0])+'\n')

def gaussian(xi,xj,sigma):
	return np.exp(-(linalg.norm(xi-xj))/(2*sigma**2))

############################ NMF ################################
if len(dataList)==1 and not args.semi:
	X= X[0]
elif len(dataList)==2 and args.agg=='nmf': # NMF
	X = nmf(k,0)
elif len(dataList)>1 and args.agg=='avg': # average
	Z = np.zeros((713,268*268)) # reshape
	for i in range(len(args.file)):
		Z = np.add(Z,X[i])
	X = Z/len(args.file)	
elif len(dataList)>1 and args.agg=='max': # average
	Z = np.zeros((713,268*268)) # reshape
	for i in range(len(args.file)):
		Z = np.maximum(Z,X[i])
	X = Z
elif len(dataList)>1 and args.agg=='min': # average
	Z = 10*np.ones((713,268*268)) # reshape
	for i in range(len(args.file)):
		Z = np.minimum(Z,X[i])
	X = Z
try:
    try:
	os.remove('.'.join(args.file)+'.'+args.model+'.predict.xml')
    except OSError:
    	pass
    
except OSError:
    pass

dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
svr = SVR(kernel='poly', C=1e3, degree=2)
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
if args.semi=='svr' and len(args.file)==2: # one labled and one unlabled 
	two_stage_svr(X,y,Xp)
elif args.semi=='isotonic' and len(args.file)==2: # one labled and one unlabled 
	isotoic_reg(X,y,Xp,True)
elif args.semi =='sr' and len(args.file)==2:
	spectral_reg(X,y,Xp)
elif args.semi =='harmonic' and len(args.file)==2:
	harmonic_reg(X,y,Xp)
elif args.semi and len(args.file)!=2:
	print('please use single labled data.')
else:
	svr_reg(X,y)	
