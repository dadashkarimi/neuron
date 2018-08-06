from sklearn.metrics import mean_squared_error

y=[]
yp=[]
with open('all_behav.csv') as f:
	for line in f.readlines():
		y.append(float(line))
with open('sr.svr.0.5.1.0.predict.xml') as f:
	for line in f.readlines():
		yp.append(float(line))	

sum_ = 0.0
for i in range(len(y)):
	sum_ += (y[i]-yp[i]) **2
print(mean_squared_error(y,yp))
