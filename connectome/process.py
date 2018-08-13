import os
import json
import numpy as np

for root,dirs, files in os.walk('.txt'):
	train = np.zeros((len(files),268,268))
	z = 0
	print(np.shape(np.transpose(train)))
	for filename in files:
		with open(filename) as f: 	
			M = np.array(list([line.strip().split('\t') for line in f]))
		
			print(filename)
			train[z][:][:] = M
			z = z +1	
	with open('unlabled.csv','w') as g:
		g.write(json.dumps(train))
