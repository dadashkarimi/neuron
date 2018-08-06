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
#with open('unlabled.csv','w') as g:
#	g.write('[\n    ')
#	for root,dirs, files in os.walk('.'):
#		train = np.zeros((268,268,len(files)))
#		z=0
#		for filename in files:
#			with open(filename) as f: 	
#				i = 0
#				for line in f.readlines():
#					j = 0 
#					for item in line.rstrip().split('\t'):
#						train[]	
#						j = j+1
#					i = i + 1
#			z = z+1
#			#g.write('],\n\t\t')
	#g.write(']')
