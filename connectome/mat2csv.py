import scipy.io
import numpy as np
import json 

data = scipy.io.loadmat("all_behav_843.mat")

for i in data:
	if '__' not in i and 'readme' not in i:
		#np.savetxt((i+".csv"),data[i],delimiter=',')
		with open('test.txt', 'a') as f: 
			f.write(str(data))
			#f.write(json.dumps(data[i], default=lambda x: list(x), indent=4))
			#f.write('\n')
