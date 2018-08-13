import json
import os
import sys
import numpy as np
import argparse

from nilearn import datasets
from nilearn import plotting

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
parser.add_argument('--input', '-i', help='[lrg(linear regression),svr(support vector regression),dt (decision tree),rf(random forest)]')
args = parser.parse_args()


#motor_images = datasets.fetch_neurovault_motor_task()
#stat_img = motor_images.images[0]
display = plotting.plot_glass_brain("images/MNI_T1_1mm.nii")
plotting.show()
#display.show()
#display.save('d')
