import json
import os
import sys
import numpy as np
import argparse
from nilearn import datasets
from nilearn import plotting
import nibabel as nib

import matplotlib.pyplot as plt
from scipy import misc
from scipy.ndimage import rotate

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

def rot(image, xy, angle):
    im_rot = rotate(image,angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot, new+rot_center

def transformation(filename):
	from nilearn import plotting 
	from nilearn import datasets
	x0,y0 = 300,200
	img = nib.load(filename)
	print(img.get_fdata().shape)
  	flip_x = np.flip(img.get_fdata(),0)
  	flip_y = np.flip(img.get_fdata(),1)
  	flip_z = np.flip(img.get_fdata(),2)
	#flip_x.to_filename('more_smooth_anat_img.nii.gz')		
	new_img = nib.Nifti1Image(flip_x,affine = np.eye(4))
	nib.save(new_img, os.path.join('images','flip_x.'+filename.split('/')[-1]))
	#data_orig = img.get_fdata()[...,1]#misc.face()
	#print(data_orig.shape)
	#fig,axes = plt.subplots(1,2)
	#axes[0].imshow(data_orig,cmap = plt.cm.gray, vmin = 10, vmax = 200)
	#axes[0].scatter(x0,y0,c="r" )
	#axes[0].set_title("original")
	#data_rot, (x1,y1) = rot(data_orig, np.array([x0,y0]), 66)
	##data_rot = rot(data_orig, 66)
	#axes[1].imshow(data_rot,cmap = plt.cm.gray)
	#axes[1].scatter(x1,y1,c="r" )
	#axes[1].set_title("Rotation: {} deg".format(66))
	#plt.show()
	##plotting.plot_roi(img, cmap=plotting.cm.bwr_r,title='yale imaging')
	#plotting.show()

def parcelation(filename):
	from nilearn import plotting 
	from nilearn import datasets
	parcellations = datasets.fetch_atlas_basc_multiscale_2015(version='sym')
	networks_64 = parcellations['scale064']
	networks_197 = parcellations['scale197']
	networks_444 = parcellations['scale444']
	x0,y0 = 580,300
	img = nib.load(filename)
	data_orig = img.get_fdata()[1,...]#misc.face()
	print(data_orig.shape)
	fig,axes = plt.subplots(1,2)
	axes[0].imshow(data_orig)
	axes[0].scatter(x0,y0,c="r" )
	axes[0].set_title("original")
	data_rot, (x1,y1) = rot(data_orig, np.array([x0,y0]), 66)
	axes.flatten()[1].imshow(data_rot)
	axes.flatten()[1].scatter(x1,y1,c="r" )
	axes.flatten()[1].set_title("Rotation: {}deg".format(66))
	plt.show()
	plotting.plot_roi(img, cmap=plotting.cm.bwr_r,title='yale imaging')
	plotting.show()
	#plotting.plot_roi(nib.fliplr(img), cmap=plotting.cm.bwr_r,title='yale imaging')

def main(argv):
	#parcelation('images/MNI_T1_1mm.nii.gz')
	transformation('images/MNI_T1_1mm.nii.gz')


if __name__ == '__main__':
	main(sys.argv)
