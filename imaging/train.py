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
	import scipy.ndimage as ndimage
	img = nib.load(filename)
	print(img.get_fdata().shape)
  	flip_x = np.flip(img.get_fdata(),0)
  	flip_y = np.flip(img.get_fdata(),1)
  	flip_z = np.flip(img.get_fdata(),2)
	flip_img_x = nib.Nifti1Image(flip_x,affine = np.eye(4))
	flip_img_y = nib.Nifti1Image(flip_y,affine = np.eye(4))
	flip_img_z = nib.Nifti1Image(flip_z,affine = np.eye(4))
	gaussian_img = nib.Nifti1Image(ndimage.gaussian_filter(img.get_fdata(),sigma=(5,5,0), order=0),affine=np.eye(4))
	from nilearn import image
	smooth_img = image.smooth_img(img,[50.1,10.0,0.2])
	nib.save(flip_img_x, os.path.join('images','flip_x.'+filename.split('/')[-1]))
	nib.save(flip_img_y, os.path.join('images','flip_y.'+filename.split('/')[-1]))
	nib.save(flip_img_z, os.path.join('images','flip_z.'+filename.split('/')[-1]))
	nib.save(smooth_img, os.path.join('images','smooth.'+filename.split('/')[-1]))
	print(type(smooth_img))
	print(type(gaussian_img))
	nib.save(gaussian_img, os.path.join('images','gaussian.'+filename.split('/')[-1]))

def parcelation(filename):
	from nilearn import plotting 
	from nilearn import datasets
	parcellations = datasets.fetch_atlas_basc_multiscale_2015(version='sym')
	networks_64 = parcellations['scale064']
	networks_197 = parcellations['scale197']
	networks_444 = parcellations['scale444']
	x0,y0 = 580,300
	img = nib.load(filename)
	data_orig = img.get_fdata()[...,1]#misc.face()
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
