import sys
import numpy as np
import cv2
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--img_dir", help="path to input hazy image")
parser.add_argument("--depth_dir", help="path to depth image")
parser.add_argument("--beta", type=float, default=2, help="number of beta to compute transmission map")
a = parser.parse_args()

def d2t(depth, beta):
    d = depth/255.0
    tr = np.exp(np.dot(-beta, d))
    t = tr * 255.0
    return t

def DarkChannel(im,sz):
	b,g,r = cv2.split(im)
	dc = cv2.min(cv2.min(r,g),b)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
	dark = cv2.erode(dc,kernel)
	return dark

def AtmLight(im,dark):
	[h,w] = im.shape[:2]
	imsz = h*w
	numpx = int(max(math.floor(imsz/1000),1))
	darkvec = dark.reshape(imsz,1)
	imvec = im.reshape(imsz,3)
	indices = darkvec.argsort()
	indices = indices[imsz-numpx::]
	atmsum = np.zeros([1,3])
	for ind in range(1,numpx):
		atmsum = atmsum + imvec[indices[ind]]
	A = atmsum / numpx
	return A

def Guidedfilter(im,p,r,eps):
	mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
	mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
	mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
	cov_Ip = mean_Ip - mean_I*mean_p
	mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
	var_I   = mean_II - mean_I*mean_I
	a = cov_Ip/(var_I + eps)
	b = mean_p - a*mean_I
	mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
	mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
	q = mean_a*im + mean_b
	return q

def TransmissionRefine(im,et):
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	gray = np.float64(gray)/255
	r = 60
	eps = 0.0001
	t = Guidedfilter(gray,et,r,eps)
	return t

def Recover(im,t,A,tx=0.1):
	res = np.empty(im.shape,im.dtype)
	t = cv2.max(t,tx)
	for ind in range(0,3):
		res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
	return res

if __name__ == '__main__':

	src = cv2.imread(a.img_dir)
	depth = cv2.imread(a.depth_dir,cv2.IMREAD_GRAYSCALE)
	beta = a.beta

	I = src/255.0
	trans = d2t(depth, beta)
	dark = DarkChannel(I,15)
	A = AtmLight(I,dark)
	te = trans/255.0
	t = TransmissionRefine(src,te)
	J = Recover(I,t,A)
	cv2.imshow('TransmissionEstimate',te)
	cv2.imshow('TransmissionRefine',t)
	cv2.imshow('Origin',src)
	cv2.imshow('Dehaze',J)
	cv2.waitKey(0)
	save_path_t1 = a.img_dir[:-4]+'_t'+a.img_dir[-4:len(a.img_dir)]
	save_path = a.img_dir[:-4]+'_Dehaze'+a.img_dir[-4:len(a.img_dir)]
	save_path_t = a.img_dir[:-4]+'_trans'+a.img_dir[-4:len(a.img_dir)]
	cv2.imwrite(save_path,J*255)
	cv2.imwrite(save_path_t, t*255)
	cv2.imwrite(save_path_t1, trans*255)
