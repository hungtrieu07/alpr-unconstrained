
import numpy as np
import cv2
import time

from os.path import splitext

from src.label import Label
from src.utils import getWH, nms
from src.projection_utils import getRectPts, find_T_matrix


class DLabel (Label):

	def __init__(self,cl,pts,prob):
		self.pts = pts
		tl = np.amin(pts,axis=1)
		br = np.amax(pts,axis=1)
		Label.__init__(self,cl,tl,br,prob)

def save_model(model,path,verbose=0):
	path = splitext(path)[0]
	model_json = model.to_json()
	with open('%s.json' % path,'w') as json_file:
		json_file.write(model_json)
	model.save_weights('%s.h5' % path)
	if verbose: print(('Saved to %s' % path))

def load_model(path,custom_objects={},verbose=0):
	from keras.models import model_from_json

	path = splitext(path)[0]
	with open('%s.json' % path,'r') as json_file:
		model_json = json_file.read()
	model = model_from_json(model_json, custom_objects=custom_objects)
	model.load_weights('%s.h5' % path)
	if verbose: print(('Loaded from %s' % path))
	return model

def normal(pts, side, mn, MN):
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop

def reconstruct(I,Iresized,Yr,threshold=.9):
	net_stride 	= 2**4
	side 		= ((208. + 40.)/2.)/net_stride

	one_line = (470, 110)
	two_lines = (280, 200)

	Probs = Yr[...,0]
	Affines = Yr[...,2:]

	xx,yy = np.where(Probs > threshold)
	# CNN input image size
	WH = getWH(Iresized.shape)
	# output feature map size
	MN = WH/net_stride

	vxx = vyy = 0.5 #alpha

	base = lambda vx,vy: np.matrix([[-vx,-vy,1],[vx,-vy,1],[vx,vy,1],[-vx,vy,1]]).T
	labels = []
	labels_frontal = []

	for i in range(len(xx)):
		x, y = xx[i],yy[i]
		affine = Affines[x,y]
		prob = Probs[x,y]

		mn = np.array([float(y) + 0.5,float(x) + 0.5])

		# affine transformation matrix
		A = np.reshape(affine,(2,3))
		A[0,0] = max(A[0,0],0)
		A[1,1] = max(A[1,1],0)
		# identity transformation
		B = np.zeros((2,3))
		B[0,0] = max(A[0,0],0)
		B[1,1] = max(A[1,1],0)

		pts = np.array(A*base(vxx,vyy)) #*alpha
		pts_frontal = np.array(B*base(vxx,vyy))

		pts_prop = normal(pts, side, mn, MN)
		frontal = normal(pts_frontal, side, mn, MN)

		labels.append(DLabel(0,pts_prop,prob))
		labels_frontal.append(DLabel(0, frontal, prob))

	final_labels = nms(labels,0.1)
	final_labels_frontal = nms(labels_frontal, 0.1)

	assert final_labels_frontal, "No licese plate is found!"

	out_size, lp_type = (two_lines, 2) if ((final_labels_frontal[0].wh()[0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)

	TLp = []
	Cor = []
	if len(final_labels):
		final_labels.sort(key=lambda x: x.prob(), reverse=True)
		for _,label in enumerate(final_labels):
			t_ptsh 	= getRectPts(0,0,out_size[0],out_size[1])
			ptsh 	= np.concatenate((label.pts*getWH(I.shape).reshape((2,1)),np.ones((1,4))))
			H 		= find_T_matrix(ptsh,t_ptsh)
			Ilp 	= cv2.warpPerspective(I,H,out_size,borderValue=0)
			TLp.append(Ilp)
			Cor.append(ptsh)
	return final_labels,TLp, lp_type, Cor
	

def detect_lp(model, I, max_dim, lp_threshold):
    min_dim_img = min(I.shape[:2])
    factor = float(max_dim) / min_dim_img
    w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist()
    Iresized = cv2.resize(I, (w, h))
    T = Iresized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    Yr = model.predict(T)
    Yr = np.squeeze(Yr)
    #print(Yr.shape)
    L, TLp, lp_type, Cor = reconstruct(I, Iresized, Yr, lp_threshold)
    return L, TLp, lp_type, Cor