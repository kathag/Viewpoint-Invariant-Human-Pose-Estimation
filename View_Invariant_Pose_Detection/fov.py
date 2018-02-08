from __future__ import print_function

import numpy as np
import tensorflow as tf
from transformer import spatial_transformer_network
from keras.preprocessing.image import load_img, img_to_array, array_to_img


#import numpy as np
import cv2
from matplotlib import pyplot as plt
#from pyntcloud import PyntCloud
#import pyntcloud
joint_id_to_name = {
  0: 'Head',
  1: 'Neck',
  2: 'R Shoulder',
  3: 'L Shoulder',
  4: 'R Elbow',
  5: 'L Elbow',
  6: 'R Hand',
  7: 'L Hand',
  8: 'Torso',
  9: 'R Hip',
  10: 'L Hip',
  11: 'R Knee',
  12: 'L Knee',
  13: 'R Foot',
  14: 'L Foot',
}


def foveat(img,x,y):
        height = img.shape[0] # Get the dimensions
        width = img.shape[1]
        mask = 255*np.ones((height,width), dtype='uint8')
        cv2.circle(mask, (x,y), 1, (255,255,255), thickness=2)
        out = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        scale_factor = 10
        filtered = img.copy()
        img_float = img.copy().astype('float')
        # Number of channels
        if len(img_float.shape) == 3:
                num_chan = img_float.shape[2]
        #else:
          #num_chan = 1
          #img_float = img_float[:,:,None]
          #filtered = filtered[:,:,None]
        for y1 in range(height):
                for x1 in range(width):
                        if out[y1,x1] == 0.0:
                                continue
                        mask_val = np.ceil(out[y1,x1] / scale_factor)
                        if mask_val <= 3:
                                mask_val = 3
                        beginx = x1-int(mask_val/2)
                        if beginx < 0:
                                beginx=0
                        beginy = y1-int(mask_val/2)
                        if beginy < 0:
                                beginy=0
                        endx = x1+int(mask_val/2)
                        if endx >= width:
                                endx = width-1
			endy = y1+int(mask_val/2)
                        if endy >= height:
                                endy = height-1
                        xvals = np.arange(beginx, endx+1)
                        yvals = np.arange(beginy, endy+1)
                        (col_neigh,row_neigh) = np.meshgrid(xvals, yvals)
                        col_neigh = col_neigh.astype('int')
                        row_neigh = row_neigh.astype('int')
                        for ii in range(num_chan):
                                chan = img_float[:,:,ii]
                                pix = chan[row_neigh, col_neigh].ravel()
                                filtered[y,x,ii] = int(np.mean(pix))
        if num_chan == 1:
                filtered = filtered[:,:,0]
        return filtered


def load_data(dims, img_name, view=False):
        """
        Util function for processing RGB image into 4D tensor.

        Returns tensor of shape (1, H, W, C)
        """
        image_path = './data/' + img_name
        img = load_img(image_path, target_size=dims)
        if view:
                img.show()
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img
DIMS = (600, 600)
CAT1 = 'cat1.jpg'
def main():
        # out dims
        out_H = 400
        out_W = 400
        out_dims = (out_H, out_W)

        # load 4 cat images
        img1 = load_data(DIMS, CAT1, view=True)
 	img2=foveat(img1,300,300)
	img2.show()

if __name__ == '__main__':
        main()
                



