import h5py
import numpy as np
import cv2
depth_maps = h5py.File('ITOP_side_test_depth_map.h5', 'r')
labels = h5py.File('ITOP_side_test_labels.h5', 'r')
point_cloud= h5py.File('ITOP_side_test_point_cloud.h5', 'r')
points=100*np.array(point_cloud['data'][1005])
i=1005
depth_map = depth_maps['data'][i].astype(np.float32)
joints = labels['image_coordinates'][i]
img = cv2.normalize(depth_map, depth_map, 0, 1, cv2.NORM_MINMAX)
img = np.array(img * 255, dtype = np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
for i in range(0,240):
	for j in range(0,320):
		x=320*i+j
		print (points[x][0], points[x][1], points[x][2], img[i][j][0], img[i][j][1], img[i][j][2])


