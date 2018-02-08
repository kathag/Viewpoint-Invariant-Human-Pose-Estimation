import h5py
import numpy as np
import cv2
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
depth_maps = h5py.File('ITOP_side_test_depth_map.h5', 'r')
labels = h5py.File('ITOP_side_test_labels.h5', 'r')

def main():
	for i in range(depth_maps['data'].shape[0]):
		if labels['is_valid'][i]:
			depth_map = depth_maps['data'][i].astype(np.float32)
			joints = labels['image_coordinates'][i]
			img = depth_map_to_image(depth_map, joints)
			filename = "images/file_%d.jpg"%i
			cv2.imwrite(filename,img)

def depth_map_to_image(depth_map, joints=None):
    img = cv2.normalize(depth_map, depth_map, 0, 1, cv2.NORM_MINMAX)
    img = np.array(img * 255, dtype = np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = cv2.applyColorMap(img, cv2.COLORMAP_OCEAN)
    for j in range(15):
        x, y = joints[j, 0], joints[j, 1]
        cv2.circle(img, (x,y), 1, (255,255,255), thickness=2)
        cv2.putText(img, joint_id_to_name[j], (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255))
    return img
 
if __name__ == '__main__':
    main()




