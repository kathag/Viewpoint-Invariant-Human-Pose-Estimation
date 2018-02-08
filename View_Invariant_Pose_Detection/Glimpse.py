import tensorflow as tf

def get_Glimpse(input_map, joints):
	H = tf.shape(input_fmap)[0]
	W = tf.shape(input_fmap)[1]
	#C = tf.shape(input_fmap)[2]
	for j in range(len(joints)):
        	x, y = joints[j, 0], joints[j, 1]
