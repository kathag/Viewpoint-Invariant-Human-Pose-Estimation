#import cv2
import h5py
import numpy as np
import tensorflow as tf
from spatial_transfer import spatial_transformer_network as stn
from layers import *
from operation import *
#placeholders
x = tf.placeholder(tf.float32, [10, 240,320], name='x')
joints=tf.placeholder(tf.float32, [10, 15,2], name='joints')
y = tf.placeholder(tf.uint8, [10,15,3], name='y')
Cloud=tf.placeholder(tf.float32,[10,76800,3],name='cloud')

#Voxel=tf.placeholder(tf.uint8, [10,76800,3], name='y')
##phase = tf.placeholder(tf.bool, name='phase')
#Cloud=tf.placeholder(tf.float32,[10,76800,3],name='cloud')

#parameters
learning_rate=0.01
Epochs=10
learning_rate = 0.01
batch_size=10
mini_batch_size=10 
refinement_steps=10
rnn_num_hidden = 24 
H_image, W_image, C_image = 240,320,1
H,W,D,C=40,40,40,15    # parameters of Glimpse
B,J=10,15

## Model = Localisation networks + VGG16   + RNN with LSTM 

def model(x,joints,Cloud):
	filter1 = tf.get_variable('weights1', [10,6400,2048], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)
	filter2 = tf.get_variable('weights2', [10,2048,2048], initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32), dtype=tf.float32)


	weights = {
	# Weights for localisation network  
        'W_stn_conv1':tf.Variable(tf.random_normal([3, 3,15, 32])),
        'W_stn_conv2':tf.Variable(tf.random_normal([3, 3, 32, 32])),
        'W_stn_conv3':tf.Variable(tf.random_normal([3, 3, 32, 32])),
	'W_stn_conv4':tf.Variable(tf.random_normal([3, 3, 32, 32])),
        'W_stn_fc': tf.Variable(tf.random_normal([5*5*32,12])),
	
	
        #Weigths for VGG-16
	'W_conv1_1': tf.Variable(tf.random_normal([3,3,15,32])),
        'W_conv1_2': tf.Variable(tf.random_normal([3, 3, 32, 32])),

        'W_conv2_1': tf.Variable(tf.random_normal([3, 3, 32, 64])),
        'W_conv2_2': tf.Variable(tf.random_normal([3, 3, 64, 64])),

        'W_conv3_1': tf.Variable(tf.random_normal([3,3, 64, 128])),
        'W_conv3_2': tf.Variable(tf.random_normal([3, 3, 128,128])),
        'W_conv3_3': tf.Variable(tf.random_normal([3, 3, 128, 128])),

        'W_conv4_1': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        'W_conv4_2': tf.Variable(tf.random_normal([3, 3, 256, 256])),
        'W_conv4_3': tf.Variable(tf.random_normal([3, 3, 256, 256])),

        'W_conv5_1': tf.Variable(tf.random_normal([3, 3, 256, 256])),
        'W_conv5_2': tf.Variable(tf.random_normal([3, 3, 256, 256])),
        'W_conv5_3': tf.Variable(tf.random_normal([3, 3, 256,256])),

        'W_fc6': tf.Variable(tf.random_normal([10,2*2*256, 2048])),
        'W_fc7': tf.Variable(tf.random_normal([10,2048, 2048])),
        'W_fc8': tf.Variable(tf.random_normal([10,2048,45]))

	}	
	

	biases = {
	#Biases for localisation-network
        'b_stn_conv1':tf.Variable(tf.random_normal([32])),
        'b_stn_conv2':tf.Variable(tf.random_normal([32])),
        'b_stn_conv3':tf.Variable(tf.random_normal([32])),
        'b_stn_conv4':tf.Variable(tf.random_normal([32])),
        'b_stn_fc':tf.Variable(tf.random_normal([12])),

	#Biases for VGG-16

	'b_conv1_1': tf.Variable(tf.random_normal([32])),
        'b_conv1_2': tf.Variable(tf.random_normal([32])),

        'b_conv2_1': tf.Variable(tf.random_normal([64])),
        'b_conv2_2': tf.Variable(tf.random_normal([64])),

        'b_conv3_1': tf.Variable(tf.random_normal([128])),
        'b_conv3_2': tf.Variable(tf.random_normal([128])),
        'b_conv3_3': tf.Variable(tf.random_normal([128])),

        'b_conv4_1': tf.Variable(tf.random_normal([256])),
        'b_conv4_2': tf.Variable(tf.random_normal([256])),
        'b_conv4_3': tf.Variable(tf.random_normal([256])),


        'b_conv5_1': tf.Variable(tf.random_normal([256])),
        'b_conv5_2': tf.Variable(tf.random_normal([256])),
        'b_conv5_3': tf.Variable(tf.random_normal([256])),


        'b_fc6': tf.Variable(tf.random_normal([2048])),
        'b_fc7': tf.Variable(tf.random_normal([2048])),
	'b_fc8': tf.Variable(tf.random_normal([45]))


	}
	
	print(x.get_shape(),'Input_shape')

	glimpse=get_glimpse(x,joints)
	Glimpse=tf.reshape(glimpse,[B,2*H,2*W,J])        # size of glimpse is [10,80,80,15]
	print(Glimpse.get_shape(),'Glimpse_shape')
	
	points=get_cloud(Cloud,joints)
	print(points.get_shape(),'Point_cloud_shape')

	Volume_voxel=get_voxel(points)
	print(Volume_voxel.get_shape(),'Volume_voxel_shape')

	print('Localisation Network start')

	conv_stn1=conv_layer(Glimpse,weights['W_stn_conv1'],biases['b_stn_conv1'] )   # 32 filter of size 3*3 stride=1 padding=1
        pool_stn1=max_pool(conv_stn1, 'pool_stn1')                                    #pool 2*2 region with stride=2
	print(pool_stn1.get_shape(),'pool1')

        conv_stn2=conv_layer(pool_stn1,weights['W_stn_conv2'],biases['b_stn_conv2'] )   # 32 filter of size 3*3 stride=1 padding=1
        pool_stn2=max_pool(conv_stn2, 'pool_stn2')
	print(pool_stn2.get_shape(),'pool2')

        conv_stn3=conv_layer(pool_stn2,weights['W_stn_conv3'],biases['b_stn_conv3'] )   # 32 filter of size 3*3 stride=1 padding=1
        pool_stn3=max_pool(conv_stn3, 'pool_stn3')
	print(pool_stn3.get_shape(),'pool3')


	conv_stn4=conv_layer(pool_stn3,weights['W_stn_conv4'],biases['b_stn_conv4'] )   # 32 filter of size 3*3 stride=1 padding=1
        pool_stn4=max_pool(conv_stn4, 'pool_stn4')
	print(pool_stn4.get_shape(),'pool4')
	
	pool_stn4_flat =tf.reshape(pool_stn4 , [-1, 5*5*32])
	print(pool_stn4_flat.get_shape(),'pool_flat')
	theta = tf.nn.relu(tf.matmul(pool_stn4_flat,weights['W_stn_fc'] ) +biases['b_stn_fc'])
	#print(theta.get_shape(),'pppppppppppppppppp')

	theta =tf.reshape(theta , [-1,12])

	print(theta.get_shape(),'Theta_parameter for spatial Transfer network')
		
	#Call spatial Transfer network by using stn defined in spatial_Transfer_network_file
	embedd,grid=stn(Glimpse,theta,Volume_voxel)                        # embedded has size [10,40,40,15]
	print('Localisation Network End')

	#####################################################################################
	
	print('VGG_Network Start From Here')

	conv1_1 = conv_layer(embedd,weights['W_conv1_1'],biases['b_conv1_1'] )
	print(conv1_1.get_shape(),'conv11')
        conv1_2 = conv_layer(conv1_1, weights['W_conv1_2'],biases['b_conv1_2'])
	print(conv1_2.get_shape(),'conv11')
        pool1 = max_pool(conv1_2, 'pool1')
	print(pool1.get_shape(),'pool1')

        conv2_1 = conv_layer(pool1,weights['W_conv2_1'], biases['b_conv2_1'] )
	print(conv2_1.get_shape(),'conv21')
        conv2_2 = conv_layer(conv2_1,weights['W_conv2_2'], biases['b_conv2_2'])
        print(conv2_2.get_shape(),'conv22')
	pool2 = max_pool(conv2_2, 'pool2')
	print(pool2.get_shape(),'pool2')

        conv3_1 = conv_layer(pool2, weights['W_conv3_1'],biases['b_conv3_1'])
        print(conv3_1.get_shape(),'conv31')
	conv3_2 = conv_layer(conv3_1,weights['W_conv3_2'], biases['b_conv3_2'])
	print(conv3_2.get_shape(),'conv32')
        conv3_3 = conv_layer(conv3_2,weights['W_conv3_3'], biases['b_conv3_3'])
	print(conv3_3.get_shape(),'conv33')
        pool3 = max_pool(conv3_3, 'pool3')
	print(pool3.get_shape(),'pool3')

        conv4_1 =conv_layer(pool3,weights['W_conv4_1'], biases['b_conv4_1'])
	print(conv4_1.get_shape(),'conv41')
        conv4_2 = conv_layer(conv4_1,weights['W_conv4_2'], biases['b_conv4_2'])
	print(conv4_2.get_shape(),'conv42')
        conv4_3 =conv_layer(conv4_2, weights['W_conv4_3'], biases['b_conv4_3'])
        print(conv4_3.get_shape(),'conv43')
	pool4 = max_pool(conv4_3, 'pool4')
	print(pool4.get_shape(),'pool4')


        conv5_1 = conv_layer(pool4, weights['W_conv5_1'] ,biases['b_conv5_1'])
	print(conv5_1.get_shape(),'conv51')
        conv5_2 = conv_layer(conv5_1, weights['W_conv5_2'], biases['b_conv5_2'])
	print(conv5_2.get_shape(),'conv52')
        conv5_3 = conv_layer(conv5_2, weights['W_conv5_3'] , biases['b_conv5_3'])
	print(conv5_3.get_shape(),'conv53')
        pool5 = max_pool(conv5_3, 'pool5')
	print(pool5.get_shape(),'pool5')


	pool5_flat =tf.reshape(pool5 , [-1,1, 2*2*256])
	print(pool5_flat.get_shape(),'pool5_flat')
	fc6=tf.nn.relu(tf.matmul(pool5_flat, weights['W_fc6']  ) +biases['b_fc6'])
	print(fc6.get_shape(),'fc6')
	
	fc7=tf.nn.relu(tf.matmul(fc6, weights['W_fc7']  ) +biases['b_fc7'])
        print(fc7.get_shape(),'fc7')
	#Remove Last dense layer of VGG16
	#fc8=tf.nn.relu(tf.matmul(fc7, filter3 ) +biases['b_fc8'])
        #print(fc8.get_shape(),'fc8')
	data=tf.reshape(fc7 , [-1,2048])
	print(data.get_shape(),'VGG_output')
        #fc6 = fc_layer(pool5_flat,  weights['W_fc6'] ,biases['b_fc6'])
        #assert fc6.get_shape().as_list()[1:] == [4096]
        #relu6 = tf.nn.relu(fc6)
        #fc7 = self.fc_layer(relu6,weights['W_fc7'] , biases['b_fc7'])
        #relu7 = tf.nn.relu(fc7)
	data1=tf.expand_dims(data,axis=0)
        #fc8 = self.fc_layer(relu7, weights['W_fc8'], biases['b_fc8'])
        #relu8=tf.nn.relu(fc8)
	#pool5_flat =tf.reshape(pool5 , [-1,15,1, 5*5*256])
	cell = tf.nn.rnn_cell.LSTMCell(2048,state_is_tuple=True)
	initial_state = cell.zero_state(10, tf.float32)
	val, state = tf.nn.dynamic_rnn(cell, data1, dtype=tf.float32)
	print(val.get_shape(),'rnn_out')
	val_flat=tf.expand_dims(tf.squeeze(val),axis=1)
	print(val_flat.get_shape(),'rnn_out_after_flat')
	#############################################################
 	out=tf.nn.relu(tf.matmul(val_flat, weights['W_fc8']  ) +biases['b_fc8'])	
	#result =tf.reshape(out , [10,15,3])
	print(out.get_shape(),'final')

	return Glimpse,points,Volume_voxel,embedd,grid




def generate_batch_indices(X,Valid, batch_size):
        num_train = X.shape[0]
        total_batch = int(np.ceil(num_train / float(batch_size)))
        batch_indices = [(i * batch_size, min(num_train, (i + 1) * batch_size))
                         for i in range(0, total_batch)]
        return total_batch, batch_indices


def main():
	depth_maps_side = h5py.File('../ITOP_side_test_depth_map.h5', 'r')['data'][100:200].astype(np.float32)
	Joint_part=h5py.File('../ITOP_side_test_labels.h5', 'r')['image_coordinates'][100:200].astype(np.float32)
	point_cloud=h5py.File('../ITOP_side_test_point_cloud.h5', 'r')['data'][100:200].astype(np.float32)
	Valid=h5py.File('../ITOP_side_test_labels.h5', 'r')['is_valid'][100:200].astype(np.float32)
	Target=h5py.File('../ITOP_side_test_labels.h5', 'r')['real_world_coordinates'][100:200].astype(np.float32)
	X_train=depth_maps_side

	
	Glimpse,points,Volume_voxel,embedd,grid=model(x,joints,Cloud)
	y=tf.reshape(tf.cast(y,'float32'),[10,15,3])
	cross_entropy = tf.losses.mean_squared_error(logits=val, labels=y)
	loss = tf.reduce_mean(cross_entropy)
	loss= tf.reduce_mean(tf.squared_difference(val, y))
	logits=val

	global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step)

	define accuracy
	correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	initial_position=tf.constant(np.average(Target, axis=0))
	MODE='train'
	print('Session start')

	with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                if MODE == 'train':
                        print("Training on samples of length",X_train.shape[0])
                        iter_per_epoch, batch_indices = generate_batch_indices(X_train,Valid,10)
                        epoch_num = 0
                        for i in range(iter_per_epoch):
				print('Train By Batch:',i)
                                idx = batch_indices[i]
                                mask = np.arange(idx[0], idx[1]).tolist()
				batch_X_train,batch_Joint,batch_y,batch_cloud= X_train[mask],Joint_part[mask],Target[mask],point_cloud[mask]
				train_feed_dict = {x: batch_X_train,joints: batch_Joint,y: batch_y,Cloud: batch_cloud}
				for itr in range(4):
					v,e,p,g,h = sess.run([Glimpse,points,Volume_voxel,embedd,grid], feed_dict=train_feed_dict)
					print('hello at iteration:',itr,np.count_nonzero(v),np.count_nonzero(e),np.count_nonzero(p),np.count_nonzero(g),np.count_nonzero(h))				


if __name__ == '__main__':
	main()



