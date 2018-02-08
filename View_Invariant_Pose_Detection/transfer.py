#import cv2
import tensorflow as tf
import numpy as np
import h5py

depth_max=200     # some constant number
B=10
J=15
H=40
W=40
D=40
C=15
def spatial_transformer_network(input_fmap, theta,volume, out_dims=None, **kwargs):       #volume=[10,15,40,40,40]
	#input=(10,160,160,15)
	#theta=(10,12)
	#Joints=(10,15,2)
	#point_cloud=(10,15,160,160,3)
	B0 = tf.shape(input_fmap)[0] #B=no of joints 
	H0 = tf.shape(input_fmap)[1] #H=height
	W0 = tf.shape(input_fmap)[2] #W=width
	C0 = tf.shape(input_fmap)[3] #C=channel
	#Voxel=tf.zeros((B,J,H,W,D))
	theta = tf.reshape(theta, [B, 3, 4])
	if out_dims:
		out_H = out_dims[0]
		out_W = out_dims[1]
		batch_grids = affine_grid_generator(out_H, out_W, theta)
	else:
		batch_grids = affine_grid_generator(B,H, W,D, theta)
	#points=get_cloud(point_cloud,Joints)  #10,15,25600,3
	#voxel=get_voxel(points,160,160,200)   #10,15,160,160,200
	#extract x and y coordinates
	batch_grids=tf.tile(tf.expand_dims(batch_grids,1),[1,C,1,1,1,1])  #shape of batch_grid is [10,15,3,H,W,D]

	x_s = tf.squeeze(batch_grids[:,:,0,:, :, :])            # shape=(10, 15, 40, 40,40)
	y_s = tf.squeeze(batch_grids[:,:, 1,:, :, :])           #shape=(10, 15, 40, 40, 40)
	z_s = tf.squeeze(batch_grids[:,:, 2,:, :, :])		#shape=(10, 15, 40, 40,40)	
	'''for j in range(B):
		Voxel[j]=get_voxel(cloud[j],H,W,D)'''
	embedd=trilinear_sampler(volume, x_s,y_s,z_s)          #volume_shape(10,15,40,40,40)
	#voxel,n1,n2=get_voxel(point_cloud)
	#out_fmap = trilinear_sampler(input_fmap,xs,ys,zs,voxel)
	#Cloud_joint=get_cloud(point_cloud,Joints)
	#voxel=get_voxel(Cloud_joint)
	#return out_fmap
	return embedd,batch_grids



def affine_grid_generator(num_batch,height, width,depth, theta):
	x = tf.linspace(-1.0, 1.0, width)
	y = tf.linspace(-1.0, 1.0, height)
	z = tf.linspace(-1.0, 1.0, depth)
	x_t, y_t , z_t = tf.meshgrid(x, y,z)
	x_t_flat = tf.reshape(x_t, [-1])
	y_t_flat = tf.reshape(y_t, [-1])
	z_t_flat = tf.reshape(z_t, [-1])
	
	# reshape to (x_t, y_t,z_t_flat , 1)   ************
	ones = tf.ones_like(x_t_flat)
	sampling_grid = tf.stack([x_t_flat, y_t_flat, z_t_flat, ones])
	
	print(sampling_grid.get_shape(),'preeeeeeeeeeeeeeeeeeeeeeeee')
	# repeat grid num_batch times
	sampling_grid = tf.expand_dims(sampling_grid, axis=0)
	#sampling_grid = tf.expand_dims(sampling_grid, axis=0)
	#print(sampling_grid.get_shape(),'preeeeeeeeeeeeeeeeeeeeeeeee')
	sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))
	#print(sampling_grid.get_shape(),'prossssssssssssssssssssssssssssssss')

	# cast to float32 (required for matmul)
	theta = tf.cast(theta, 'float32')
	sampling_grid = tf.cast(sampling_grid, 'float32')

	# transform the sampling grid - batch multiply
	batch_grids = tf.matmul(theta, sampling_grid)
	# batch grid has shape (num_batch, 2, H*W*D)
	
	# reshape to (num_batch, H, W,D, 2)
	batch_grids = tf.reshape(batch_grids, [num_batch, 3, height, width,depth])
	# batch_grids = tf.transpose(batch_grids, [0, 2, 1, 3])

	return batch_grids

def kernal(x):
	return tf.maximum(tf.cast(0,'float32'),tf.cast((1-x),'float32'))

def trilinear_sampler(img, x,y,z):
	#x_shape=(10, 15, 40, 40, 40)
	#img_hape=(10, 15, 160, 160, 200)
	# prepare useful params
	B0 = tf.shape(img)[0]
	H0 = tf.shape(img)[1]
	W0 = tf.shape(img)[2]
	C0 = tf.shape(img)[3]
	max_y = tf.cast(H - 1, 'int32')
	max_x = tf.cast(W - 1, 'int32')
	max_z = tf.cast(D - 1, 'int32')
	zero = tf.zeros([], dtype='int32')
	# cast indices as float32 (for rescaling)
	x = tf.cast(x, 'float32')
	y = tf.cast(y, 'float32')	
	z = tf.cast(z, 'float32')
	# rescale x and y to [0, W/H]
	x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
	y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))
	z = 0.5 * ((z + 1.0) * tf.cast(D, 'float32'))
	# grab 8  nearest corner points for each (x_i, y_i)
	x0 = tf.cast(tf.floor(x-1), 'int32')
	x1 = x0 + 1
	y0 = tf.cast(tf.floor(y-1), 'int32')
	y1 = y0 + 1
	z0 = tf.cast(tf.floor(z-1), 'int32')
	z1 = z0 + 1
	# clip to range [0, H/W] to not violate img boundaries
	x0 = tf.clip_by_value(x0, zero, max_x)
	x1 = tf.clip_by_value(x1, zero, max_x)
	y0 = tf.clip_by_value(y0, zero, max_y)
	y1 = tf.clip_by_value(y1, zero, max_y)
	z0 = tf.clip_by_value(z0, zero, max_z)
	z1 = tf.clip_by_value(z1, zero, max_z)
	# get pixel value at corner coords
	Ia0 = get_pixel_value(img, x0, y0,z0)
	Ib0 = get_pixel_value(img, x0, y1,z0)
	Ic0 = get_pixel_value(img, x1, y0,z0)
	Id0 = get_pixel_value(img, x1, y1,z0)
	Ia1 = get_pixel_value(img, x0, y0,z1)
	Ib1 = get_pixel_value(img, x0, y1,z1)
	Ic1 = get_pixel_value(img, x1, y0,z1)
	Id1 = get_pixel_value(img, x1, y1,z1)
	# recast as float for delta calculation
	x0 = tf.cast(x0, 'float32')
	x1 = tf.cast(x1, 'float32')
	y0 = tf.cast(y0, 'float32')
	y1 = tf.cast(y1, 'float32')
	z0 = tf.cast(z0, 'float32')
	z1 = tf.cast(z1, 'float32')
	# calculate deltas
	#print(z1.get_shape(),'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
	wa0 = (x1-x) * (y1-y) * (z-z0)
	wb0 = (x1-x) * (y-y0) * (z-z0)
	wc0 = (x-x0) * (y1-y) * (z-z0)
    	wd0 = (x-x0) * (y-y0) * (z-z0)
    	wa1 = (x1-x) * (y1-y) * (z1-z)
    	wb1 = (x1-x) * (y-y0) * (z1-z)
    	wc1 = (x-x0) * (y1-y) * (z1-z)
    	wd1 = (x-x0) * (y-y0) * (z1-z)
    	# add dimension for addition
    	#wa0 = tf.expand_dims(wa0, axis=4)
    	#wb0 =tf.expand_dims(wb0, axis=4)
    	#wc0 = tf.expand_dims(wc0, axis=4)
   	#wd0 = tf.expand_dims(wd0, axis=4)
    	#wa1 = tf.expand_dims(wa1, axis=4)
    	#wb1 = tf.expand_dims(wb1, axis=4)
    	#wc1 = tf.expand_dims(wc1, axis=4)
    	#wd1 = tf.expand_dims(wd1, axis=4)
    	# compute output
	#print(wd1.get_shape(),'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    	out = tf.add_n([wa0*Ia0, wb0*Ib0, wc0*Ic0, wd0*Id0,wa1*Ia1, wb1*Ib1, wc1*Ic1, wd1*Id1])
	two_D=tf.reshape(tf.reduce_sum(out, 4),[B,H,W,C])
    	return two_D




def get_pixel_value(img, x, y,z):
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    depth=shape[3]
    '''batch_idx1 = tf.range(0, 10)
    batch_idx = tf.reshape(batch_idx, (10, 1,1, 1))
    b = tf.tile(batch_idx, (1, 160, 160,200))'''
    batch_idx1 = tf.range(0, B)
    batch_idx2 = tf.range(0, C)
    i1, i2 = tf.meshgrid(tf.range(B),tf.range(C), indexing="ij")
    i1 = tf.cast(tf.tile(i1[:, :, tf.newaxis,tf.newaxis,tf.newaxis], [1, 1,H,W,D]),'float32')
    i2 =tf.cast(tf.tile(i2[:, :, tf.newaxis,tf.newaxis,tf.newaxis], [1, 1,H,W,D]),'float32')        #[10,15,160,200,200]
    x=tf.cast(x,'float32')
    y=tf.cast(y,'float32')
    z=tf.cast(z,'float32')
    
    indices = tf.cast(tf.stack([i1, i2,x,y,z], axis=-1),'int32')

    #print(b.get_shape,batch_idx.get_shape(),'asdfghjkl;sdfghjkl;sdfghjkldfghj')
    #indices = tf.stack([b,  x,y,z],4)
    print(indices.get_shape(),img.get_shape(),'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
    return tf.gather_nd(img, indices)


def trilinear_sampler(img, x,y,z):
        B0 = tf.shape(img)[0]
        H0 = tf.shape(img)[1]
        W0 = tf.shape(img)[2]
        C0 = tf.shape(img)[3]
        #Three_D=tf.zeros((B,J,H,W,D)) 
        Two_D=tf.zeros((B,J,H,W))
        for i in range(B):
                for j in range(J):
                        for m in range(H):
                                for n in range(W):
                                        out=0
                                        for l in range(D):
                                                sum1=0
                                                for a in range(H):
                                                        for b in range(W):
                                                                for c in range(D):
                                                                        x=tf.cast(tf.gather_nd(x,[i,j,m,n,l]),'float32')
                                                                        y=tf.cast(tf.gather_nd(y,[i,j,m,n,l]),'float32')
                                                                        z=tf.cast(tf.gather_nd(z,[i,j,m,n,l]),'float32')
                                                                        #z=0
                                                                        H0=tf.cast(H,'float32')
                                                                        W0=tf.cast(W,'float32')
                                                                        D0=tf.cast(D,'float32')
                                                                        v=tf.cast(tf.gather_nd(img,[i,j,a,b,c]),'float32')
                                                                        k=v * kernal((a-x)/H) * kernal((b-y)/W) * kernal((c-z)/D)
                                                                        #print(sum1)    
                                                                        sum1=sum1+k
                                                                        out=out+k
                                        tf.scatterTwo_D[j][m][n]=out
                                        print(out)
        return Two_D

'''





'''Voxel=tf.Variable(tf.cast(tf.zeros((10,15,160,160,200)),'float32'))
def get_voxel(cloud):
	#cloud=tf.cast(cloud,'float32')
	#num_batch=10
	#num_joint=15
	#Voxel = tf.get_variable("Voxel",shape=(10,15,160,160,200), dtype=tf.float32, initializer=tf.zeros_initializer())
	#Voxel=tf.Variable(tf.cast(tf.zeros((10,15,160,160,200)),'float32'))
	x=tf.floor(15*cloud[:,:,:,:,0])
	y=tf.floor(15*cloud[:,:,:,:,1])
	z=tf.floor(15*cloud[:,:,:,:,2])
	x_min=tf.reduce_min(x)
	y_min=tf.reduce_min(y)
	z_min=tf.reduce_min(z)
	x=tf.cast(tf.clip_by_value(x-x_min,0,159),'float32')
	y=tf.cast(tf.clip_by_value(y-y_min,0,159),'float32')
	z=tf.cast(tf.clip_by_value(z-z_min,0,199),'float32')
	#points=tf.cast(tf.stack([x,y,z],axis=4),'int32')
	batch_idx1 = tf.range(0, 10)
	batch_idx2 = tf.range(0, 15)
	batch_idx1 = tf.reshape(batch_idx1, (10, 1, 1,1))
	batch_idx2 = tf.reshape(batch_idx2, (1,15, 1, 1))
	b1 = tf.cast(tf.tile(batch_idx1, (1, 15,160,160)),'float32')
	b2 = tf.cast(tf.tile(batch_idx2, (10, 1,160,160)),'float32')
	indices = tf.cast(tf.stack([b1,b2, x,y,z ], 4),'int32')
	update=tf.cast(tf.ones((10,15,160,160)),'float32')
	voxel=tf.scatter_nd_update(Voxel,indices,update)
	n1=tf.count_nonzero(cloud)
	n2=tf.count_nonzero(voxel)
	return voxel,n1,n2
	for b in range(num_batch):
		for j in range(num_joint):
			refer=tf.Variable(Voxel[b,j,:,:,:])
			indices=points[b,j,:,:,:]
			globals()['g_{0}'.format(j)]=tf.scatter_nd_update(refer,indices,update)
		globals()['B_{0}'.format(b)]=tf.stack([g_0,g_1,g_2,g_3,g_4,g_5,g_6,g_7,g_8,g_9,g_10,g_11,g_12,g_13,g_14],axis=0)
		#del g_0,g_1,g_2,g_3,g_4,g_5,g_6,g_7,g_8,g_9,g_10,g_11,g_12,g_13,g_14
	voxel=tf.stack([B_0,B_1,B_2,B_3,B_4,B_5,B_6,B_7,B_8,B_9],axis=0)
	n1=tf.count_nonzero(cloud)
	n2=tf.count_nonzero(voxel)
	return voxel,n1,n2'''




















''' 
	Two_D=tf.zeros((B,H,W,J))
	for i in range(B):
		for j in range(J):
			for m in range(H):
				for n in range(W):
					out=0
					for l in range(D):
						sum1=0
						for a in range(H):
							for b in range(W):
								for c in range(D):
									x=tf.cast(tf.gather_nd(grid,[i,0,m,n,l]),'float32')
									y=tf.cast(tf.gather_nd(grid,[i,1,m,n,l]),'float32')
									z=tf.cast(tf.gather_nd(grid,[i,2,m,n,l]),'float32')
									#z=0
									H0=tf.cast(H,'float32')
									W0=tf.cast(W,'float32')
									D0=tf.cast(D,'float32')
									v=tf.cast(tf.gather_nd(voxel,[i,j,a,b,c]),'float32')
									k=v * kernal((a-x)/H) * kernal((b-y)/W) * kernal((c-z)/D)
									#print(sum1)	
									sum1=sum1+k
									out=out+k
					Two_D[m][n][j]=out
					print(out)
	return Two_D
					

	# rescale x and y to [0, W/H]
	x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
	y = 0.5 * ((y + 1.0) * tf.cast(H, 'float32'))
	z = 0.5 * ((z + 1.0) * tf.cast(depth_max, 'float32'))

	# grab 4 nearest corner points for each (x_i, y_i)
	# i.e. we need a rectangle around the point of interest
	x0 = tf.cast(tf.floor(x), 'int32')
	x1 = x0 + 1
	y0 = tf.cast(tf.floor(y), 'int32')
	y1 = y0 + 1
	z0 = tf.cast(tf.floor(z), 'int32')
        z1 = y0 + 1


	# clip to range [0, H/W] to not violate img boundaries
	x0 = tf.clip_by_value(x0, zero, max_x)
	x1 = tf.clip_by_value(x1, zero, max_x)
	y0 = tf.clip_by_value(y0, zero, max_y)
	y1 = tf.clip_by_value(y1, zero, max_y)
	z0 = tf.clip_by_value(y0, zero, max_z)
        z1 = tf.clip_by_value(y1, zero, max_z)

	# get pixel value at corner coords
	Ia0 = get_pixel_value(img, x0, y0,z0)
	Ib0 = get_pixel_value(img, x0, y1,z0)
	Ic0 = get_pixel_value(img, x1, y0,z0)
	Id0 = get_pixel_value(img, x1, y1,z0)
	Ia1 = get_pixel_value(img, x0, y0,z1)
	Ib1 = get_pixel_value(img, x0, y1,z1)
	Ic1 = get_pixel_value(img, x1, y0,z1)
	Id1 = get_pixel_value(img, x1, y1,z1)



	# recast as float for delta calculation
	x0 = tf.cast(x0, 'float32')
	x1 = tf.cast(x1, 'float32')
	y0 = tf.cast(y0, 'float32')
	y1 = tf.cast(y1, 'float32')
	z0 = tf.cast(z0, 'float32')
        z1 = tf.cast(z1, 'float32')

	# calculate deltas
	wa0 = (x1-x) * (y1-y) * (z-z0)
	wb0 = (x1-x) * (y-y0) * (z-z0)
	wc0 = (x-x0) * (y1-y) * (z-z0)
	wd0 = (x-x0) * (y-y0) * (z-z0)
	wa1 = (x1-x) * (y1-y) * (z1-z)
	wb1 = (x1-x) * (y-y0) * (z1-z)
	wc1 = (x-x0) * (y1-y) * (z1-z)
	wd1 = (x-x0) * (y-y0) * (z1-z)

	# add dimension for addition
	wa0 = tf.expand_dims(wa0, axis=3)
	wb0 = tf.expand_dims(wb0, axis=3)
	wc0 = tf.expand_dims(wc0, axis=3)
	wd0 = tf.expand_dims(wd0, axis=3)
	wa1 = tf.expand_dims(wa1, axis=3)
	wb1 = tf.expand_dims(wb1, axis=3)
	wc1 = tf.expand_dims(wc1, axis=3)
	wd1 = tf.expand_dims(wd1, axis=3)'''

''' def get_voxel(cloud,dimx,dimy,dimz):
	#size=cloud.shape[0]
	size=B
	Joint=J
	cloud=tf.cast(cloud,'float32')
	x=tf.floor(tf.squeeze(cloud[:,:,:,0])*100)
	y=tf.floor(tf.squeeze(cloud[:,:,:,1])*100)
	z=tf.floor(tf.squeeze(cloud[:,:,:,2])*100)
	x = tf.cast(x, 'int32')
	y = tf.cast(y, 'int32')
	z = tf.cast(z, 'int32')
	x_max=tf.reduce_max(x)
	x_min=tf.reduce_min(x)
	y_max=tf.reduce_max(y)
	y_min=tf.reduce_min(y)
	z_max=tf.reduce_max(z)
	z_min=tf.reduce_min(z)
	x=x-x_min
	y=y-y_min
	z=z-z_min
	Cx=x_max-x_min+1
	Cy=y_max-y_min+1
	Cz=z_max-z_min+1
	print(Cx)
	voxel=tf.zeros((B,J,Cx,Cy,Cz))
	for i in range(B):
		for j in range(J):
			a=x[i,j]
			b=y[i,j]
			c=z[i,j]
			voxel[a,b,c]=1
	#voxel=tf.resize(voxel, (dimx,dimy,dimz))
	p.close()
	return voxel'''

'''
def get_voxel(cloud):
        #size=cloud.shape[0]
        update=tf.ones((25600))
        cloud=tf.cast(cloud,'float32')
        x=tf.floor(tf.squeeze(cloud[:,:,:,0])*100)
        y=tf.floor(tf.squeeze(cloud[:,:,:,1])*100)
        z=tf.floor(tf.squeeze(cloud[:,:,:,2])*100)
        cld=tf.floor(tf.squeeze(cloud[:,:,:,:])*100)
        x = tf.cast(x, 'int32')
        y = tf.cast(y, 'int32')
        z = tf.cast(z, 'int32')
        x_max=tf.reduce_max(x)
        x_min=tf.reduce_min(x)
        y_max=tf.reduce_max(y)
        y_min=tf.reduce_min(y)
        z_max=tf.reduce_max(z)
        z_min=tf.reduce_min(z)
        x=x-x_min
        y=y-y_min
        z=z-z_min
        Cx=x_max-x_min+1
        Cy=y_max-y_min+1
        Cz=z_max-z_min+1
        print(Cx)
        u=tf.cast(tf.zeros((10,15,100,100,100)),'int32')
        voxel=tf.Variable(tf.zeros( [10,15,160,160,200]))
        for i in range(B):
                for j in range(J):
                        a=tf.cast(tf.gather_nd(x,[i,j]),'int32')
                        b=tf.cast(tf.gather_nd(y,[i,j]),'int32')
                        c=tf.cast(tf.gather_nd(z,[i,j]),'int32')
                        #I=tf.constant(i)
                        #J0=tf.constant(j)
                        I=tf.cast(j*tf.ones((25600)),'int32')
                        J0=tf.cast(j*tf.ones((25600)),'int32')
                        a=tf.clip_by_value(a,0,159)
                        b=tf.clip_by_value(b,0,159)
                        c=tf.clip_by_value(c,0,199)
                        I=tf.clip_by_value(I,0,9)
                        J0=tf.clip_by_value(J0,0,14)
                        #voxel=voxel[I,J0,a,b,c].assign(1)
                        a=tf.expand_dims(a, axis=1)
                        b=tf.expand_dims(b, axis=1)
                        c=tf.expand_dims(c, axis=1)
                        I=tf.expand_dims(I, axis=1)
                        J0=tf.expand_dims(J0, axis=1)
                        #I=tf.cast(j*tf.ones((25600)),'int32')
                        #J0=tf.cast(j*tf.ones((25600)),'int32')
                        indices=tf.cast(tf.stack([I,J0,a,b,c],axis=1),'int32')
                        index=tf.squeeze(indices,axis=2)
                        voxel=tf.scatter_nd_update(voxel,index,update)
        return voxel
def get_cloud(cloud,joints):
        voxel=tf.Variable(tf.zeros((10,15,25600,3)))
	cloud=tf.cast(cloud,'float32')
	joints=tf.cast(joints,'float32')
        for b in range(B):               
                #batch=tf.constant([0,0,0])
                #batch=tf.expand_dims(batch, axis=0)  
                for j in range(J):
                        b0=tf.constant(b)
                        #part=tf.constant([0,0,0])
                        #part=tf.expand_dims(part, axis=0)  #[1,3]
                        x,y=joints[b,j,0],joints[b,j,1]
                        x1=tf.clip_by_value(tf.cast(x,'int32')-80,0,239)
                        y1=tf.clip_by_value(tf.cast(y,'int32')-80,0,319)
                        x2=tf.clip_by_value(tf.cast(x,'int32')+80,0,239)
                        y2=tf.clip_by_value(tf.cast(x,'int32')-80,0,319)
                        for m in range(160):
                                for n in range(160):
                                        #voxel=tf.Variable(voxel)
                                        k=tf.clip_by_value(240*(x1+m)+(y1+n),0,76799)
                                        point0=tf.Variable(tf.gather_nd(cloud,[b,k]))
                                        #point1=tf.Variable(tf.gather_nd(cloud,[b,k,1]))
                                        #point2=tf.Variable(tf.gather_nd(cloud,[b,k,2]))
                                        voxel[b,j,k].assign(point0)
                                        #voxel=tf.Variable(voxel[b,j,k,1]).assign(point1)
                                        #voxel=tf.Variable(voxel[b,j,k,2]).assign(point2)
                                        point=tf.expand_dims(point, axis=0)  #(1,3)  
                                        part=tf.concat([part,point], 0)                                   
                        part=tf.expand_dims(part, axis=0)
                        batch=tf.concat([batch,part], 0)
                batch=tf.expand_dims(batch, axis=0)
                voxel=tf.concat([voxel,batch], 0)
        return voxel'''	
