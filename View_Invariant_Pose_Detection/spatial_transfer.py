#import cv2
import tensorflow as tf
depth_max=200     # some constant number
B=10
J=15
H=160
W=160
D=200
C=1
def spatial_transformer_network(input_fmap, theta, Joints, point_cloud, out_dims=None, **kwargs):
	B0 = tf.shape(input_fmap)[0] #B=no of joints 
	H0 = tf.shape(input_fmap)[1] #H=height
	W0 = tf.shape(input_fmap)[2] #W=width
	C0 = tf.shape(input_fmap)[3] #C=channel
	#Voxel=tf.zeros((B,J,H,W,D))
	theta = tf.reshape(theta, [B,J, 3, 4])
	if out_dims:
		out_H = out_dims[0]
		out_W = out_dims[1]
		batch_grids = affine_grid_generator(out_H, out_W, theta)
	else:
		batch_grids = affine_grid_generator(B,J,H, W,D, theta)
	#voxel=get_voxel(cloud,160,160,200)
	#extract x and y coordinates
	'''x_s = tf.squeeze(batch_grids[:, 0:1, :, :])
	y_s = tf.squeeze(batch_grids[:, 1:2, :, :])
	z_s = tf.squeeze(batch_grids[:, 2:3, :, :])
	for j in range(B):
		Voxel[j]=get_voxel(cloud[j],H,W,D)'''
	
	#out_fmap = trilinear_sampler(input_fmap,batch_grids,voxel)
	Cloud_joint=get_cloud(point_cloud,Joints)
	voxel=get_voxel(Cloud_joint)
	#return out_fmap
	return voxel



def affine_grid_generator(num_batch,joint,height, width,depth, theta):
	#num_batch = tf.shape(theta)[0]  # that's B
	# create normalized 3D grid
	x = tf.linspace(-1.0, 1.0, width)
	y = tf.linspace(-1.0, 1.0, height)
	z = tf.linspace(-1.0, 1.0, depth)
	x_t, y_t , z_t = tf.meshgrid(x, y,z)
	
	# flatten
	x_t_flat = tf.reshape(x_t, [-1])
	y_t_flat = tf.reshape(y_t, [-1])
	z_t_flat = tf.reshape(z_t, [-1])
	
	# reshape to (x_t, y_t,z_t_flat , 1)   ************
	ones = tf.ones_like(x_t_flat)
	sampling_grid = tf.stack([x_t_flat, y_t_flat, z_t_flat, ones])
	
	print(sampling_grid.get_shape(),'preeeeeeeeeeeeeeeeeeeeeeeee')
	# repeat grid num_batch times
	sampling_grid = tf.expand_dims(sampling_grid, axis=0)
	sampling_grid = tf.expand_dims(sampling_grid, axis=0)
	print(sampling_grid.get_shape(),'preeeeeeeeeeeeeeeeeeeeeeeee')
	sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch,joint, 1, 1]))
	print(sampling_grid.get_shape(),'prossssssssssssssssssssssssssssssss')

	# cast to float32 (required for matmul)
	theta = tf.cast(theta, 'float32')
	sampling_grid = tf.cast(sampling_grid, 'float32')

	# transform the sampling grid - batch multiply
	batch_grids = tf.matmul(theta, sampling_grid)
	# batch grid has shape (num_batch, 2, H*W*D)
	
	# reshape to (num_batch, H, W,D, 2)
	batch_grids = tf.reshape(batch_grids, [num_batch,joint, 3, height, width,depth])
	# batch_grids = tf.transpose(batch_grids, [0, 2, 1, 3])

	return batch_grids

def kernal(x):
	return tf.maximum(tf.cast(0,'float32'),tf.cast((1-x),'float32'))

def trilinear_sampler(img, grid, voxel):
	# prepare useful params
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
									x=tf.cast(tf.gather_nd(grid,[i,j,0,m,n,l]),'float32')
									y=tf.cast(tf.gather_nd(grid,[i,j,1,m,n,l]),'float32')
									z=tf.cast(tf.gather_nd(grid,[i,j,0,m,n,l]),'float32')
									#z=0
									H0=tf.cast(H,'float32')
									W0=tf.cast(W,'float32')
									D0=tf.cast(D,'float32')
									v=tf.cast(tf.gather_nd(voxel,[i,j,a,b,c]),'float32')
									k=v * kernal((a-x)/H) * kernal((b-y)/W) * kernal((c-z)/D)
									#print(sum1)	
									sum1=sum1+k
									out=out+k
					Two_D[j][m][n]=out
					print(out)
	return Two_D
					

	# rescale x and y to [0, W/H]
	'''x = 0.5 * ((x + 1.0) * tf.cast(W, 'float32'))
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

'''def get_voxel(cloud,dimx,dimy,dimz):
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
        voxel=tf.Variable(tf.zeros( [10,15,100,100,100]))
        for i in range(B):
                for j in range(J):
                        a=tf.cast(tf.gather_nd(x,[i,j]),'int32')
                        b=tf.cast(tf.gather_nd(y,[i,j]),'int32')
                        c=tf.cast(tf.gather_nd(z,[i,j]),'int32')
                        #I=tf.constant(i)
                        #J0=tf.constant(j)
                        I=tf.cast(j*tf.ones((25600)),'int32')
                        J0=tf.cast(j*tf.ones((25600)),'int32')
                        a=tf.clip_by_value(a,0,99)
                        b=tf.clip_by_value(b,0,99)
                        c=tf.clip_by_value(c,0,99)
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
                                        voxel=tf.Variable(voxel)
                                        k=tf.clip_by_value(240*(x1+m)+(y1+n),0,76799)
                                        point0=tf.Variable(tf.gather_nd(cloud,[b,k]))
                                        #point1=tf.Variable(tf.gather_nd(cloud,[b,k,1]))
                                        #point2=tf.Variable(tf.gather_nd(cloud,[b,k,3]))
                                        voxel=voxel[b,j,k].assign(point0)
                                        #voxel=tf.Variable(voxel[b,j,k,1]).assign(point1)
                                        #voxel=tf.Variable(voxel[b,j,k,2]).assign(point2)
                                        '''point=tf.expand_dims(point, axis=0)  #(1,3)  
                                        part=tf.concat([part,point], 0)                                   
                        part=tf.expand_dims(part, axis=0)
                        batch=tf.concat([batch,part], 0)
                batch=tf.expand_dims(batch, axis=0)
                voxel=tf.concat([voxel,batch], 0)'''
        return voxel	
