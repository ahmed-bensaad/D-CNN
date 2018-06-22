import tensorflow as tf
import numpy as np
import threading
import PIL.Image as Image
from functools import partial
from multiprocessing import Pool
import cv2
import sys
import os

HEIGHT=192
WIDTH=256
NUM_PLANES = 20

PIXEL_UNIT=0.2
DEPTH_UNIT=5
def importdata(n_img):
	FILE='planes_scannet_val.tfrecords'
	VAL_DATA="train_data/"

	filename_queue = tf.train.string_input_producer([FILE], num_epochs=1)
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)


	features = tf.parse_single_example(
	            serialized_example,
				
	            features={
	                'image_raw': tf.FixedLenFeature([], tf.string),
	                'image_path': tf.FixedLenFeature([], tf.string),
	                'num_planes': tf.FixedLenFeature([], tf.int64),
	                'plane': tf.FixedLenFeature([NUM_PLANES * 3], tf.float32),
	                'segmentation_raw': tf.FixedLenFeature([], tf.string),
	                'depth': tf.FixedLenFeature([HEIGHT * WIDTH], tf.float32),
	                'normal': tf.FixedLenFeature([HEIGHT * WIDTH * 3], tf.float32),
	                'semantics_raw': tf.FixedLenFeature([], tf.string),                
	                'boundary_raw': tf.FixedLenFeature([], tf.string),
	                'info': tf.FixedLenFeature([4 * 4 + 4], tf.float32),                
	            })

	#image
	image = tf.decode_raw(features['image_raw'], tf.uint8)
	image = tf.cast(image, tf.float32) * (1. / 255) 
	image = tf.reshape(image, [HEIGHT, WIDTH, 3])


	image_p= features['image_path']


	depth = features['depth']
	depth = tf.reshape(depth, [HEIGHT, WIDTH, 1])


	normal = features['normal']
	normal = tf.reshape(normal, [HEIGHT, WIDTH, 3])
	normal = tf.nn.l2_normalize(normal, dim=2)


	semantics = tf.decode_raw(features['semantics_raw'], tf.uint8)
	semantics = tf.cast(tf.reshape(semantics, [HEIGHT, WIDTH]), tf.int32)


	numPlanes = tf.minimum(tf.cast(features['num_planes'], tf.int32), 20)
	numPlanesOri = numPlanes
	numPlanes = tf.maximum(numPlanes, 1)


	planes = features['plane']
	planes = tf.reshape(planes, [NUM_PLANES, 3])
	planes = tf.slice(planes, [0, 0], [numPlanes, 3])


	shuffle_inds = tf.one_hot(tf.range(numPlanes), numPlanes)


	planes = tf.transpose(tf.matmul(tf.transpose(planes), shuffle_inds))
	planes = tf.reshape(planes, [numPlanes, 3])
	planes = tf.concat([planes, tf.zeros([20 - numPlanes, 3])], axis=0)
	planes = tf.reshape(planes, [20, 3])


	boundary = tf.decode_raw(features['boundary_raw'], tf.uint8)
	boundary = tf.cast(tf.reshape(boundary, (HEIGHT, WIDTH, 2)), tf.float32)


	segmentation = tf.decode_raw(features['segmentation_raw'], tf.uint8)
	segmentation = tf.reshape(segmentation, [HEIGHT, WIDTH, 1])


	coef = tf.range(numPlanes)
	coef = tf.reshape(tf.matmul(tf.reshape(coef, [-1, numPlanes]), tf.cast(shuffle_inds, tf.int32)), [1, 1, numPlanes])
	plane_masks = tf.cast(tf.equal(segmentation, tf.cast(coef, tf.uint8)), tf.float32)
	plane_masks = tf.concat([plane_masks, tf.zeros([HEIGHT, WIDTH, 20 - numPlanes])], axis=2)
	plane_masks = tf.reshape(plane_masks, [HEIGHT, WIDTH, 20])
	non_plane_mask = 1 - tf.reduce_max(plane_masks, axis=2, keep_dims=True)






	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	train_data_rgb=[]
	train_data_depth=[]
	train_labels=[]

	try:
		ii=1
		while ii<=n_img :  
			print(ii)
			ii=ii+1
			r_image, r_depth, r_image_path , r_numPlanes, r_numPlanesOri, r_planes,r_segmentation ,r_plane_masks,r_non_plane_mask= sess.run([image,depth,image_p,numPlanes,numPlanesOri,planes,segmentation,plane_masks,non_plane_mask])
			train_data_rgb.append(r_image)
			train_data_depth.append(r_depth)
			train_labels.append(r_plane_masks)
			
	except tf.errors.OutOfRangeError :
		print("done.")
		
	finally:
		print("stop.")
		coord.request_stop()

	coord.join(threads)
	sess.close()
	return(train_data_rgb,train_data_depth,train_labels)