from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import h5py


tf.logging.set_verbosity(tf.logging.INFO)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def wsp_model(features, labels, mode):
  #TODO create patches for input layer
  patches=tf.extract_image_patches(
    features["x"],
    ksizes = [1, 99, 99, 1],
    strides = [1, 21, 61, 1],
    rates = [1, 1, 1, 1],
    padding="VALID",
    name=None
)

  input_layer = patches

  #conv1
  conv1 = tf.layers.conv2d(
      name="conv1",
      inputs=input_layer,
      filters=64,
      kernel_size=5,
      padding="same",
      strides=2,
      trainable=True,
      activation=tf.nn.relu)

  #pool1
  pool1 = tf.layers.max_pooling2d(name="pool1",inputs=conv1, pool_size=2, strides=2)

  #local response normalization
  norm1 = tf.nn.local_response_normalization(
      input = pool1,
      depth_radius=5,
      bias=1,
      alpha=0.0001,
      beta=0.75)

  #padding 
  paddings = tf.constant([[1, 1,], [1, 1],[1,1],[1,1]])
  norm1 = tf.pad(norm1,paddings,"CONSTANT")


    #conv2
  conv2 = tf.layers.conv2d(
      name="conv2",
      inputs=norm1,
      filters=128,
      kernel_size=3,
      trainable=True,
      padding="same",
      activation=tf.nn.relu)

    #pool2
  pool2 = tf.layers.max_pooling2d(name="pool2",inputs=conv2, pool_size=2, strides=2)

  #padding 
  paddings = tf.constant([[1, 1,], [1, 1],[1,1],[1,1]])
  pool2 = tf.pad(pool2,paddings,"CONSTANT")


  #conv3
  conv3 = tf.layers.conv2d(
      name="conv3",
      inputs=pool2,
      filters=256,
      kernel_size=3,
      padding="same",
      trainable=True,
      strides=1,
      activation=tf.nn.relu)

    #fc4
  fc4 = tf.layers.dense(name="fc4",inputs=conv3, units=512, activation=tf.nn.relu,trainable=True)

    #dropout
  dropout = tf.layers.dropout(
      inputs=fc4, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)


  #fc5
  fc5 = tf.layers.dense(name="fc5",inputs=dropout, units=19,trainable=True) #a unit for each class

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=fc5, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(fc5, name="softmax_tensor")
    }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=fc5)

    # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)





def d_cnn_model(features, labels, mode):
  #TODO fix input layers
  input_layer= features

  #conv1
  conv1 = tf.layers.conv2d(
    name="conv1", 
      inputs=input_layer,
      filters=64,
      kernel_size=5,
      padding="same",
      strides=2,
      activation=tf.nn.relu)

  #pool1
  pool1 = tf.layers.max_pooling2d(name="pool1",inputs=conv1, pool_size=2, strides=2)

  #local response normalization
  norm1 = tf.nn.local_response_normalization(
      input = pool1,
      depth_radius=5,
      bias=1,
      alpha=0.0001,
      beta=0.75)

  #padding 
  paddings = tf.constant([[1, 1,], [1, 1]])
  norm1 = tf.pad(norm1,paddings,"CONSTANT")


    #conv2
  conv2 = tf.layers.conv2d(
      name="conv2",
      inputs=norm1,
      filters=128,
      kernel_size=3,
      padding="same",
      activation=tf.nn.relu)

    #pool2
  pool2 = tf.layers.max_pooling2d(name="pool2",inputs=conv2, pool_size=2, strides=2)

  #padding 
  paddings = tf.constant([[1, 1,], [1, 1]])
  pool2 = tf.pad(pool2,paddings,"CONSTANT")


  #conv3
  conv3 = tf.layers.conv2d(
      name="conv3",
      inputs=pool2,
      filters=256,
      kernel_size=3,
      padding="same",
      strides=1,
      activation=tf.nn.relu)

    #fc4
  fc4 = tf.layers.dense(name="fc4",inputs=conv3, units=512, activation=tf.nn.relu)

  #dropout
  dropout = tf.layers.dropout(
      inputs=fc4, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  tf.train.init_from_checkpoint("/tmp/wsp_model",{'conv1':'conv1','conv2':'conv2','conv3':'conv3','pool1':'pool1','pool2':'pool2'})

  #pool5_spm3
  pool5_spm3 = tf.layers.max_pooling2d(inputs=fc4, pool_size=10, strides=10)

  #pool5_spm3_flatten
  pool5_spm3_flatten = tf.layers.flatten(pool5_spm3)

  #pool5_spm2
  pool5_spm2 = tf.layers.max_pooling2d(inputs=fc4, pool_size=14, strides=14)

  #pool5_spm2_flatten
  pool5_spm2_flatten = tf.layers.flatten(pool5_spm2)

  #pool5_spm1
  pool5_spm1 = tf.layers.max_pooling2d(inputs=fc4, pool_size=29, strides=29)

  #pool5_spm1_flatten
  pool5_spm1_flatten = tf.layers.flatten(pool5_spm2)

  #pool5_spm
  pool5_spm = tf.concat([pool5_spm1_flatten, pool5_spm2_flatten, pool5_spm3_flatten],1)

  # Inner Product
  fc4_2 = tf.layers.fully_connected(
    input=pool5_spm,
    num_outputs=512,
    activation_fn=tf.nn.relu,
    weight_initializer = tf.truncated_normal_initializer(stddev=0.1),
    biases_initializer= tf.constant_initializer(value=0.1),
    )

  #dropout
  dropout = tf.layers.dropout(
      inputs=fc4, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Inner Product
  fc5 = tf.layers.fully_connected(
    input=fc4_2,
    num_outputs=19,
    activation_fn=tf.nn.relu,
    weight_initializer = tf.truncated_normal_initializer(stddev=0.1),
    biases_initializer= tf.constant_initializer(value=0),
    )

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=fc5, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(fc5, name="softmax_tensor")
    }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=fc5)

    # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)





def main(unused_argv):
  # Load training and eval data
  nbImagesTrain = 1200
  with h5py.File('nyu_depth_v2_labeled.mat', 'r') as file:
    train_data_images=np.transpose(np.array(file[('images')])[:nbImagesTrain,:,:,:], (0, 3, 2, 1)).astype(np.float16)
    train_data_depths=np.transpose(np.array(file[('rawDepths')])[:nbImagesTrain,:,:], (0, 2, 1)).astype(np.float16)
    eval_data_images=np.transpose(np.array(file[('images')])[nbImagesTrain+1:,:,:,:], (0, 3, 2, 1)).astype(np.float16)
    eval_data_depths=np.transpose(np.array(file[('rawDepths')])[nbImagesTrain+1:,:,:], (0, 2, 1)).astype(np.float16)
    train_labels=np.transpose(np.array(file['labels'])[:nbImagesTrain], (0, 2, 1)).astype(np.int32)
    eval_labels=np.transpose(np.array(file['labels'])[nbImagesTrain+1:], (0, 2, 1)).astype(np.int32)
    file.close()


  print('beginning learning')
  wsp_model_estimator = tf.estimator.Estimator(
  model_fn=wsp_model, model_dir="/tmp/wsp_model")
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data_images},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)





  wsp_model_estimator.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data_images},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  eval_results = wsp_model_estimator.evaluate(input_fn=eval_input_fn)
  #print(eval_results)

  dcnn_model = tf.estimator.Estimator(
  model_fn=d_cnn_model, model_dir="/tmp/d_cnn_model")
  # Set up logging for predictions
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
  tensors=tensors_to_log, every_n_iter=50)
  train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": [train_data_images,train_data_depths]},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True)





  dcnn_model.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook])
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)
  eval_results = dcnn_model.evaluate(input_fn=eval_input_fn)




if __name__ == "__main__":
  tf.app.run()
