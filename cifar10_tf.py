'''
TO DO:
 - comment fully
 - training loop is a mess!
 - plot/print info re CNN
 - data augmentation with flips, rotates, color changes
 - vars for batch size, epochs  - flags for cmd line?
 - select between GPU/CPU?
 - add Tensorboard visualisation
 - saving/freezing
 - use tf.data pipeline & Estimator
'''

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def dataset_prep():
  '''
  Load cifar10 dataset into memory
  Dataset of 60,000 32x32 colour images labelled into 10 categories/classes.
  50,000 images for training, 10,000 images for test.
  Test and training images are normalized.
  Args:
    None
  Returns:
    cifar10_labels            : list of category/class names
    train_images, train_labels: tuple of numpy ndarray - normalized images
    test_images, test_labels  : tuple of numpy ndarray - normalized images
  '''
  # use keras to fetch cifar10 dataset
  (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
  
  
  # min/max normalization of test & training set images
  min_val = np.min(train_images)
  max_val = np.max(train_images)
  train_images = (train_images-min_val) / (max_val-min_val)
  min_val = np.min(test_images)
  max_val = np.max(test_images)
  test_images = (test_images-min_val) / (max_val-min_val)

  #
  train_labels = tf.keras.utils.to_categorical(train_labels)
  test_labels = tf.keras.utils.to_categorical(test_labels)
  
  
  # Print some information about the datasets
  print("*****************************************************")
  print("Training set images shape: {shape}".format(shape=train_images.shape))
  print("Training set labels shape: {shape}".format(shape=train_labels.shape))
  print("Test set images shape    : {shape}".format(shape=test_images.shape))
  print("Test set labels shape    : {shape}".format(shape=test_labels.shape))
  print("*****************************************************")
  
  # create a list of categories (class labels)
  cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  
  # plot an image - should be a truck
  plt.imshow(train_images[1])
  print("Training set label 1: {tlabel}".format(tlabel=train_labels[1]))
  print("Training set label type: {tlabeltype}".format(tlabeltype=type(train_labels[1])))

  return cifar10_labels, (train_images, train_labels), (test_images, test_labels)




def build_model(x, keep_prob):
    '''
    Builds the convolutional neural network model layers
    Args:
      x
      keep_prob
    Returns:
      out
    '''
    
    print("CNN input shape: {shape}".format(shape=x.shape))
    
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
    conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

    # 1, 2
    conv1 = tf.nn.conv2d(input=x, filter=conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv3_bn = tf.layers.batch_normalization(conv3_pool)

    # 7, 8
    conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv4_bn = tf.layers.batch_normalization(conv4_pool)

    # 9
    flat = tf.contrib.layers.flatten(conv4_bn)

    # 10
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # 11
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    # 12
    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob)
    full3 = tf.layers.batch_normalization(full3)

    # 13
    full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    full4 = tf.nn.dropout(full4, keep_prob)
    full4 = tf.layers.batch_normalization(full4)

    # 14
    out = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=10, activation_fn=None)
    
    print("CNN output shape: {shape}".format(shape=out.shape))
                
    return out



def do_training(session, optimizer, keep_probability, feature_batch, label_batch):
    session.run(optimizer,
                feed_dict={
                    image_placeholder: feature_batch,
                    label_placeholder: label_batch,
                    keep_prob: keep_probability
                })

    
def main():
    
  # prepare the dataset
  cifar10_labels, (train_images, train_labels), (test_images, test_labels) = dataset_prep()

  # Hyper parameters
  epochs = 1
  batch_size = 100
  keep_probability = 0.7
  learning_rate = 0.001
  
  # create placeholders for feeding datasets into model
  image_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
  label_placeholder = tf.placeholder(tf.float32, shape=[None, 10])
  keep_prob = tf.placeholder(tf.float32)
  print("keep_prob shape: {shape}".format(shape=keep_prob.shape))
  
  # Build model
  cnn_out = build_model(image_placeholder, keep_prob)

  # Loss and Optimizer
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_placeholder, logits=cnn_out))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

  # Accuracy
  correct_pred = tf.equal(tf.argmax(cnn_out, 1), tf.argmax(label_placeholder, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
  
  
  num_batches = int(50000 / batch_size)
  print("Number of batches: {nb}".format(nb=num_batches))  
 
  last_batch_size = 50000 % batch_size
  print("Last batch size: {nb}".format(nb=last_batch_size))  
  

  with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())



  # Training cycle
    for epoch in range(epochs): # 0 to epochs-1
#      # Loop over all batches
#      n_batches = 5
      for i in range(num_batches):  # 0 to 499
         batch_features = train_images[i*batch_size: ((i+1)*batch_size)]
 #        print("batch_features shape: {shape}".format(shape=batch_features.shape))
         batch_labels = train_labels[i*batch_size: ((i+1)*batch_size)]
#         print("batch_labels shape: {shape}".format(shape=batch_labels.shape))
         
         sess.run(optimizer,
                  feed_dict={
                    image_placeholder: batch_features,
                    label_placeholder: batch_labels,
                    keep_prob: keep_probability
                })

#         do_training(sess, optimizer, keep_probability, batch_features, batch_labels)
         print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, i), end='\n')
#        print_stats(sess, batch_features, batch_labels, cost, accuracy)
        
    print("Accuracy: {acc}".format(acc=accuracy))
    print("Correct Predictions: {cp}".format(cp=correct_pred))
    
    

if __name__ == "__main__":
    main()
    
    
    
