######################################################
# CIFAR-10 example
# Mark Harvey
# Dec 2018
######################################################
import os
import sys
import shutil
import tensorflow as tf


#####################################################
# Housekeeping
#####################################################
print("Tensorflow version. ", tf.VERSION)
print("Keras version. ", tf.keras.__version__)


#####################################################
# Set up directories
#####################################################
# Returns the directory the current script (or interpreter) is running in
def get_script_directory():
    path = os.path.realpath(sys.argv[0])
    if os.path.isdir(path):
        return path
    else:
        return os.path.dirname(path)


SCRIPT_DIR = get_script_directory()
print('This script is located in: ', SCRIPT_DIR)

GRAPH_FILE = 'graph.pbtxt'
CHKPT_FILE = 'float_model.ckpt'
CHKPT_DIR = 'checkpoints'

TB_LOG_DIR = os.path.join(SCRIPT_DIR, 'tb_logs')
MODEL_DIR = os.path.join(SCRIPT_DIR, 'model')
CHKPT_DIR = os.path.join(SCRIPT_DIR, CHKPT_DIR)
CHKPT_PATH = os.path.join(CHKPT_DIR, CHKPT_FILE)


# create a directory for the TensorBoard data if it doesn't already exist
# delete it and recreate if it already exists
if (os.path.exists(TB_LOG_DIR)):
    shutil.rmtree(TB_LOG_DIR)
os.makedirs(TB_LOG_DIR)
print("Directory " , TB_LOG_DIR ,  "created ") 


# create a directory for the checkpoints if it doesn't already exist
# delete it and recreate if it already exists
if (os.path.exists(CHKPT_DIR)):
    shutil.rmtree(CHKPT_DIR)
os.makedirs(CHKPT_DIR)
print("Directory " , CHKPT_DIR ,  "created ") 

#####################################################
# Hyperparameters
#####################################################
LEARNRATE = 0.0001
EPOCHS = 500
BATCHSIZE = 100



#####################################################
# Dataset preparation
#####################################################
# CIFAR10 datset has 60k images. Training set is 50k, test set is 10k.
# Each image is 32x32 pixels RGB
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Scale image data from range 0:255 to range 0:1
x_train = x_train / 255.0
x_test = x_test / 255.0

# one-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


# calculate total number of batches
total_batches = int(len(x_train)/BATCHSIZE)


#####################################################
# Create the Computational graph
#####################################################

# define placeholders for the input data & labels
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='images_in')
y = tf.placeholder(tf.float32, [None, 10], name='labels_in')


# define out layers of our CNN
def cnn(x):
  conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=3, kernel_initializer=tf.glorot_uniform_initializer(), activation=tf.nn.relu, name='conv1')
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2, name='pool1')
  conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=3, kernel_initializer=tf.glorot_uniform_initializer(), activation=tf.nn.relu, name='conv2')
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2, name='pool2')
  conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=3, kernel_initializer=tf.glorot_uniform_initializer(), activation=tf.nn.relu, name='conv3')
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, name='pool3')
  flat1 = tf.layers.flatten(inputs=pool3,name='flat1')
  fc1 = tf.layers.dense(inputs=flat1, units=1024, kernel_initializer=tf.glorot_uniform_initializer(), activation=tf.nn.relu, name='fc1')
  prediction = tf.layers.dense(inputs=fc1, units=10, kernel_initializer=tf.glorot_uniform_initializer(), activation=tf.nn.softmax, name='prediction')

  return prediction

# build the network, input comes from the 'x' placeholder
prediction = cnn(x)

# Define a cross entropy loss function
loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=prediction, onehot_labels=y))

# Define the optimizer function
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNRATE).minimize(loss)

# Check to see if predictions match the labels
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

 # Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorBoard data collection
tf.summary.scalar('cross_entropy_loss', loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.image('input_images', x)


# set up saver object
saver = tf.train.Saver()



#####################################################
# Run the graph in a Session
#####################################################
# Launch the graph
with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())
    
    # TensorBoard writer
    writer = tf.summary.FileWriter(TB_LOG_DIR, sess.graph)
    tb_summary = tf.summary.merge_all()

    # Training cycle with training data
    for epoch in range(EPOCHS):
        print ("Epoch:", epoch)

        # process all batches
        for i in range(total_batches):
            
            # fetch a batch from training dataset
            batch_x, batch_y = x_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE], y_train[i*BATCHSIZE:i*BATCHSIZE+BATCHSIZE]

            # Run graph for optimization, loss, accuracy - i.e. do the training
            _, acc, s = sess.run([optimizer, accuracy, tb_summary], feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(s, (epoch*total_batches + i))
            # Display accuracy per 100 batches
            if i % 100 == 0:
              print (" Batch:", i, 'Training accuracy: ', acc)

    print("Training Finished!")
    writer.flush()
    writer.close()

    # Evaluation cycle with test data
    print ("Final Accuracy with test set:", sess.run(accuracy, feed_dict={x: x_test, y: y_test}))

    # save checkpoint & graph file as protobuf text
    save_path = saver.save(sess, CHKPT_PATH)
    tf.train.write_graph(sess.graph_def, CHKPT_DIR, GRAPH_FILE)

print('FINISHED!')
print('Run `tensorboard --logdir=%s --port 6006 --host localhost` to see the results.' % TB_LOG_DIR)

