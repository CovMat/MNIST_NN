from read_file import read_file
import numpy as np
import tensorflow as tf

nb_classes = 10
nb_nerons = 800

# read the training data set:
train_label_num, _, _, train_label_data = read_file( "data/train-labels-idx1-ubyte" )
train_image_num, row_num, col_num, train_image_data = read_file( "data/train-images-idx3-ubyte" )
# read the testing data set
test_label_num, _, _, test_label_data = read_file( "data/t10k-labels-idx1-ubyte" )
test_image_num, _, _, test_image_data = read_file( "data/t10k-images-idx3-ubyte" )
# transfer label and image data to proper format
X_train = np.asarray( train_image_data ).reshape( train_image_num, row_num*col_num  )
Y_train = np.zeros( (train_label_num, nb_classes) )
for i in range(nb_classes):
    Y_train[ : , i ] = ( np.asarray(train_label_data) == i )
X_test = np.asarray( test_image_data ).reshape( test_image_num, row_num*col_num  )

# Build computation Graph of Neural Network
X_input = tf.placeholder( tf.float32, shape = ( None, row_num*col_num ) )
Y_output= tf.placeholder( tf.float32, shape = ( None, nb_classes ) )
# Hidden Layer 1
W1 = tf.Variable( tf.random_normal([ row_num*col_num, nb_nerons  ]), name="W1" )
b1 = tf.Variable( tf.zeros([nb_nerons], tf.float32), name="b1" )
Z1 = tf.matmul( X_input, W1 ) + b1
A1 = tf.nn.relu( Z1 )
# Hidden Layer 2
W2 = tf.Variable( tf.random_normal([ nb_nerons, nb_nerons ]), name="W2" )
b2 = tf.Variable( tf.zeros([nb_nerons], tf.float32), name="b2" )
Z2 = tf.matmul( A1, W2 ) + b2
A2 = tf.nn.relu( Z2 )
# Output Layer
Wo = tf.Variable( tf.random_normal([ nb_nerons, nb_classes ]), name = "W_output" )
bo = tf.Variable( tf.zeros([nb_classes], tf.float32), name="b_output" )
Zo = tf.matmul( A2, Wo ) + bo
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2( labels = Y_output, logits = Zo )
cost = tf.reduce_mean( cross_entropy )
# for TensorBoard
cost_summ = tf.summary.scalar( "Cost Function", cost )
#
optimizer = tf.train.AdamOptimizer( learning_rate = 0.0003 ).minimize( cost )

# run (mini-batch)
summary = tf.summary.merge_all()
sess = tf.Session()
sess.run( tf.global_variables_initializer() )
writer = tf.summary.FileWriter("./logs/learning_rate_0.0003_nerons_800")
writer.add_graph( sess.graph )
#
batch_size = 512
batch_num = train_label_num // batch_size
#
global_step = 0
for epoch in range(20):
    for i in range(batch_num):
        st = i*batch_size
        ed = (i+1)*batch_size
        if ( i == batch_num-1 ):
            ed = train_label_num
        s, _ = sess.run( [ summary, optimizer], feed_dict={\
                                        X_input:X_train[ st:ed, :],\
                                        Y_output:Y_train[ st:ed, :]} )
        writer.add_summary( s, global_step = global_step )
        global_step += 1

# Accuracy
logit_train = sess.run( Zo, feed_dict={X_input:X_train} )
max_index = sess.run( tf.argmax(logit_train, axis=1) )
accuracy = sess.run( tf.reduce_mean(tf.cast( max_index == train_label_data, dtype=tf.float32) ) )
print( "Accuracy of Training set:"+ str(accuracy) )
logit_test = sess.run( Zo, feed_dict={X_input:X_test} )
max_index = sess.run( tf.argmax(logit_test, axis=1) )
accuracy = sess.run( tf.reduce_mean(tf.cast( max_index == test_label_data, dtype=tf.float32) ) )
print( "Accuracy of Testing set:"+ str(accuracy) )
   
