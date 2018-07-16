import tensorflow as tf
import numpy as np
import sys

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

target_class = 0
target_class = int(sys.argv[1])
num_of_class = 10
train_x = mnist.train.images
train_y = mnist.train.labels

sorted_index = train_y.argsort()
train_x = train_x[sorted_index]
train_y = train_y[sorted_index]

cnt_class = []
for i in range(num_of_class):
    cnt_class.append(0)
for i in range(len(train_y)):
    cnt_class[train_y[i]] += 1

train_x_per_class = []
for i in range(num_of_class):
    train_x_per_class.append([])
for i in range(train_x.shape[0]):
    train_x_per_class[train_y[i]].append(train_x[i])
for i in range(num_of_class):
    train_x_per_class[i] = np.asarray(train_x_per_class[i])
    print(train_x_per_class[i].shape)
train_x_per_class = np.asarray(train_x_per_class)

print(train_x.shape, train_y.shape)
print(train_x_per_class.shape)

total_epochs = 70
batch_size =  100 #275 #should be divisor of 5500 (== 55000/10)
learning_rate = 0.00022
random_size = 16

init = tf.random_normal_initializer(mean=0.0, stddev = 0.01)

def generator(z, reuse = False):
    l = [random_size, 16, 784]

    with tf.variable_scope(name_or_scope = "Gen") as scope:
        gw1 = tf.get_variable(name = "w1", shape = [l[0], l[1]], initializer = init)
        gb1 = tf.get_variable(name = "b1", shape = [l[1]], initializer = init)
        gw2 = tf.get_variable(name = "w2", shape = [l[1], l[2]], initializer = init)
        gb2 = tf.get_variable(name = "b2", shape = [l[2]], initializer = init)


    hidden1 = tf.nn.relu(tf.matmul(z , gw1) + gb1)
    output = tf.nn.sigmoid(tf.matmul(hidden1, gw2) + gb2)
    return output


def discriminator(x, reuse = False):
    l = [784, 256, 1]
    with tf.variable_scope(name_or_scope="Dis", reuse = reuse) as scope:
        dw1 = tf.get_variable(name = "w1",  shape = [l[0], l[1]], initializer = init)
        db1 = tf.get_variable(name = "b1", shape = [l[1]], initializer = init)
        dw2 = tf.get_variable(name = "w2", shape = [l[1], l[2]], initializer = init)
        db2 = tf.get_variable(name = "b2", shape = [l[2]],  initializer = init)


    hidden1 = tf.nn.relu(tf.matmul(x , dw1) + db1)
    output = tf.nn.sigmoid(tf.matmul(hidden1, dw2)  + db2)
    return output

def random_noise(batch_size):
    return np.random.normal(size=[batch_size , random_size])




g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, 784])
    Z = tf.placeholder(tf.float32, [None, random_size])

    fake_x = generator(Z)

    result_of_fake = discriminator(fake_x)
    result_of_real = discriminator(X, reuse=True)

    g_loss = tf.reduce_mean( tf.log(result_of_fake))
    d_loss = tf.reduce_mean( tf.log(result_of_real) + tf.log(1 - result_of_fake))

    t_vars = tf.trainable_variables()
    
    g_vars = [var for var in t_vars if "Gen" in var.name]
    d_vars = [var for var in t_vars if "Dis" in var.name]

    optimizer = tf.train.AdamOptimizer(learning_rate)
    gvs = optimizer.compute_gradients(-g_loss, var_list = g_vars)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]

    g_train = optimizer.apply_gradients(capped_gvs)
    d_train = optimizer.minimize(-d_loss, var_list = d_vars)



with tf.Session(graph = g) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    minimum_class_size = cnt_class[target_class]
    total_batchs = int(minimum_class_size / batch_size) 

    for epoch in range(total_epochs):
        for batch in range(total_batchs):
            batch_x = train_x_per_class[target_class][batch * batch_size : (batch+1) * batch_size]
            batch_y = np.asarray([target_class] * batch_size) 

            noise = random_noise(batch_size)
            sess.run(g_train, feed_dict = {Z : noise})
            sess.run(d_train, feed_dict = {X : batch_x , Z : noise})
      
        if epoch % 10 == 0:
            gl, dl = sess.run([g_loss, d_loss], feed_dict = {X : train_x , Z : random_noise(train_x.shape[0])})
            print("=======Epoch : ", epoch , " =======")
            print("generator: " , - gl )
            print("discriminator: " , - dl )

        """
        if epoch % 5 == 0:
            #f=open('re_epoch'+str(epoch)+'.txt','w')
            sample_noise = random_noise(1)
            generated = sess.run(fake_x , feed_dict = { Z : sample_noise})
            cnt = 0
            sample_out = ""
            for j in generated[0]:
                if j > 0.66:
                    sample_out += "@"
                elif j > 0.33:
                    sample_out += "0"
                else:
                    sample_out += " "
                #f.write("%0.2f " % j)
                cnt+=1
                if cnt % 28 == 0:
                    sample_out += "\n" #f.write('\n')
            sample_out += "\n" #f.write('\n')
            print(sample_out) #f.close()
        """
    
    np.set_printoptions(threshold=np.nan)
    w = sess.run(g_vars)
    f = open('./weight_small/'+str(target_class)+"_"+sys.argv[2]+'.txt','w')
    for i in range(len(w)):
        w[i] = np.asarray(w[i]) 
        f.write(np.array2string(w[i], max_line_width=np.nan).replace('[','').replace(']',''))
        f.write('\n\n')
    f.close()

