import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

train_x = mnist.train.images
train_y = mnist.train.labels

print(train_x.shape, train_y.shape)

total_epochs = 450
batch_size = 100
learning_rate = 0.0002
random_size = 784

def generator( z , reuse = False ):
    l = [random_size, 512, 512, 784]
    if reuse==False:
        with tf.variable_scope(name_or_scope = "Gen") as scope:
            gw1 = tf.get_variable(name = "w1", shape = [l[0], l[1]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gb1 = tf.get_variable(name = "b1", shape = [l[1]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gw2 = tf.get_variable(name = "w2", shape = [l[1], l[2]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gb2 = tf.get_variable(name = "b2", shape = [l[2]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gw3 = tf.get_variable(name = "w3", shape = [l[2], l[3]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gb3 = tf.get_variable(name = "b3", shape = [l[3]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))

    else:
        with tf.variable_scope(name_or_scope="Gen", reuse = True) as scope:
            gw1 = tf.get_variable(name = "w1", shape = [l[0], l[1]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gb1 = tf.get_variable(name = "b1", shape = [l[1]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gw2 = tf.get_variable(name = "w2", shape = [l[1], l[2]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gb2 = tf.get_variable(name = "b2", shape = [l[2]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gw3 = tf.get_variable(name = "w3", shape = [l[2], l[3]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))
            gb3 = tf.get_variable(name = "b3", shape = [l[3]], initializer = tf.random_normal_initializer(mean=0.0, stddev = 0.01))


    hidden1 = tf.nn.relu(tf.matmul(z , gw1) + gb1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, gw2) + gb2)
    output = tf.nn.sigmoid(tf.matmul(hidden2, gw3) + gb3)
    return output


def discriminator( x , reuse = False):
    if(reuse == False):
        with tf.variable_scope(name_or_scope="Dis") as scope:
            dw1 = tf.get_variable(name = "w1", shape = [784, 256], initializer = tf.random_normal_initializer(0.0, 0.01))
            db1 = tf.get_variable(name = "b1", shape = [256], initializer = tf.random_normal_initializer(0.0, 0.01))
            dw2 = tf.get_variable(name = "w2", shape = [256, 256], initializer = tf.random_normal_initializer(0.0, 0.01))
            db2 = tf.get_variable(name = "b2",  shape = [256], initializer = tf.random_normal_initializer(0.0, 0.01))
            dw3 = tf.get_variable(name = "w3", shape = [256, 1], initializer = tf.random_normal_initializer(0.0, 0.01))
            db3 = tf.get_variable(name = "b3",  shape = [1], initializer = tf.random_normal_initializer(0.0, 0.01))

    else:
        with tf.variable_scope(name_or_scope="Dis", reuse = True) as scope:
            dw1 = tf.get_variable(name = "w1",  shape = [784, 256], initializer = tf.random_normal_initializer(0.0, 0.01))
            db1 = tf.get_variable(name = "b1", shape = [256], initializer = tf.random_normal_initializer(0.0, 0.01))
            dw2 = tf.get_variable(name = "w2", shape = [256, 256], initializer = tf.random_normal_initializer(0.0, 0.01))
            db2 = tf.get_variable(name = "b2", shape = [256],  initializer = tf.random_normal_initializer(0.0, 0.01))
            dw3 = tf.get_variable(name = "w3", shape = [256, 1], initializer = tf.random_normal_initializer(0.0, 0.01))
            db3 = tf.get_variable(name = "b3", shape = [1],  initializer = tf.random_normal_initializer(0.0, 0.01))


    hidden1 = tf.nn.relu(tf.matmul(x , dw1) + db1)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, dw2) + db2)
    output = tf.nn.sigmoid(tf.matmul(hidden2, dw3)  + db3)
    return output

def random_noise(batch_size):
    return np.random.normal(size=[batch_size , random_size])




g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, 784])
    Z = tf.placeholder(tf.float32, [None, random_size])


    fake_x = generator(Z)
    result_of_fake = discriminator(fake_x)
    result_of_real = discriminator(X , True)


    g_loss = tf.reduce_mean( tf.log(result_of_fake))
    d_loss = tf.reduce_mean( tf.log(result_of_real) + tf.log(1 - result_of_fake))


    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if "Gen" in var.name]
    d_vars = [var for var in t_vars if "Dis" in var.name]
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gvs = optimizer.compute_gradients(-g_loss, var_list = g_vars)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    g_train = optimizer.apply_gradients(capped_gvs) #minimize(-g_loss, var_list= g_vars)
    d_train = optimizer.minimize(-d_loss, var_list = d_vars)




with tf.Session(graph = g) as sess:
    sess.run(tf.global_variables_initializer())
    total_batchs = int(train_x.shape[0] / batch_size)

    for epoch in range(total_epochs):
        for batch in range(total_batchs):
            batch_x = train_x[batch * batch_size : (batch+1) * batch_size]
            batch_y = train_y[batch * batch_size : (batch+1) * batch_size]
            noise = random_noise(batch_size)

            sess.run(g_train , feed_dict = {Z : noise})
            sess.run(d_train, feed_dict = {X : batch_x , Z : noise})

            gl, dl = sess.run([g_loss, d_loss], feed_dict = {X : batch_x , Z : noise})

        print("======= Epoch : ", epoch , " =======")
        print("generator: " , -gl )
        print("discriminator: " , -dl )


        samples = 20
        if epoch == 0 or (epoch + 1) % 10 == 0:
            sample_noise = random_noise(samples)
            generated = sess.run(fake_x , feed_dict = { Z : sample_noise})
            f=open('epoch'+str(epoch)+'.txt','w')
            for i in range(samples):
                cnt = 0
                for j in generated[i]:
                    f.write("%0.2f " % j)
                    cnt+=1
                    if cnt % 28 == 0:
                        f.write('\n')
                f.write('\n')

            f.close()
