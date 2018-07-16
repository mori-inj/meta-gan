import tensorflow as tf
import numpy as np
import os

train_x = []
filepath = './weight_half/'
for filename in os.listdir(filepath):
    _, ext = os.path.splitext(filepath+filename)
    if ext != '.txt':
        continue
    f = open(filepath+filename, 'r')
    in_str = ""
    for line in f:
        in_str += line
    f.close()
    m = np.fromstring(in_str, dtype=float, sep=' ')
    train_x.append(m)
train_x = np.asarray(train_x)
print(train_x.shape)

total_epochs = 5000
batch_size = 100
learning_rate = 0.00002
random_size = 128

image_size = 14

gen_weight_size = [16, 16, image_size*image_size] #[16, 64, 784]
weight_size = 0
for i in range(1,len(gen_weight_size)):
    weight_size += gen_weight_size[i] * (gen_weight_size[i-1] + 1)
print(weight_size)
gen_weight_shape = []
for i in range(1, len(gen_weight_size)):
    gen_weight_shape.append(gen_weight_size[i] * gen_weight_size[i-1])
    gen_weight_shape.append(gen_weight_size[i])
print(gen_weight_shape)
for i in range(1, len(gen_weight_shape)):
    gen_weight_shape[i] += gen_weight_shape[i-1]


init = tf.random_normal_initializer(mean=0.0, stddev = 0.01)

def generator(z, reuse = False):
    l = [random_size, 512, 2048, weight_size]
    #l = [random_size, 4096, weight_size]

    with tf.variable_scope(name_or_scope = "Gen") as scope:
        gw1 = tf.get_variable(name = "w1", shape = [l[0], l[1]], initializer = init)
        gb1 = tf.get_variable(name = "b1", shape = [l[1]], initializer = init)
        gw2 = tf.get_variable(name = "w2", shape = [l[1], l[2]], initializer = init)
        gb2 = tf.get_variable(name = "b2", shape = [l[2]], initializer = init)

        gw3 = tf.get_variable(name = "w3", shape = [l[2], l[3]], initializer = init)
        gb3 = tf.get_variable(name = "b3", shape = [l[3]], initializer = init)


    hidden1 = tf.nn.relu(tf.matmul(z , gw1) + gb1)
    #output = tf.matmul(hidden1, gw2) + gb2
    hidden2 = tf.nn.relu(tf.matmul(hidden1, gw2) + gb2)
    output = tf.matmul(hidden2, gw3) + gb3
    return output

def discriminator(x, reuse = False):
    #l = [weight_size, 1024, 1024, 1]
    l = [weight_size, 1024, 1]
    
    with tf.variable_scope(name_or_scope="Dis", reuse = reuse) as scope:
        dw1 = tf.get_variable(name = "w1",  shape = [l[0], l[1]], initializer = init)
        db1 = tf.get_variable(name = "b1", shape = [l[1]], initializer = init)
        dw2 = tf.get_variable(name = "w2", shape = [l[1], l[2]], initializer = init)
        db2 = tf.get_variable(name = "b2", shape = [l[2]],  initializer = init)
        
        #dw3 = tf.get_variable(name = "w3", shape = [l[2], l[3]], initializer = init)
        #db3 = tf.get_variable(name = "b3", shape = [l[3]],  initializer = init)


    hidden1 = tf.nn.relu(tf.matmul(x , dw1) + db1)
    output = tf.nn.sigmoid(tf.matmul(hidden1, dw2) + db2)
    #hidden2 = tf.nn.relu(tf.matmul(hidden1, dw2) + db2)
    #output = tf.nn.sigmoid(tf.matmul(hidden2, dw3)  + db3)
    return output

def random_noise(batch_size):
    return np.random.normal(size=[batch_size , random_size])


g = tf.Graph()

with g.as_default():
    X = tf.placeholder(tf.float32, [None, weight_size])
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
    total_batchs = int(train_x.shape[0] / batch_size)


    print(train_x.shape)
    train_x = np.tile(train_x, (10,1))
    print(train_x.shape)

    for epoch in range(total_epochs):
        for batch in range(total_batchs):
            batch_x = train_x[batch * batch_size : (batch+1) * batch_size]
            noise = random_noise(batch_size)

            sess.run(g_train , feed_dict = {Z : noise})
            sess.run(d_train, feed_dict = {X : batch_x , Z : noise})
            
            gl, dl = sess.run([g_loss, d_loss], feed_dict = {X : batch_x , Z : noise})

        print("======= Epoch : ", epoch , " =======")
        print("generator: " , -gl )
        print("discriminator: " , -dl )

        gen_w = np.asarray(sess.run(fake_x, feed_dict = { Z : random_noise(1) }))
        gen_w = np.array_split(gen_w, gen_weight_shape, axis=1)[:-1]
        for i in range(0, len(gen_w), 2):
            gen_w[i] = np.reshape(gen_w[i], (gen_weight_size[int(i/2)], gen_weight_size[int(i/2)+1]))
       
        f = open('./epoch_resize/epoch'+str(epoch)+'.txt','w')
        for _ in range(10):
            gen_z = np.random.normal(size=[1, gen_weight_size[0]])
            gen_hidden1 = np.matmul(gen_z, gen_w[0]) + gen_w[1]
            gen_hidden1 = np.maximum(gen_hidden1, 0, gen_hidden1)
            gen_output = np.matmul(gen_hidden1, gen_w[2]) + gen_w[3]
            gen_output = 1 / (1 + np.exp(-gen_output))
            
            gen_output = np.ndarray.tolist(np.reshape(gen_output,(image_size*image_size)))
            cnt = 0
            # sample_out = ""

            for j in gen_output:
                """
                if j > 0.80:
                    sample_out += "@"
                elif j > 0.60:
                    sample_out += "0"
                elif j > 0.40:
                    sample_out += "1"
                elif j > 0.20:
                    sample_out += "."
                else:
                    sample_out += " "
                """
                f.write("%0.2f " % j)
                cnt+=1
                if cnt % image_size == 0:
                    f.write('\n')# sample_out += "\n"
            f.write('\n')# sample_out += "\n"
            # print(sample_out)
        f.close()



            
            

        save_path = saver.save(sess, "./meta-gan-resize"+".ckpt")


