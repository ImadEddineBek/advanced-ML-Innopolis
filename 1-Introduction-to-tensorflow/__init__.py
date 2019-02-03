import tensorflow as tf

tf.reset_default_graph()  # To clear the defined variables and operations of the previous cell
# create graph
a = tf.constant(2, name="x")
b = tf.Variable(3, name="y")
c = tf.add(a, b, name="addition")
d = tf.multiply(a, b, name="multiplication")

zeros = tf.zeros(shape=[5,5], dtype=tf.float32, name="zeros")
ones = tf.ones_like(zeros,name="ones")

e = tf.pow(d, c, name="Pow")
o = tf.matmul(zeros, ones, name="addition2")
# creating the writer out of the session

# launch the graph in a session
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # or creating the writer inside the session
    print(sess.run(e))
    print(sess.run(o))
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    writer.close()
