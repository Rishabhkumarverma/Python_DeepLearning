import tensorflow as tf
import numpy as np
image_size = 108
fcd1 = 108*108
fcd2 = 32
fcd3 = 10
labels_size = 6
import csv
steps = 10

# input = tf.placeholder(tf.float32, [None, image_size*image_size])
# labels = tf.placeholder(tf.float32, [None, labels_size])

#
# with tf.name_scope("Network"):
#
#   x = tf.placeholder(tf.float32, (None, fcd1))
#   y = tf.placeholder(tf.float32, (None, labels_size))
#
#   W1 = tf.Variable(tf.random_normal((fcd1, fcd2)))
#   b1 = tf.Variable(tf.random_normal((fcd2,)))
#   fc_relu1 = tf.nn.relu(tf.matmul(x, W1) + b1)
#
#   W2 = tf.Variable(tf.random_normal((fcd2, fcd3)))
#   b2 = tf.Variable(tf.random_normal((fcd3,)))
#   fc_relu2 = tf.nn.relu(tf.matmul(fc_relu1, W2) + b2)
#
#   W3 = tf.Variable(tf.random_normal((fcd3, labels_size)))
#   b3 = tf.Variable(tf.random_normal((labels_size,)))
#   y_pred = tf.matmul(fc_relu2, W3) + b3
#
#   loss = tf.losses.softmax_cross_entropy(y, y_pred)
#   train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
#
#   correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
#   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#   tf.summary.scalar("loss", loss)
#   merged = tf.summary.merge_all()
#
#   sess = tf.InteractiveSession()
#   sess.run(tf.global_variables_initializer())


x = tf.placeholder(tf.float32, (None, fcd1))
y = tf.placeholder(tf.float32, (None, labels_size))

# W1 = tf.Variable(tf.random_normal((fcd1, fcd2)))
# b1 = tf.Variable(tf.random_normal((fcd2,)))
# fc_relu1 = tf.nn.relu(tf.matmul(x, W1) + b1)
#
# W2 = tf.Variable(tf.random_normal((fcd2, fcd3)))
# b2 = tf.Variable(tf.random_normal((fcd3,)))
# fc_relu2 = tf.nn.relu(tf.matmul(fc_relu1, W2) + b2)
#
# W3 = tf.Variable(tf.random_normal((fcd3, labels_size)))
# b3 = tf.Variable(tf.random_normal((labels_size,)))
# y_pred = tf.matmul(fc_relu2, W3) + b3
#
# loss = tf.losses.softmax_cross_entropy(y, y_pred)
# train_op = tf.train.AdamOptimizer(1e-10).minimize(loss)
#
# correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
# tf.summary.scalar("loss", loss)
# merged = tf.summary.merge_all()


# regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
fc_relu1 = tf.layers.dense(inputs=x, units=fcd2, activation=tf.nn.relu)
fc_relu2 = tf.layers.dense(inputs=fc_relu1, units=fcd3, activation=tf.nn.relu)
output = tf.layers.dense(inputs=fc_relu2, units=labels_size)
# output = tf.layers.dense(inputs=fc_relu2, kernel_regularizer=regularizer, units=labels_size)



loss = tf.losses.softmax_cross_entropy(y, output)
# l2_loss = tf.losses.get_regularization_loss()
# loss += l2_loss
# regularizer = tf.nn.l2_loss(weights)
# loss = tf.reduce_mean(loss + beta * regularizer)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
argmaxout = tf.argmax(output, 1)
tf.summary.scalar("loss", loss)
merged = tf.summary.merge_all()






def run_training(i, image, label):
    feed_dict = {x: image, y: label}
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict=feed_dict)
        print("Step %d, training batch accuracy %g" % (i, train_accuracy))

        # Run the optimization step
    train_op.run(feed_dict=feed_dict)

def run_test(test_images, test_labels):
    # Print the test accuracy once the training is over
    accuracyTest = accuracy.eval(feed_dict={x: test_images, y: test_labels})
    # print("Test accuracy:", accuracyTest)
    return accuracyTest

# def run_trainingepochOld(n_epochs, train_X, train_y, batch_size ):
#     step = 0
#     for epoch in range(n_epochs):
#         pos = 0
#         while pos < N:
#             batch_X = train_X[pos:pos + batch_size]
#             batch_y = train_y[pos:pos + batch_size]
#             feed_dict = {x: batch_X, y: batch_y}
#             _, summary, loss = sess.run([train_op, merged, loss], feed_dict=feed_dict)
#             print("epoch %d, step %d, loss: %f" % (epoch, step, loss))
#             tf.summary.adsmmary(summary, step)
#             writer = tf.summary.FileWriter('./graphs', sess.graph)
#
#             step += 1
#             pos += batch_size

def run_trainingepoch(n_epochs, train_X, train_y, test_x, test_y, testData,  batch_size ,numberimage):
    with tf.Session() as sess:
        #sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        # ____step 2:____ creating the writer inside the session
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        step = 0
        number_of_test_image = test_x.shape[0]
        for epoch in range(n_epochs):
            pos = 0
            iter_acc = 0.0
            count = 0.0
            while pos <  (numberimage - batch_size) :
                image = train_X[pos:pos + batch_size, :, :]
                batch_y = train_y[pos:pos + batch_size, :]
                image = np.array(image, dtype=np.float32)
                test_images = test_x[0:100, :, :]
                test_labels = test_y[0:100, :]
                test_images = np.array(test_images, dtype=np.float32)

                test_images = np.subtract(test_images, 128.0)
                image = np.subtract(image, 128.0)
                # test_images = np.subtract(255.0, test_images)
                # image = np.subtract( 255.0, image)
                test_images = np.divide(test_images, 128.0)
                image = np.divide(image, 128.0)
                batch_X = np.reshape(image, (batch_size, 108 * 108))
                test_images = np.reshape(test_images, (100, 108 * 108))


                feed_dict = {x: batch_X, y: batch_y}
                lossv, train_op_v, accuracy_v, summary, = sess.run([loss, train_op, accuracy, merged], feed_dict=feed_dict)
                # if pos % 1000 == 0:
                #     print("epoch %d, step %d, loss: %f" % (epoch, step, lossv))
                #     print("Train accuracy ; " ,  accuracy_v)
                #     test_accuracy = run_test(test_images, test_labels)


                writer.add_summary(summary, step)
                iter_acc+=accuracy_v
                count +=1.0
                step += 1
                pos += batch_size

            accuracy_v_aver = (iter_acc/count)
            print("Average Train :", accuracy_v_aver)
            if accuracy_v_aver > 0.50:
                accuracyTest = 0.0
                pos = 0
                count = 0
                while pos < number_of_test_image - batch_size:
                    test_images = test_x[pos:pos + batch_size, :, :]
                    test_labels = test_y[pos:pos + batch_size, :]
                    test_images = np.array(test_images, dtype=np.float32)
                    # test_images = np.subtract(255.0, test_images)
                    test_images = np.subtract(test_images, 128.0)
                    test_images = np.divide(test_images, 128.0)
                    test_images = np.reshape(test_images, (batch_size, 108 * 108))
                    accuracyTest += run_test(test_images, test_labels)
                    pos += batch_size
                    count += 1.0
                acc = (accuracyTest / count)
                print("testAcciracy:", acc )
                assignmentOut = dict()
                if acc > 0.45:
                    TestImageSize = testData.shape[0]
                    for imageID in range(TestImageSize):
                        imageTest = testData[imageID, :, :]
                        image = np.array(imageTest, dtype=np.float32)
                        image = np.subtract(image, 128.0)
                        image = np.divide(image, 128.0)
                        Test_batch_X = np.reshape(image, (1, 108 * 108))
                        feed_dict = {x: Test_batch_X}
                        argMax, outputNmae = sess.run([argmaxout, output],feed_dict=feed_dict)
                        # print(argMax)
                        imagestrID = str(imageID)
                        assignmentOut[imagestrID] = argMax[0]
                    fileName = "result_" + str(acc) + "_.csv"
                    with open(fileName, 'w') as f:
                        for key in assignmentOut.keys():
                            f.write("%s,%s\n" % (key, assignmentOut[key]))






        # number_of_test_image = test_x.shape[0]
        accuracyTest=0.0
        count = 0.0
        pos = 0
        while pos < (number_of_test_image - batch_size):
            test_images = test_x[pos:pos+batch_size, :, :]
            test_labels = test_y[pos:pos+batch_size, :]
            test_images = np.array(test_images, dtype=np.float32)
            # test_images = np.subtract(255.0, test_images)
            test_images = np.subtract(test_images,128.0)
            test_images = np.divide(test_images, 128.0)
            test_images = np.reshape(test_images, (batch_size, 108 * 108))
            accuracyTest += run_test(test_images, test_labels)
            pos += batch_size
            count+=1.0
        print("testAcciracy:",(accuracyTest/count))






