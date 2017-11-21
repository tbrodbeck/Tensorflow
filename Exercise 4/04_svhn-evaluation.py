with tf.Session() as session:
    saver = tf.train.Saver()
    saver.restore(session,  tf.train.latest_checkpoint("./weights/"))
    
    test_accuracy = 0
    for step, (images, labels) in enumerate(svhn.get_test_batch(300)):
        test_accuracy += session.run(
            accuracy,
            feed_dict = {x: images, desired: labels}
        )
    
print("Test Accuracy: " + str(test_accuracy / step))
