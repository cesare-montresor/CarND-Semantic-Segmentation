import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess,[vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    
    return image_input,keep_prob,layer3,layer4,layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function

    reg_scale = 1e-3
    k_reg = tf.contrib.layers.l2_regularizer

    # vgg layer aggregation to matching channel size
    vgg_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding="same",kernel_regularizer=k_reg(reg_scale))
    vgg_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding="same",kernel_regularizer=k_reg(reg_scale))
    vgg_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding="same",kernel_regularizer=k_reg(reg_scale))

    # decoder composition
    # upscale x2 the last layer convolutional layer of VGG and add layer VGG4
    dec_1_x2 = tf.layers.conv2d_transpose(vgg_7, num_classes, 4, 2,  padding="same",kernel_regularizer=k_reg(reg_scale))
    dec_2_sk = tf.add(dec_1_x2, vgg_4)

    # upscale x2 the layer above and add layer VGG3
    dec_3_x4 = tf.layers.conv2d_transpose(dec_2_sk, num_classes, 4,2,padding="same",kernel_regularizer=k_reg(reg_scale))
    dec_4_sk = tf.add(dec_3_x4, vgg_3)

    # upscale x4 the layer above
    dec_5_x4 = tf.layers.conv2d_transpose(dec_4_sk, num_classes,16,8,padding="same",kernel_regularizer=k_reg(reg_scale))

    return dec_5_x4
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # reshape logits and label to fit softmax operation
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1, num_classes))

    #define loss and training operations
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, optimizer, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    print("init vars")
    sess.run(tf.global_variables_initializer())

    lr = 0.001
    kp = 0.5
    # TODO: Implement function
    for epoch in range(epochs):
        print("Epoch:", epoch)
        epoch_loss = 0
        cnt = 0
        for image, label in get_batches_fn(batch_size):
            cnt += len(image)
            operations = [
                train_op,
                cross_entropy_loss
            ]

            params = {
                input_image: image,
                correct_label: label,
                learning_rate: lr,
                keep_prob: kp
            }
            train_result,loss = sess.run(operations, params)
            #print("Batch:", cnt,"\t", loss)
            epoch_loss += loss

        avg_cost = epoch_loss/cnt
        print("AVG Loss:", avg_cost)


tests.test_train_nn(train_nn)


def run():
    epochs = 200
    num_classes = 2
    # max size for to avoid OOM warning
    # W tensorflow/core/common_runtime/bfc_allocator.cc:217] Allocator (GPU_0_bfc) ran out of memory trying to allocate 3.16GiB. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory is available.
    batch_size = 15
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)


    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        print("Prepare NN")
        correct_label = tf.placeholder("float", [None, None, None, num_classes], name="correct_label")
        learning_rate = tf.placeholder("float", name="correct_label")

        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess,vgg_path)
        output = layers(layer3,layer4,layer7, num_classes)

        print("Prepare operations")
        logits, optimizer, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function
        print("Train")
        train_nn(sess, epochs, batch_size, get_batches_fn, optimizer, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        print("Save")
        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
