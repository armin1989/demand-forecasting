import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time


def get_mbs_error(feats, labels, x, y_hat, sess, feed_dict, mbs):
    """
    Return error on given input x by dividing it into mini-batches of size mbs and feeding it to network.

    I wrote this function because if the size of training or validation sets is too large, can't fed them
    to the network in a single step.


    :param x:  tensor for input placeholder
    :param y_hat : tensor for output of network
    :param sess:
    :param feed_dict:
    :param mbs:
    :return:
    """
    errors = []

    for i in range(int(np.ceil(np.size(feats, 0) / mbs))):
        feed_dict[x] = feats[i * mbs: (i + 1) * mbs, :, :, :]
        y = labels[i * mbs: (i + 1) * mbs, :]
        y_predict = sess.run(y_hat, feed_dict=feed_dict)
        errors.append(SMAPE(y_predict, y))

    return np.mean(errors)


def SMAPE(preds, target):
    '''
    Function to calculate SMAPE
    '''
    n = len(preds)
    # masked_arr = ~((preds==0) & (target==0))
    # preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def add_connected_layer(weight_size, prefix, W, b):
    """
    Create weights for single fully connected layer according to given sizes. Append weights to W, b.

    :param weight_size: [input_nodes, output_nodes] size of weights in each layer
    :param prefix : str, layer prefix used for naming tensors
    :param W : list of weights to append to
    :param b : list of biases append to
    :return: l2 of weights
    """
    W.append(tf.get_variable("Wf" + prefix, weight_size, tf.float32,
                             initializer=tf.contrib.layers.xavier_initializer()))
    b.append(tf.Variable(tf.constant(0.01, shape=[weight_size[-1]], dtype=tf.float32), name="bf" + prefix))
    return tf.nn.l2_loss(W[-1])


def add_connected_layers(weight_sizes, prefix):
    """
    Add fully connected layers with batch-norm and return weights, scales, shifts.

    :param weight_sizes: [[n_i, n_o] * num_layers] array of sizes of weights in each layer
    :param prefix: prefix used for naming tensors
    :return: dict containing W, gamma and beta, weight norms
    """
    variables = dict()
    variables["W_f"] = []
    variables["b_f"] = []

    w_norm_f = 0
    for i in range(len(weight_sizes) - 1):
        w_norm_f += add_connected_layer([weight_sizes[i], weight_sizes[i + 1]], prefix + str(i),
                                        variables["W_f"], variables["b_f"])
    return variables, w_norm_f


def fc_layers(input_data, variables, activation_fun, keep_prob=1.0):
    """
    Return output of a feed forward network.

    :param variables: dicitionary containing learnable variables
    :param activation_fun: List of callables, list of activation functions for each layer,
                        if len(layer_func) == 1 then the same activation function is used for all the layers
    :param input_data: Input to the network
    :param keep_prob: drop out probability (keeping probability for dropout, this percentage of connections are kept)
    :return: Output of the feed forward network.
    """
    W_f = variables["W_f"]
    b_f = variables["b_f"]
    b_o = variables["b_o"]
    W_o = variables["W_o"]

    input_next = input_data

    for l in range(len(W_f)):

        # creating local variables for moment update procedure
        x = tf.matmul(input_next, W_f[l]) + b_f[l]
        z = activation_fun(x)   # for now lets assume no batch-norm
        input_next = tf.nn.dropout(z, keep_prob)

    return tf.add(tf.matmul(input_next, W_o), b_o, name="y_hat")


def conv_layer(input_array, filter_size, kp, postfix, is_training):
    """Single convolutional layer without pooling, assuming strides are all one.

    :param input_array : Input data with shape [m, n_h, n_w, n_c]
    :param filter_size : size of filter to be used for convolution, format: [n_h, n_w, n_c, n_c']
    :param activation : activation function to be used in each layer
    :param kp : keeping probability in dropout
    :param postfix : used for naming created variables
    :param is_training : used for batchnormalization
    :return: output of layer, W, b
    """
    # filter weights
    W = tf.get_variable("Wc" + postfix, filter_size, tf.float32, initializer=tf.contrib.layers.xavier_initializer())
    # convolution
    x = tf.nn.conv2d(input_array, W, [1, 1, 1, 1], padding="SAME")
    # batch-norm before activation
    x_norm = tf.layers.batch_normalization(x, training=is_training, axis=3)
    # dropout (activation is applied after dropout)
    output = tf.nn.dropout(x_norm, kp)

    return output, W


def add_conv_layers(net_input, filter_sizes, prefix, kp, is_training, activation_fun=tf.nn.relu):
    """
    Add a bunch of conv layers according to length of filter sizes and return ouput of conv layers, weight l2 norms,
    weights and biases

    Note : no max-pooling, strides are length one, using batch-norm

    :param filter_sizes: [[n_h, n_w, n_c', n_c] * num_layers] array of sizes of layers
    :param net_input : input to conv layers
    :param activation_fun: activation function (e.g. tf.nn.relu, tf.nn.tanh, ... )
    :param prefix : str, used for naming variables
    :param kp : keeping probability for dropout
    :param is_training : to indicate if we are training for batch-norm
    :return: output of convolutional layers
             norm of weights, used for regularization
             list containing weights
             list containing biases
    """
    weights = []
    w_norm = 0

    input_next = net_input
    for l in range(len(filter_sizes)):
        conv_output, W = conv_layer(input_next, filter_sizes[l], kp, prefix + str(l), is_training)
        # since the conv layer does not perform activation, we will do it here
        input_next = activation_fun(conv_output)
        weights.append(W)
        w_norm += tf.nn.l2_loss(W)

    return input_next, w_norm, weights


def create_graph(input_size, params):
    """
    Create and return tensorflow graph and associated tensors.

    Note : in this version I am not using pooling layers and use strides of 1

    :param input_size: [n_h, n_w, n_c] size of each input sample
    :param output_size:  size of output (Ng + 1)
    :param params:
    :return: graph
    """
    # getting filter sizes from params dict
    conv_filters = params["conv_filters"]
    res_filters = params["res_filters"]
    inception_filters = params["inception_filters"]
    inception_reduce_filters = params["inception_reduce_filters"]
    fc_filters = params["fc_filters"]
    epsilon = 10e-8

    graph = tf.Graph()

    with graph.as_default():
        # placeholders for input ant output and keep_prob in dropout
        x = tf.placeholder(tf.float32, [None] + list(input_size), name="x")
        y = tf.placeholder(tf.float32, [None, 1], name="y")
        # keeping probability in dropout, only used dropout in fully connected layers
        kp = tf.placeholder(tf.float32, name="kp")
        kp_c = tf.placeholder(tf.float32, name="kp_c")
        is_training = tf.placeholder(tf.bool, name="is_training")

        # in inference time, feed normalize with 1, evaluate mean_data_const and std_data_const and feed it back to
        # graph in corresponding placeholders

        # integer to indicate whether we are training
        normalize = tf.placeholder(tf.float32, shape=[1], name="normalize")
        # mean of data used during inference
        mean_data = tf.placeholder(tf.float32, list(input_size), name="mean_data")
        # std of data used to normalize it during inference
        std_data = tf.placeholder(tf.float32, list(input_size), name="std_data")
        # creating variables to be update by the above placeholders, the point of this is to keep mean and std of data
        # in the graph itself
        mean_data_const = tf.Variable(initial_value=np.zeros(list(input_size)),
                                      trainable=False, name="mean_data_tensor", dtype=tf.float32)
        std_data_const = tf.Variable(initial_value=np.ones(list(input_size)),
                                     trainable=False, name="std_data_tensor", dtype=tf.float32)
        mean_update = mean_data_const.assign(mean_data)
        std_update = std_data_const.assign(std_data)
        # normalize data with amounts in mean_data_const and std_data_const
        # set normalize to 1 if input data is not normalized prior to being fed to the network
        x_normalized = (x - normalize * mean_data_const) / \
                       ((1 - normalize) + normalize * std_data_const + epsilon)

        # convolutional layers
        x_conv, w_norm_conv, _ = \
            add_conv_layers(x_normalized, conv_filters, '_c_', kp_c, is_training)

        # adding inception or residual layers"
        if res_filters:
            # residual layers
            x_res, w_norm_res, _ = \
                add_res_layers(x_conv, res_filters, '_res_', kp_c, is_training)
        elif inception_filters:
             x_res, w_norm_res, _ = \
                 add_inception_layers_wskip(x_conv, inception_filters, inception_reduce_filters, '_inc_', kp_c,
                                             is_training)
        else:
            x_res = x_conv
            w_norm_res = 0

        # fully connected part
        x_flat = tf.contrib.layers.flatten(x_res)
        fc_filters.insert(0, x_flat.shape[1])
        fc_vars, w_norm_f = add_connected_layers(fc_filters, '_fc')
        # weights for output layer, just before softmax
        fc_vars["W_o"] = tf.get_variable("W_out", [fc_filters[-1], 1], tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer())
        fc_vars["b_o"] = tf.Variable(tf.constant(0.01, shape=[1, 1], dtype=tf.float32), name="b_out")
        y_hat = fc_layers(x_flat, fc_vars, tf.nn.relu, kp)
        error = tf.reduce_mean(tf.abs(y - y_hat), axis=0)

        # weight penalty term
        regularizer = tf.add(w_norm_res + w_norm_conv + w_norm_f,
                             tf.nn.l2_loss(fc_vars["W_o"]), name="weight_penalty")

        cost_func = tf.add(error, params["lambd"] * regularizer, name="cost_func")

        # Operation block: optimizing the cost function using momentum2
        # adding update ops as dependency to train operation to make sure batch moments are update
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train = tf.train.AdamOptimizer(learning_rate=params["alpha"]).minimize(cost_func)
            #train = tf.train.MomentumOptimizer(learning_rate=params["alpha"], momentum=0.9).minimize(cost_func)
            #train = tf.train.RMSPropOptimizer(learning_rate=params["alpha"]).minimize(cost_func)

        # train = tf.train.MomentumOptimizer(learning_rate=params["alpha"], momentum=0.9).minimize(cost_func)
        # train = tf.train.RMSPropOptimizer(learning_rate=params["alpha"]).minimize(cost_func)
        # Operation block: initializing all variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(name="saver", max_to_keep=None)
        tensors = {"init": init, "cost_func": cost_func,  "y_hat": y_hat,
               "saver": saver, "train": train, "error": error,
               "mean_update": mean_update, "std_update": std_update}

    return graph, tensors


def run_training(x_train, y_train, x_valid, y_valid, graph, tensors, hyper_params):
    """
    Run actual training and return session.

    :param x_train : training features
    :param y_train : training labels
    :param x_valid : validation features
    :param y_valid : validation lables
    :param graph: tensorflow graph object
    :param tensors: dictionary of necessary tensors
    :param hyper_params: dict of hyper parameters
    :return: session, array of training and validation errors
    """
    mbs = hyper_params["mbs"]
    num_epochs = hyper_params["num_epochs"]
    keep_p = hyper_params["kp"]
    keep_p_c = hyper_params["kp_c"]
    max_validation_check = hyper_params["validation_limit"]

    # normalizing data
    epsilon = 1e-7
    mean_data = np.mean(np.concatenate((x_train, x_valid), axis=0), 0)
    std_data = np.var(np.concatenate((x_train, x_valid), axis=0), 0) ** 0.5
    x_train = (x_train - mean_data) / (std_data + epsilon)
    x_valid = (x_valid - mean_data) / (std_data + epsilon)
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_valid = np.reshape(y_valid, (y_valid.shape[0], 1))

    init, train, cost_func, saver = tensors["init"], tensors["train"], tensors["cost_func"], tensors["saver"]
    mean_update = tensors["mean_update"]
    std_update = tensors["std_update"]

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=hyper_params["gpu_load"])
    # config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(graph=graph)
    validation_check = 0
    sess.run(init)


    # getting placeholders from graph
    kp = graph.get_tensor_by_name("kp:0")
    kp_c = graph.get_tensor_by_name("kp_c:0")
    x = graph.get_tensor_by_name("x:0")
    y_hat = graph.get_tensor_by_name("y_hat:0")
    y = graph.get_tensor_by_name("y:0")
    normalize = graph.get_tensor_by_name("normalize:0")
    mean_data_placeholder = graph.get_tensor_by_name("mean_data:0")
    std_data_placeholder = graph.get_tensor_by_name("std_data:0")
    is_training = graph.get_tensor_by_name("is_training:0")

    feed_dict = dict()
    # no need to normalized here since shuffle data which is called earlier does that for us
    # set normalize to 1 during inference, in this case mean and std of data, already stored in graph, should be fed back
    feed_dict[normalize] = [0]
    feed_dict[mean_data_placeholder] = mean_data
    feed_dict[std_data_placeholder] = std_data
    epoch = 0
    train_error = []
    valid_error = []
    while epoch < num_epochs:

        start_time = time.time()
        # going through training data
        for i in range(int(np.ceil(np.size(x_train, 0) / mbs))):
            feed_dict[kp] = keep_p
            feed_dict[kp_c] = keep_p_c
            feed_dict[is_training] = True
            x_batch = x_train[i * mbs: (i + 1) * mbs, :, :, :]
            y_batch = y_train[i * mbs: (i + 1) * mbs, :]
            # preparing feed dictionary #
            feed_dict[x] = x_batch
            feed_dict[y] = y_batch
            sess.run([train], feed_dict=feed_dict)

        # getting training and validation errors after we went through all minibatches
        feed_dict[kp] = 1
        feed_dict[kp_c] = 1
        feed_dict[is_training] = False
        train_error.append(get_mbs_error(x_train, y_train, x, y_hat, sess, feed_dict, mbs))
        valid_error.append(get_mbs_error(x_valid, y_valid, x, y_hat, sess, feed_dict, mbs))

        end_time = time.time()


        # check for early stopping #
        if epoch >= 1 and valid_error[epoch] > valid_error[epoch - 1] * 1.001:
            validation_check += 1
            #print("Validation error increased! counting " + str(validation_check))

        if validation_check == max_validation_check:
            print("Reached max validation, quitting!")
            break

        if epoch % 5 == 0:
            print("epoch = %d" % epoch)
            print("elpased time (per epoch):" + str(end_time - start_time))
            print("training error = %f, validation error = %f" % (train_error[epoch], valid_error[epoch]))

        if epoch % 50 == 0 and epoch > 0:
            train.learning_rate = hyper_params["alpha"] / 2
            hyper_params["alpha"] /= 2

        epoch += 1

    return sess, graph, mean_data, train_error, valid_error


def add_res_layers(net_input, filter_sizes, prefix, kp, is_training, activation_fun=tf.nn.relu):
    """
    Add a bunch of conv layers according to length of filter sizes and return weights, scale and shifts (W, gamma, beta)
    Also, pass input through these layers and return result along with batch moments and update ops for them

    Note : no max-pooling, strides are length one, using batch-norm

    :param filter_sizes: [[n_h, n_w, n_c', n_c] * num_layers] array of sizes of layers
    :param net_input : input to conv layers
    :param activation_fun: activation function (e.g. tf.nn.relu, tf.nn.tanh, ... )
    :param prefix : str, used for naming variables
    :param kp : keeping probability for dropout
    :return: dictionary containing W, gamma and beta, norm of weights
             norm of weights used for regularization
             output of layers
             average mean of batches
             average var of batches
    """
    weights = []
    w_norm = 0

    input_next_a = net_input
    for l in range(len(filter_sizes)):

        # weights for convolution according to given filter size
        conv_output_a, W_a = conv_layer(input_next_a, filter_sizes[l], kp, prefix + 'a' + str(l), is_training)
        input_next_b = activation_fun(conv_output_a)

        # weights for second layer of conv
        conv_output_b, W_b = conv_layer(input_next_b, filter_sizes[l], kp, prefix + 'b' + str(l), is_training)
        # skip connection and second activation function
        input_next_a = activation_fun(conv_output_b + input_next_a)

        weights.extend([W_a, W_b])
        w_norm += tf.nn.l2_loss(W_a) + tf.nn.l2_loss(W_b)

    return input_next_a, w_norm, weights


def add_inception_layers(net_input, filter_sizes, reduce_filter_sizes, prefix, kp, is_training,
                             activation_fun=tf.nn.relu):
    """
    Add a bunch of inception layers according to length of filter sizes and return weights.
    Also, pass input through these layers and return result


    :param filter_sizes: [[n_h, n_w, n_c', n_c] * num_layers] array of sizes of layers
    :param reduce_filter_sizes: [[n_h, n_w, n_c', n_c] * num_layers] array of sizes of filters in reduce banks layers
    :param net_input : input to inception layers
    :param activation_fun: activation function (e.g. tf.nn.relu, tf.nn.tanh, ... )
    :param prefix : str, used for naming variables
    :param kp : keeping probability for dropout
    :return: output of layers
             norm of weights used for regularization
    """
    weights = []
    w_norm = 0

    input_next = net_input
    for l in range(len(filter_sizes)):
        filters = filter_sizes[l]
        reduce_filters = reduce_filter_sizes[l]

        outputs = list()

        # stand-alone 1x1 filter bank (technically the size can be anything, just that there are no reduce filters)
        out_temp, W_0 = conv_layer(input_next, filters[0], kp, prefix + '_main_0_' + str(l), is_training)
        outputs.append(activation_fun(out_temp))

        weights.append(W_0)
        w_norm += tf.nn.l2_loss(W_0)

        for i_l in range(1, len(filters)):

            reduced, W = conv_layer(input_next, reduce_filters[i_l - 1], kp, prefix + '_reduce_{}_{}'.format(i_l, l),
                                    is_training)
            weights.append(W)
            w_norm += tf.nn.l2_loss(W)

            out_temp, W = conv_layer(activation_fun(reduced), filters[i_l], kp,
                                        prefix + '_main_{}_{}'.format(i_l, l), is_training)
            outputs.append(activation_fun(out_temp))

            weights.append(W)
            w_norm += tf.nn.l2_loss(W)

        # appending final max-pool
        # max_pooled = tf.nn.avg_pool(input_next, [1, 5, 5, 1], [1, 1, 1, 1], padding="SAME")
        # out_temp, W = conv_layer(max_pooled, reduce_filters[i_l], kp,
        #                             prefix + '_reduced_{}_{}'.format(i_l, l), is_training)
        # weights.append(W)
        # w_norm += tf.nn.l2_loss(W)
        #
        # outputs.append(activation_fun(out_temp))

        input_next = tf.concat(outputs, 3)

    return input_next, w_norm, weights


def add_inception_layers_wskip(net_input, filter_sizes, reduce_filter_sizes, prefix, kp, is_training,
                             activation_fun=tf.nn.relu):
    """
    Add a bunch of inception layers according to length of filter sizes and return weights.
    Also, pass input through these layers and return result


    :param filter_sizes: [[n_h, n_w, n_c', n_c] * num_layers] array of sizes of layers
    :param reduce_filter_sizes: [[n_h, n_w, n_c', n_c] * num_layers] array of sizes of filters in reduce banks layers
    :param net_input : input to inception layers
    :param activation_fun: activation function (e.g. tf.nn.relu, tf.nn.tanh, ... )
    :param prefix : str, used for naming variables
    :param kp : keeping probability for dropout
    :return: output of layers
             norm of weights used for regularization
    """
    weights = []
    w_norm = 0

    input_next_a = net_input
    for l in range(len(filter_sizes)):
        filters = filter_sizes[l]
        reduce_filters = reduce_filter_sizes[l]

        outputs = list()

        # stand-alone 1x1 filter bank (technically the size can be anything, just that there are no reduce filters)
        out_temp, W_0 = conv_layer(input_next_a, filters[0], kp, prefix + '_main_a_0_' + str(l), is_training)
        outputs.append(activation_fun(out_temp))

        weights.append(W_0)
        w_norm += tf.nn.l2_loss(W_0)

        for i_l in range(1, len(filters)):
            reduced, W = conv_layer(input_next_a, reduce_filters[i_l - 1], kp, prefix + '_reduce_a_{}_{}'.format(i_l, l),
                                    is_training)
            weights.append(W)
            w_norm += tf.nn.l2_loss(W)

            out_temp, W = conv_layer(activation_fun(reduced), filters[i_l], kp,
                                     prefix + '_main_a_{}_{}'.format(i_l, l), is_training)
            outputs.append(activation_fun(out_temp))

            weights.append(W)
            w_norm += tf.nn.l2_loss(W)

        # appending final max-pool
        # max_pooled = tf.nn.avg_pool(input_next_a, [1, 5, 5, 1], [1, 1, 1, 1], padding="SAME")
        # out_temp, W = conv_layer(max_pooled, reduce_filters[i_l], kp,
        #                          prefix + '_reduced_a_{}_{}'.format(i_l, l), is_training)
        # weights.append(W)
        # w_norm += tf.nn.l2_loss(W)
        #
        # outputs.append(activation_fun(out_temp))

        input_next_b = tf.concat(outputs, 3)

        # second layer of inception filter banks, activation function is at the end now
        outputs = list()

        # stand-alone 1x1 filter bank (technically the size can be anything, just that there are no reduce filters)
        out_temp, W_0 = conv_layer(input_next_b, filters[0], kp, prefix + '_main_b_0_' + str(l), is_training)
        outputs.append(out_temp)

        weights.append(W_0)
        w_norm += tf.nn.l2_loss(W_0)

        for i_l in range(1, len(filters)):
            reduced, W = conv_layer(input_next_b, reduce_filters[i_l - 1], kp, prefix + '_reduce_b_{}_{}'.format(i_l, l),
                                    is_training)
            weights.append(W)
            w_norm += tf.nn.l2_loss(W)

            out_temp, W = conv_layer(activation_fun(reduced), filters[i_l], kp,
                                     prefix + '_main_b_{}_{}'.format(i_l, l), is_training)
            outputs.append(out_temp)

            weights.append(W)
            w_norm += tf.nn.l2_loss(W)

        # appending final max-pool
        # max_pooled = tf.nn.avg_pool(input_next_b, [1, 5, 5, 1], [1, 1, 1, 1], padding="SAME")
        # out_temp, W = conv_layer(max_pooled, reduce_filters[i_l], kp,
        #                          prefix + '_reduced_b_{}_{}'.format(i_l, l), is_training)
        # weights.append(W)
        # w_norm += tf.nn.l2_loss(W)
        # outputs.append(activation_fun(out_temp))

        input_next_a = activation_fun(tf.concat(outputs, 3) + input_next_a)

    return input_next_a, w_norm, weights


def one_hot_encoder(df, ohe_cols=['store','item','dayofmonth','dayofweek','month','weekofyear']):
    '''
    One-Hot Encoder function
    '''
    print('Creating OHE features..\nOld df shape:{}'.format(df.shape))
    df = pd.get_dummies(df, columns=ohe_cols)
    print('New df shape:{}'.format(df.shape))
    return df



def get_hyerparams():
    """
    Return hyperparameters for NN training.

    :param state_type: str, linear or square, specifies type of state used for NN training, filter sizes depend on this
    :return:
    """
    n_f = 40
    params = dict()
    params["graph_file_name"] = ""
    params["lambd"] = 0.0001  # coeff of weight penalty
    params["num_epochs"] = 200
    params["kp"] = 1
    params["kp_c"] = 1
    params["optimizer"] = "Adam"
    params["do_bn"] = True
    params["conv_filters"] = [[5, 1, 1, n_f]]
    params["res_filters"] = [[10, 1, n_f, n_f]] * 5
    params["inception_filters"] = []
    params["inception_reduce_filters"] = []
    params["fc_filters"] = [256, 256, 256] * 3
    params["alpha"] = 0.0001   # learning rate
    params["mbs"] = 128# mini-batch size
    params["train_window"] = 365 * 50 * 10
    params["valid_window"] = 365 * 50 * 10
    params["validation_limit"] = 1000

    return params



if __name__ == "__main__":
    hyper_params = get_hyerparams()

    parser = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    train_data = pd.read_csv("train.csv", parse_dates=["date"])

    # explore this data set
    print(train_data.describe())
    print(train_data.head())

    # creating features
    train_data['dayofmonth'] = train_data.date.dt.day
    train_data['dayofyear'] = train_data.date.dt.dayofyear
    train_data['dayofweek'] = train_data.date.dt.dayofweek
    train_data['month'] = train_data.date.dt.month
    train_data['year'] = train_data.date.dt.year
    train_data['weekofyear'] = train_data.date.dt.weekofyear
    train_data['is_month_start'] = (train_data.date.dt.is_month_start).astype(int)
    train_data['is_month_end'] = (train_data.date.dt.is_month_end).astype(int)
    train_data.head()

    train_data.sort_values(by=['date', 'store', 'item'], axis=0, inplace=True)

    train = one_hot_encoder(train_data, ohe_cols=['store', 'item', 'dayofweek', 'month'])
    features = train.drop(["date", "sales"], axis=1)
    targets = train["sales"]
    #print(train.shape)
    #print(features.info())

    graph, tensors = create_graph((features.shape[1], 1, 1), hyper_params)
    #print(graph)
    tw = hyper_params["train_window"]
    vw = hyper_params["valid_window"]
    # nested cross validation
    valid_errors = []
    train_errors = []
    v_idx = 0
    for end_idx in range(tw, train.shape[0] - vw, tw):
        print(v_idx)
        x_train = np.reshape(features.iloc[:end_idx].values,
                             (end_idx, features.shape[1], 1, 1))
        y_train = targets.iloc[:end_idx].values
        x_valid = np.reshape(features.iloc[end_idx: end_idx + vw].values,
                             (vw, features.shape[1], 1, 1))
        y_valid = targets.iloc[end_idx: end_idx + vw].values
        _, _, _, train_error, valid_error = run_training(x_train, y_train, x_valid, y_valid, graph, tensors, hyper_params)
        plt.plot(train_error, color="blue", label="Training error")
        plt.plot(valid_error, color="red", label="Valid error")
        plt.legend()
        plt.show()
        valid_errors.append(valid_error)
        train_errors.append(train_error)
        v_idx += 1
    print("SMAPE for training error : %.3f" % np.mean(train_errors))
    print("SMAPE for valid error : %.3f" % np.mean(valid_errors))

