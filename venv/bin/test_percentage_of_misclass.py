from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, FastFeatureAdversaries,MadryEtAl,SaliencyMapMethod,MomentumIterativeMethod,SPSA,VirtualAdversarialMethod, FastFeatureAdversaries, DeepFool, CarliniWagnerL2
from cleverhans_tutorials.tutorial_models import make_basic_mlp_non_conv, make_basic_cnn, make_basic_mlp_non_conv2
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans_tutorials.mnist_blackbox import substitute_model
from data import preprocess, get_data_from_file
from pathlib import Path
import os

FLAGS = flags.FLAGS


def mnist_tutorial(train_start=0, train_end=8363, test_start=0,
                   test_end=2092, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64, num_threads=None):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Get MNIST test data
    #X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,train_end=train_end,test_start=test_start,
    # test_end=test_end)

    base_dir = str(Path().resolve().parent)
    file = base_dir + '/bin/Wednesday-workingHours.pcap_ISCX.csv'
    #file = base_dir + '/bin/datasetDoS.csv'
    dataset = get_data_from_file(file)
    X_train, X_test, Y_train, Y_test = preprocess(dataset)


    train_end = X_train.shape[0]
    test_end = X_test.shape[0]
    #y_target = np.empty((Y_test.shape[0], Y_test.shape[1]))


    input_shape = X_train.shape[1]
    # Use label smoothing
    print("Dimensione dell'input Train: "+str(X_train.shape)+"\n")
    print("Dimensione dell'output Train: " + str(Y_train.shape) + "\n")

    print("Dimensione dell'input Test: " + str(X_test.shape) + "\n")
    print("Dimensione dell'output Test: " + str(Y_test.shape) + "\n")
#    assert Y_train.shape[1] == 2
#    label_smooth = .1
#    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, input_shape))
    y = tf.placeholder(tf.float32, shape=(None, 2))

    model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }

    fgsm_params = {'clip_min':2.2250738585072014e-308, 'clip_max': 1.7976931348623157e+308,'theta':0.01, 'gamma':0.7}
    #fgsm_params = {'clip_min': 0.0, 'clip_max': 1.0, 'theta': 0.01,'gamma': 0.7}
    rng = np.random.RandomState([2017, 8, 30])

    if clean_train:
        #model=make_basic_cnn(nb_filters=64)
        #model = substitute_model(img_rows=1, img_cols=78, nb_classes=2)
        model = make_basic_mlp_non_conv(input_shape=(None,input_shape))
        #model2 = make_basic_mlp_non_conv2(input_shape=(None,input_shape))
        #model = make_basic_cnn(nb_filters=nb_filters)
        preds = model.get_probs(x)

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)

        model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                    args=train_params, rng=rng, var_list=model.get_params())

        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        #fgsm = FastGradientMethod(model, sess=sess)
        #fgsm = VirtualAdversarialMethod(model, sess=sess)
        #fgsm = BasicIterativeMethod(model, sess=sess)
        #fgsm = MadryEtAl(model, sess = sess)
        fgsm = SaliencyMapMethod(model, sess=sess)
        #fgsm = FastFeatureAdversaries(model, sess=sess)
        #fgsm = MomentumIterativeMethod(model,sess=sess)
        #fgsm = DeepFool(model, sess=sess)
        #fgsm = CarliniWagnerL2(model, sess=sess)
        adv_x = fgsm.generate(x, **fgsm_params)

        preds_adv = model.get_probs(adv_x)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, preds.eval({x:X_test}, session=sess), args=eval_par)
        print('Test amount of misclassification adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc
        #print("Repeating the process, using adversarial training")

        '''
        outdir = "vatm"+"-eps"+str(fgsm_params["eps"])+"-numIter"+str(fgsm_params["num_iterations"])+"-xi"+str(fgsm_params["xi"])+"-scaled"

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        #with sess.as_default(), open("/Users/andreadelvecchio/PycharmProjects/CleverhansVirgin/"+outdir+"/adv_x.txt","w") as file1, open("/Users/andreadelvecchio/PycharmProjects/CleverhansVirgin/"+outdir+"/x_test.txt", "w") as file2:
        with sess.as_default(), open("./" + outdir + "/adv_x.txt","w") as file1, open("./" + outdir + "/x_test.txt", "w") as file2:

            array = adv_x.eval({x: X_test})
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    file1.write('%s,' % array[i, j])
                    file2.write('%s,' % X_test[i, j])
                file1.write('\n')
                file2.write('\n')

        #with sess.as_default(), open("/Users/andreadelvecchio/PycharmProjects/CleverhansVirgin/"+outdir+"/pred_adv.txt","w") as file3, open("/Users/andreadelvecchio/PycharmProjects/CleverhansVirgin/"+outdir+"/y_test.txt", "w") as file4:
        with sess.as_default(), open("./" + outdir + "/pred_adv.txt","w") as file3, open("./" + outdir + "/y_test.txt", "w") as file4:
            array2 = preds_adv.eval({x: X_test})
            for i in range(array2.shape[0]):
                for j in range(array2.shape[1]):
                    file3.write('%s,' % array2[i, j])
                    file4.write('%s,' % Y_test[i, j])
                file3.write('\n')
                file4.write('\n')
        
        '''


    # Redefine TF model graph


    return report


def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 4, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()