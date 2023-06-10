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
#from data import preprocess, get_data_from_file
from dataPrep import preprocess, get_data_from_file
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import os

from decisionTreeTest import decisionTree, compute_metrics
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

FLAGS = flags.FLAGS
features = ['Protocol', 'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets',
            'Total_Length_of_Bwd_Packets', 'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean',
            'Fwd_Packet_Length_Std', 'Bwd_Packet_Length_Max', 'Bwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean',
            'Bwd_Packet_Length_Std', 'Flow_Bytes/s', 'Flow_Packets/s', 'Flow_IAT_Mean', 'Flow_IAT_Std', 'Flow_IAT_Max',
            'Flow_IAT_Min', 'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max', 'Fwd_IAT_Min',
            'Bwd_IAT_Total',
            'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Fwd_PSH_Flags', 'Bwd_PSH_Flags',
            'Fwd_URG_Flags',
            'Bwd_URG_Flags', 'Fwd_Header_Length', 'Bwd_Header_Length', 'Fwd_Packets/s', 'Bwd_Packets/s',
            'Min_Packet_Length',
            'Max_Packet_Length', 'Packet_Length_Mean', 'Packet_Length_Std', 'Packet_Length_Variance', 'FIN_Flag_Count',
            'SYN_Flag_Count', 'RST_Flag_Count', 'PSH_Flag_Count', 'ACK_Flag_Count', 'URG_Flag_Count', 'CWE_Flag_Count',
            'ECE_Flag_Count', 'Down/Up_Ratio', 'Average_Packet_Size', 'Avg_Fwd_Segment_Size', 'Avg_Bwd_Segment_Size',
            'Fwd_Avg_Bytes/Bulk', 'Fwd_Avg_Packets/Bulk', 'Fwd_Avg_Bulk_Rate',
            'Bwd_Avg_Bytes/Bulk',
            'Bwd_Avg_Packets/Bulk', 'Bwd_Avg_Bulk_Rate', 'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes',
            'Subflow_Bwd_Packets',
            'Subflow_Bwd_Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
            'min_seg_size_forward',
            'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min', 'Label']
def find_n_neighbours(df,n):
    order = np.argsort(df.values, axis=1)[:, :n] #controllare comportamento di questa istruzion. Questa istruzione non e' necessaria
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index,
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    # il dataset restituito e' quello contenente gli indici dei valori top n
    return df

def User_item_score(user,item, sim_user_n, similarity_matrix, oracle):
    a = sim_user_n[sim_user_n.index==user].values #definisce gli indici degli n oggetti piu' simili
    b = a.squeeze().tolist() #lista degli indici
    c = oracle.loc[:,item] #colonna con gli indici
    d = c[c.index.isin(b)] #elementi effettivi associati agli indici in b
    f = d[d.notnull()] #valori not null delle label per la stima

    #avg_user = Mean.loc[Mean['userId'] == user,'rating'].values[0]
    index = f.index.values.squeeze().tolist()
    corr = similarity_matrix.loc[user,index]
    fin = pd.concat([f, corr], axis=1)
    fin.columns = ['adg_score','correlation']
    fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1) #qui andrebbe tolta la moltiplicazione con x[correlation probabilmente]
    nume = fin['score'].sum()
    deno = fin['correlation'].sum() #e' giusto questo calcolo? Ossia al denumeratore non ci dovrebbero essere il numero di user comparati?
    #final_score = avg_user + (nume/deno)
    final_score =  (nume / deno)
    return final_score


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

    base_dir = str(Path().resolve().parent)

    fileTrain = base_dir+'/bin/ADV-training.csv'
    datasetTrain = get_data_from_file(fileTrain)
    fileTest = base_dir+'/bin/ADV-baseline-PARTE1.csv'
    datasetTest = get_data_from_file(fileTest)
    fileVal = base_dir + '/bin/ADV-baseline-PARTE2.csv'
    datasetVal = get_data_from_file(fileVal)

    X_train, Y_train, scaler1 = preprocess(datasetTrain, True)
    X_test, Y_test, scaler = preprocess(datasetTest, True)
    X_FIL, Y_FIL, scaler2 = preprocess(datasetVal, False)

    train_end = X_train.shape[0]
    test_end = X_test.shape[0]
    #y_target = np.empty((Y_test.shape[0], Y_test.shape[1]))


    input_shape = X_train.shape[1]
    # Use label smoothing
    print("Dimensione dell'input Train: "+str(X_train.shape)+"\n")
    print("Dimensione dell'output Train: " + str(Y_train.shape) + "\n")

    print("Dimensione dell'input Test: " + str(X_test.shape) + "\n")
    print("Dimensione dell'output Test: " + str(Y_test.shape) + "\n")

    print("Dimensione dell'input Val: " + str(X_FIL.shape) + "\n")
    print("Dimensione dell'output Val: " + str(Y_FIL.shape) + "\n")
#    assert Y_train.shape[1] == 2
#    label_smooth = .1
#    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, input_shape))
    y = tf.placeholder(tf.float32, shape=(None, 2))




    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }



    #fgsm_params = {'clip_min': 2.2250738585072014e-308, 'clip_max': 1.7976931348623157e+308, 'eps': 0.1} #FGSM
    fgsm_params = {'clip_min': 2.2250738585072014e-308, 'clip_max': 1.7976931348623157e+308, 'theta': 0.1, 'gamma': 0.5}  # JSMA
    #fgsm_params = {'clip_min': 2.2250738585072014e-308, 'clip_max': 1.7976931348623157e+308, 'nb_candidate': 2,'nb_classes': 2, 'overshoot':0.9, 'max_iter':15}  # DeepFool
    #fgsm_params = {'clip_min':2.2250738585072014e-308,'clip_max': 1.7976931348623157e+308,'eps': 3.5,'xi':3.5,'num_iterations':1} #VirtualADV



    rng = np.random.RandomState([2017, 8, 30]) #riproducibilita'

    if clean_train:

        model = make_basic_mlp_non_conv(input_shape=(None,input_shape))
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
        fgsm = SaliencyMapMethod(model, sess=sess)
        #fgsm = DeepFool(model, sess=sess)
        #fgsm = VirtualAdversarialMethod(model, sess=sess)

        adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model.get_probs(adv_x)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc


        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc
        #print("Repeating the process, using adversarial training")

        #outdir = "bim"+"-eps"+str(fgsm_params["eps"])+"-scaled"


        '''
        adv_wLabel = np.concatenate((adv_x.eval({x:X_test}, session=sess),Y_test),axis=1)
        adv_wLabel = np.delete(adv_wLabel, 78, axis=1)
        print(adv_wLabel.shape)

        #scaler = StandardScaler()
        notScaledAdv = scaler.inverse_transform(adv_x.eval({x:X_test}, session=sess))

        notScaledAdv_wLabel = np.concatenate((notScaledAdv,Y_test),axis=1)
        notScaledAdv_wLabel = np.delete(notScaledAdv_wLabel, 78, axis=1)
        print(notScaledAdv_wLabel.shape)

        if not os.path.exists(outdir):
            os.makedirs(outdir)

        #with sess.as_default(), open("/Users/andreadelvecchio/PycharmProjects/CleverhansVirgin/"+outdir+"/adv_x.txt","w") as file1, open("/Users/andreadelvecchio/PycharmProjects/CleverhansVirgin/"+outdir+"/x_test.txt", "w") as file2:
        
        with sess.as_default(), open("./" + outdir + "/adv_x.txt","w") as file1, open("./" + outdir + "/x_test.txt", "w") as file2, open("./" + outdir + "/adv_wLabelScaled.txt", "w") as file5, open("./" + outdir + "/adv_wLabelNOTScaled.txt", "w") as file6:

            array = adv_x.eval({x: X_test})
            array2 = adv_wLabel
            array3 = notScaledAdv_wLabel
            for i in range(array.shape[0]):
                for j in range(array.shape[1]):
                    file1.write('%s,' % array[i, j])
                    file2.write('%s,' % X_test[i, j])
                    file5.write('%s,' % array2[i, j])
                    file6.write('%s,' % array3[i, j])
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
        ###Scaled test
        adv_x_test = np.concatenate((adv_x.eval({x: X_test}, session=sess), np.zeros(shape=(X_test.shape[0], 1))),
                                    axis=1)
        #####

        # adv_x_test =  adv_x_test[:10000,]
        adversarial_samples = adv_x.eval({x: X_test}, session=sess)

        #####Solo  per DeepFool###### Cancellare per gli altri elementi
        shape = adv_x_test.shape[0]
        i = 0
        while i < shape:
            if (np.isinf(adv_x_test[i]).any() or np.isnan(adv_x_test[i]).any()):
                adv_x_test = np.delete(adv_x_test, obj=i, axis=0)
                adversarial_samples = np.delete(adversarial_samples, obj=i, axis=0)
                X_test = np.delete(X_test, obj=i, axis=0)
                Y_test = np.delete(Y_test, obj=i, axis=0)
                shape = shape - 1
            else:
                i += 1


        clf = decisionTree()  # contiene il calcolo delle metriche per il file MOD-test

        predicition_original = clf.predict(X_test)
        #prediction_test = clf.predict(adv_x.eval({x:X_test}, session=sess))
        prediction_test = clf.predict(adversarial_samples)

        print("*****CALCOLO RISULTATI ORIGINALI*****")
        compute_metrics(Y_test, predicition_original)
        print("*****CALCOLO RISULTATI AVVERSARI*****")
        compute_metrics(Y_test, prediction_test)



        #Collaborative filetring first test
        print("######### Collaborative Filtering Test ###########")
        ####Not Scaled test
        #x_fil = scaler2.inverse_transform(X_FIL)
        #x_fil = np.concatenate((x_fil, Y_FIL), axis=1)
        #######

        ####Scaled test
        x_fil = np.concatenate((X_FIL, Y_FIL), axis=1)
        #######


        #x_fil = x_fil[:10000,]

        #x_train = np.delete(x_train, 78, axis=1)
        print("Generazione x_val eseguita")

        filteringDataframe = pd.DataFrame(x_fil)
        print("Generazione dataFrame eseguito")
        #print(filteringDataframe.head())

        ####Not Scaled test
        #adv_x_test = scaler.inverse_transform(adv_x.eval({x: X_test}, session=sess))
        #adv_x_test = np.concatenate((adv_x_test, np.zeros(shape=(X_test.shape[0], 1))),axis=1)
        ########

        '''
        ###Scaled test
        adv_x_test = np.concatenate((adv_x.eval({x:X_test}, session=sess),np.zeros(shape=(X_test.shape[0],1))),axis=1)
        #####






        #adv_x_test =  adv_x_test[:10000,]
        adversarial_samples = adv_x.eval({x:X_test}, session=sess)

        #####Solo  per DeepFool###### Cancellare per gli altri elementi
        shape = adv_x_test.shape[0]
        i = 0
        while i < shape:
            if (np.isinf(adv_x_test[i]).any() or np.isnan(adv_x_test[i]).any()):
                adv_x_test = np.delete(adv_x_test, obj=i, axis=0)
                adversarial_samples = np.delete(adversarial_samples, obj=i, axis=0)
                X_test = np.delete(X_test, obj=i, axis=0)
                Y_test = np.delete(Y_test, obj=i, axis=0)
                shape = shape - 1
            else:
                i += 1
        '''
        #cosine = cosine_similarity_n_space(filteringDataframe, filteringDataframe, batch_size=100)
        cosine = cosine_similarity(adv_x_test, filteringDataframe)
        print("Calcolo della matrice di similarita' eseguito")
        print(cosine.shape)
        print(str(adv_x_test.shape))
        print(str(adversarial_samples.shape))
        similarity_matrix = pd.DataFrame(cosine, index=range( adv_x_test.shape[0]))
        similarity_matrix.columns = filteringDataframe.index

        #print(similarity_matrix.head())

        sim_matrix_30 = find_n_neighbours(similarity_matrix,30)

        print(str(adv_x.shape[1]))
        #sig_adv = np.empty(shape=(0,adv_x.shape[1]))
        sig_adv = np.empty(shape=(0, adversarial_samples.shape[1]))
        #original_sig_samples = np.empty(shape=(0,adv_x.shape[1]))
        original_sig_samples = np.empty(shape=(0, adversarial_samples.shape[1]))
        #sig_adv_Y = np.empty(shape=(0,77))
        sig_ORIGINAL_Y = np.empty(shape=(0,Y_test.shape[1]))

        print ('Sig_adv init: '+ str(sig_adv.shape))
        label_rec = np.zeros(adv_x_test.shape[0])

        for index in range(adv_x_test.shape[0]):
            score = User_item_score(index, 77, sim_matrix_30, similarity_matrix, filteringDataframe)
            #print("score (u,i) is", score)

            if (score < 0.3):
                label_rec[index] = 0
            elif (score > 0.7):
                label_rec[index] = 1.0
            else:
                label_rec[index] = score

        print("Total sum: ", label_rec.sum(axis=0))

        label_count = abs(label_rec - Y_test[:,0])

        mod = 0
        grey = 0
        for ind in range(label_count.shape[0]):
            if(label_count[ind] != 0):
                if(label_count[ind] == 1):
                    mod += 1
                else:
                    grey += 1
            else:
                sig_adv = np.concatenate((sig_adv, np.array(adversarial_samples[ind]).reshape(1, adversarial_samples.shape[1])),axis=0)
                original_sig_samples = np.concatenate((original_sig_samples, np.array(X_test[ind]).reshape(1, adversarial_samples.shape[1])),axis=0)
                sig_ORIGINAL_Y = np.concatenate((sig_ORIGINAL_Y, np.array(Y_test[ind]).reshape(1, Y_test.shape[1])),axis=0)

        print("% mod: ", mod/label_count.shape[0], "--- ", str(mod))
        print("% grey: ", grey/label_count.shape[0], "--- ", str(grey))


        print('Total not modified: '+str(sig_adv.shape[0]))



        clf = decisionTree() #contiene il calcolo delle metriche per il file MOD-test


        predicition_original = clf.predict(original_sig_samples)
        prediction_test = clf.predict(sig_adv)
        print('Original: '+str(original_sig_samples.shape))
        print('ADV: ' + str(sig_adv.shape))

        #mod = float((abs(prediction_test - sig_ORIGINAL_Y[:, 0]).sum()))

        #mod = mod / float(Y_test.shape[0])

        #print('Accuracy of decision tree: ' + str(float(1 - mod)))
        print("*****CALCOLO RISULTATI ORIGINALI*****")
        compute_metrics(sig_ORIGINAL_Y, predicition_original)
        print("*****CALCOLO RISULTATI AVVERSARI*****")
        compute_metrics(sig_ORIGINAL_Y, prediction_test)


        '''
        ##########
        
        model_2 = make_basic_mlp_non_conv(input_shape=(None, input_shape))
        preds_2 = model_2(x)


        # Perform and evaluate adversarial training
        model_train(sess, x, y, preds_2, X_train, Y_train,
                    evaluate=evaluate,
                    args=train_params, rng=rng, var_list=model_2.get_params())
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc
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
    flags.DEFINE_integer('nb_epochs', 5, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))

    tf.app.run()
