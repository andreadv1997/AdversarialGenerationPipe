from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging


from mlp_test import MultilayerPerceptron #ADDESTRAMENTO NORMALE
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, FastFeatureAdversaries,MadryEtAl,SaliencyMapMethod,MomentumIterativeMethod,SPSA,VirtualAdversarialMethod, FastFeatureAdversaries, DeepFool, CarliniWagnerL2
from cleverhans_tutorials.tutorial_models import make_basic_mlp_non_conv, make_basic_cnn, make_basic_mlp_non_conv2
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans_tutorials.mnist_blackbox import substitute_model

from dataPrep import preprocess, get_data_from_file
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import os

from sklearn.metrics import recall_score
from decisionTree import retrain, decisionTree, compute_metrics as cmDT, adversarialTraining as advTrainDT #ADDESTRAMENTO NORMALE
from randomForest import randomForest, compute_metrics as cmRF, adversarialTraining as advTrainRF #ADDESTRAMENTO NORMALE
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances

base_dir = str(Path().resolve().parent)

dict_General_values = {}
originalFeatures = ['Flow_ID', 'Source_IP', 'Source_Port', 'Destination_IP', 'Destination_Port', 'Protocol', 'Timestamp',
         'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets',
         'Total_Length_of_Bwd_Packets', 'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean',
         'Fwd_Packet_Length_Std', 'Bwd_Packet_Length_Max', 'Bwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean',
         'Bwd_Packet_Length_Std', 'Flow_Bytes/s', 'Flow_Packets/s', 'Flow_IAT_Mean', 'Flow_IAT_Std', 'Flow_IAT_Max',
         'Flow_IAT_Min', 'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max', 'Fwd_IAT_Min', 'Bwd_IAT_Total',
         'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Fwd_PSH_Flags', 'Bwd_PSH_Flags', 'Fwd_URG_Flags',
         'Bwd_URG_Flags', 'Fwd_Header_Length', 'Bwd_Header_Length', 'Fwd_Packets/s', 'Bwd_Packets/s',
         'Min_Packet_Length',
         'Max_Packet_Length', 'Packet_Length_Mean', 'Packet_Length_Std', 'Packet_Length_Variance', 'FIN_Flag_Count',
         'SYN_Flag_Count', 'RST_Flag_Count', 'PSH_Flag_Count', 'ACK_Flag_Count', 'URG_Flag_Count', 'CWE_Flag_Count',
         'ECE_Flag_Count', 'Down/Up_Ratio', 'Average_Packet_Size', 'Avg_Fwd_Segment_Size', 'Avg_Bwd_Segment_Size',
         'Fwd_Avg_Bytes/Bulk', 'Fwd_Avg_Packets/Bulk', 'Fwd_Avg_Bulk_Rate', 'Bwd_Avg_Bytes/Bulk',
         'Bwd_Avg_Packets/Bulk', 'Bwd_Avg_Bulk_Rate', 'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes', 'Subflow_Bwd_Packets',
         'Subflow_Bwd_Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
         'min_seg_size_forward',
         'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min',
         'Label']

originalFeaturesNoLabel = ['Flow_ID', 'Source_IP', 'Source_Port', 'Destination_IP', 'Destination_Port', 'Protocol', 'Timestamp',
         'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets',
         'Total_Length_of_Bwd_Packets', 'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min', 'Fwd_Packet_Length_Mean',
         'Fwd_Packet_Length_Std', 'Bwd_Packet_Length_Max', 'Bwd_Packet_Length_Min', 'Bwd_Packet_Length_Mean',
         'Bwd_Packet_Length_Std', 'Flow_Bytes/s', 'Flow_Packets/s', 'Flow_IAT_Mean', 'Flow_IAT_Std', 'Flow_IAT_Max',
         'Flow_IAT_Min', 'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max', 'Fwd_IAT_Min', 'Bwd_IAT_Total',
         'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Fwd_PSH_Flags', 'Bwd_PSH_Flags', 'Fwd_URG_Flags',
         'Bwd_URG_Flags', 'Fwd_Header_Length', 'Bwd_Header_Length', 'Fwd_Packets/s', 'Bwd_Packets/s',
         'Min_Packet_Length',
         'Max_Packet_Length', 'Packet_Length_Mean', 'Packet_Length_Std', 'Packet_Length_Variance', 'FIN_Flag_Count',
         'SYN_Flag_Count', 'RST_Flag_Count', 'PSH_Flag_Count', 'ACK_Flag_Count', 'URG_Flag_Count', 'CWE_Flag_Count',
         'ECE_Flag_Count', 'Down/Up_Ratio', 'Average_Packet_Size', 'Avg_Fwd_Segment_Size', 'Avg_Bwd_Segment_Size',
         'Fwd_Avg_Bytes/Bulk', 'Fwd_Avg_Packets/Bulk', 'Fwd_Avg_Bulk_Rate', 'Bwd_Avg_Bytes/Bulk',
         'Bwd_Avg_Packets/Bulk', 'Bwd_Avg_Bulk_Rate', 'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes', 'Subflow_Bwd_Packets',
         'Subflow_Bwd_Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
         'min_seg_size_forward',
         'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min']

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

featuresNoLabel = ['Protocol', 'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets',
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
            'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min']

features_save_dataframe = ['Protocol', 'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets',
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
            'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min', 'Label','Computed_Label']

features_for_scaling = ['Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets',
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
            'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min']

int_features = ['Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets',
            'Total_Length_of_Bwd_Packets', 'Fwd_Packet_Length_Max', 'Fwd_Packet_Length_Min',
            'Bwd_Packet_Length_Max', 'Bwd_Packet_Length_Min',
            'Flow_IAT_Max',
            'Flow_IAT_Min', 'Fwd_IAT_Total', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max', 'Fwd_IAT_Min',
            'Bwd_IAT_Total',
            'Bwd_IAT_Max', 'Bwd_IAT_Min', 'Fwd_PSH_Flags', 'Bwd_PSH_Flags',
            'Fwd_URG_Flags',
            'Bwd_URG_Flags', 'Fwd_Header_Length', 'Bwd_Header_Length',
            'Min_Packet_Length',
            'Max_Packet_Length', 'FIN_Flag_Count',
            'SYN_Flag_Count', 'RST_Flag_Count', 'PSH_Flag_Count', 'ACK_Flag_Count', 'URG_Flag_Count', 'CWE_Flag_Count',
            'ECE_Flag_Count',
            'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes',
            'Subflow_Bwd_Packets',
            'Subflow_Bwd_Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
            'min_seg_size_forward', 'Active_Max', 'Active_Min', 'Idle_Max', 'Idle_Min']

#to_update_feature = [ ('Flow_Bytes/s',4,5,1) , ('Flow_Packets/s',2,3,1), ('Fwd_Packets/s',2,1), ('Bwd_Packets/s',3,1)]
to_update_feature = [ ('Flow_Bytes/s','Total_Length_of_Fwd_Packets','Total_Length_of_Bwd_Packets','Flow_Duration'),
                      ('Flow_Packets/s','Total_Fwd_Packets', 'Total_Backward_Packets','Flow_Duration'),
                      ('Fwd_Packets/s','Total_Fwd_Packets','Flow_Duration'),
                      ('Bwd_Packets/s','Total_Backward_Packets','Flow_Duration'),
                      ('Fwd_Packet_Length_Mean','Total_Length_of_Fwd_Packets','Total_Fwd_Packets'),
                      ('Bwd_Packet_Length_Mean','Total_Length_of_Bwd_Packets','Total_Backward_Packets'),
                      ('Flow_IAT_Mean','Fwd_IAT_Total','Bwd_IAT_Total','Total_Fwd_Packets','Total_Backward_Packets'),
                      ('Flow_IAT_Max','Fwd_IAT_Max','Bwd_IAT_Max'),
                      ('Flow_IAT_Min','Fwd_IAT_Min','Bwd_IAT_Min'),
                      ('Fwd_IAT_Mean','Fwd_IAT_Total','Total_Fwd_Packets'),
                      ('Bwd_IAT_Mean','Bwd_IAT_Total','Total_Backward_Packets'),
                      ('Min_Packet_Length','Fwd_Packet_Length_Min','Bwd_Packet_Length_Min'),
                      ('Max_Packet_Length','Fwd_Packet_Length_Max','Bwd_Packet_Length_Max'),
                      ('Packet_Length_Mean','Total_Length_of_Fwd_Packets','Total_Length_of_Bwd_Packets','Total_Fwd_Packets','Total_Backward_Packets'),
                      ('Packet_Length_Std','Packet_Length_Variance'),
                      ('PSH_Flag_Count','Fwd_PSH_Flags','Bwd_PSH_Flags'),
                      ('URG_Flag_Count','Fwd_URG_Flags','Bwd_URG_Flags')]

#algorithm_under_test = ""

def find_n_neighbours(df,n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:n].index,index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    # il dataset restituito e' quello contenente gli indici dei valori top n
    return df

def User_item_score(user,item, sim_user_n, similarity_matrix, oracle, n):
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
    nume = fin['adg_score'].sum()

    deno = n

    final_score = (nume / deno)
    return final_score

def generate_adversarial_samples(X_train=None, Y_train=None, X_test=None, Y_test=None, originalTestDateset=None, protocol_test=None, test_start=0, test_end=2092, nb_epochs=5, batch_size=128, learning_rate=0.01, testing=False, num_threads=None, adv_params=None, algorithm=None):
    """
    Generate Adversarial Samples
    :param X_train: set of samples used to train surrogate model
    :param Y_train: labels of X_train
    :param X_test: set of samples used to generate adversarial samples
    :param Y_test: labels of X_test
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param testing: if true, complete an AccuracyReport for unit tests to verify that performance is adequate
    :param clean_train: if true, train on clean examples
    :param num_threads: number of threads for training VALUE TO BE CHECKED
    :param adv_params: dictionary containing values adopted for training
    :param algorithm: adversarial algorithm
    :return: dataset containing adversarial, scaled and not scaled, with mock Protocol feature, array of original labels, scaler adopted for conversion, original dataset as read from file, final raw adversarial samples
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
    advScaler = MaxAbsScaler()

    test_end = X_test.shape[0]



    advScaler = advScaler.fit(X_train)
    scaled_X_train = advScaler.transform(X_train)
    scaled_X_test = advScaler.transform(X_test)

    #cancellazione della colonna "Protocol"
    X_train_adv_mlp = np.delete(scaled_X_train, 0, 1)
    X_test_adv_mlp = np.delete(scaled_X_test, 0, 1)


    input_shape = X_train_adv_mlp.shape[1]
    model = make_basic_mlp_non_conv(input_shape=(None, input_shape))

    algorithms = {'FastGradientSignMethod': FastGradientMethod(model,sess=sess), 'SaliencyMapMethod': SaliencyMapMethod(model, sess=sess), 'DeepFool': DeepFool(model, sess=sess), 'VirtualAdversarialMethod': VirtualAdversarialMethod(model, sess=sess)}



    # Use label smoothing
    print("Dimensione dell'input Train: "+str(X_train_adv_mlp.shape)+"\n")
    print("Dimensione dell'output Train: " + str(Y_train.shape) + "\n")

    print("Dimensione dell'input Test: " + str(X_test_adv_mlp.shape) + "\n")
    print("Dimensione dell'output Test: " + str(Y_test.shape) + "\n")

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, input_shape))
    y = tf.placeholder(tf.float32, shape=(None, 2))




    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    print(nb_epochs)
    print(batch_size)
    print(learning_rate)


    ##NB RIESEGUIRE TRANING ADV DEEPFOOL PER RIGENERARE FILE ALTERATO

    #state of art test
    fgsm_params=adv_params
    #fgsm_params = {'clip_min': 0.0, 'clip_max': 1.1, 'theta': 0.7, 'gamma': 0.7}  # JSMA
    #fgsm_params = {'clip_min': 0.0, 'clip_max': 1.1, 'eps': 0.01}  # FGSM
    #fgsm_params = {'clip_min': 0.0, 'clip_max': 1.1, 'nb_candidate': 2, 'nb_classes': 2, 'overshoot': 0.1, 'max_iter': 1}  # DeepFool
    #fgsm_params = {'clip_min': 0.0, 'clip_max': 1.1, 'eps': 0.5, 'xi': 0.1, 'num_iterations': 15}  # VirtualADV
    #fgsm_params = {'clip_min': 0.0, 'clip_max': 1.1, 'confidence': 0.5, 'learning_rate': 0.1, 'batch_size': 1,'binary_search_steps': 10, 'max_iterations': 5, 'abort_early': False,'initial_const': 1}  # CW

    rng = np.random.RandomState([2017, 8, 30]) #riproducibilita'


    preds = model.get_probs(x) #placeholder

    def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc,prec,recall,f1 = model_eval(
                sess, x, y, preds, X_test_adv_mlp, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)
            print('Test precision on legitimate examples: %0.4f' % prec)
            print('Test recall on legitimate examples: %0.4f' % recall)
            print('Test f1 on legitimate examples: %0.4f' % f1)
    model_train(sess, x, y, preds, X_train_adv_mlp, Y_train, evaluate=evaluate, args=train_params, rng=rng, var_list=model.get_params())
        # Calculate training error
    '''
    eval_params = {'batch_size': batch_size}
    acc,prec,recall,f1 = model_eval(sess, x, y, preds, X_train_adv_mlp, Y_train, args=eval_params)
    report.train_clean_train_clean_eval = acc
    '''
    # Initialize the Fast Gradient Sign Method (FGSM) attack object and
    # graph
    fgsm= algorithms[algorithm]
    #fgsm = FastGradientMethod(model, sess=sess)
    #fgsm = SaliencyMapMethod(model, sess=sess)
    #fgsm = DeepFool(model, sess=sess)
    #fgsm = CarliniWagnerL2(model, sess=sess)
    #fgsm = VirtualAdversarialMethod(model, sess=sess)

    adv_x = fgsm.generate(x, **fgsm_params)
    preds_adv = model.get_probs(adv_x)

    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size}
    acc,prec,recall,f1 = model_eval(sess, x, y, preds_adv, X_test_adv_mlp, Y_test, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f' % acc)
    print('Test precision on legitimate examples: %0.4f' % prec)
    print('Test recall on legitimate examples: %0.4f' % recall)
    print('Test f1 on legitimate examples: %0.4f\n' % f1)

    report.clean_train_adv_eval = acc

    #global
    dict_General_values['MLPAccuracy'] = acc
    dict_General_values['MLPPrecision']=prec
    dict_General_values['MLPRecall']=recall
    dict_General_values['MLPF1']=f1



        # Calculate training error
    if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train_adv_mlp,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc


    ##COLLABORATIVE FILTERING CON 77 feature

    adversarial_s = adv_x.eval({x:X_test_adv_mlp}, session=sess)
    original_Y_Test = np.copy(Y_test)
    #SOLO PER DEEP FOOL
    shape = adversarial_s.shape[0]
    i = 0
    print("ADV s shape pre sanity")
    print(adversarial_s.shape)
    while i < shape:
        if (np.isinf(adversarial_s[i]).any() or np.isnan(adversarial_s[i]).any()):
            adversarial_s = np.delete(adversarial_s, obj=i, axis=0)
            X_test = np.delete(X_test, obj=i, axis=0)
            Y_test = np.delete(Y_test, obj=i, axis=0)
            protocol_test = np.delete(protocol_test, obj=i, axis=0)  # riaggiorna dimensione dell'array di protocol per DeepFool
            originalTestDateset.drop(originalTestDateset.index[i], inplace=True)  ###
            shape = shape - 1
        else:
            i += 1

    print("ADV s shape pre sanity -2")
    print(adversarial_s.shape)
    print(originalTestDateset.shape)
    originalTestDateset.reset_index(drop=True, inplace=True)

    adversarial_s = np.concatenate((np.zeros(shape=(adversarial_s.shape[0], 1)), adversarial_s),axis=1)
    not_scaled_samples = advScaler.inverse_transform(adversarial_s)  # descaling

    protocol_test = protocol_test.reshape(protocol_test.shape[0], 1)
    not_scaled_samples[:, 0] = protocol_test[:, 0]

    # nuova implementazione
    rawCopy = originalTestDateset.copy()
    rawCopy[features_for_scaling] = not_scaled_samples[:, 1:]

    rawCopy.to_csv(base_dir + "/updatedOutput/adv_RAW/" + algorithm+".csv", header=originalFeatures, index=False)


    return adversarial_s, not_scaled_samples, X_test, Y_test, advScaler, originalTestDateset, rawCopy, protocol_test #qui vengono restituiti sia i valori non scalati che quelli scalati (sia completi di Protocollo che no)


def get_sanitize_adversarial_samples(not_scaled_items=None, original_samples= None, original_labels= None, protocol_test=None, originalTestDataset=None):
    """
       Execute Sanity Check on a set of raw adversarial
       :param raw_adversarial_samples: dataset contaning adversarial samples
       :param original_samples: dataset containing original samples
       :param original_labels: array contaning the labels
       :param protocol_test: array containing original values for "Protocol" value
       :param originalTestDataset: original dataset of starting samples
       :return: dataset containing non scaled representative adversarial samples, corresponding original samples and related labels
    """

    # QUI I SAMPLE SONO NON SCALATI
    adversarial_samples, original_samples, original_labels, protocol_test = sannity_check_adv(not_scaled_items, protocol_test, original_samples, original_labels, originalTestDataset)
    print(adversarial_samples.shape)

    return adversarial_samples, original_samples, original_labels, protocol_test


def get_representative_samples(X_FIL=None, Y_FIL=None, X_test=None, Y_test=None, adversarial_samples=None, adv_file_name=None,originalTestDataset=None):
    """
        Execute Sanity Check on a set of raw adversarial
        :param X_FIL: oracle used for collaborative filtering
        :param Y_FIL: label used for collaborative filtering
        :param X_Test: dataset containing original samples
        :param Y_Test: original labels
        :param adversarial_samples: representative adversarial samples
        :param adv_file_name: string containing the name of the file where the results will be saved
        :param originalTestDataset: original dataset of starting samples
        :return: dataset containing non scaled sanitized and representative adversarial samples, corresponding original samples and related labels
     """
    #CALCOLO ESEMPI SIGNIFICATIVI
    #QUI I DATI SONO NON SCALATI
    sig_adv, original_sig_samples, sig_ORIGINAL_Y, non_rep, index_sig, index_non_rep = collaborative_filtering(X_FIL, Y_FIL, X_test, Y_test, adversarial_samples,adv_file_name,originalTestDataset)

    return sig_adv, original_sig_samples, sig_ORIGINAL_Y, non_rep, index_sig, index_non_rep










def collaborative_filtering(x_fil, y_fil, X_test, Y_test, adversarial_samples, file_name, originalData):
    print("######### Collaborative Filtering Test ###########")

    print("Generazione x_val eseguita")

    filteringDataframe = pd.DataFrame(x_fil)
    oracle = pd.DataFrame(np.concatenate((x_fil, y_fil), axis=1))
    print("Generazione dataFrame eseguito")

    cosine = cosine_similarity(adversarial_samples, filteringDataframe)
    #cosine = euclidean_distances(adversarial_samples, filteringDataframe)
    print("Calcolo della matrice di similarita' eseguito")
    print(cosine.shape)
    print(str(adversarial_samples.shape))
    similarity_matrix = pd.DataFrame(cosine, index=range(adversarial_samples.shape[0]))
    similarity_matrix.columns = filteringDataframe.index

    # print(similarity_matrix.head())

    #sim_matrix_30 = find_n_neighbours(similarity_matrix, 30)
    sim_matrix_30 = find_n_neighbours(similarity_matrix, 15)

    #print(str(adv_x.shape[1]))
    sig_adv = np.empty(shape=(0, adversarial_samples.shape[1]))
    original_sig_samples = np.empty(shape=(0, adversarial_samples.shape[1]))
    sig_ORIGINAL_Y = np.empty(shape=(0, Y_test.shape[1]))
    index_sig_adv=[]

    nonRep_adv = np.empty(shape=(0, adversarial_samples.shape[1]))
    nonRep_ORIGINAL_Y = np.empty(shape=(0, Y_test.shape[1]))
    original_nonRep_samples = np.empty(shape=(0, adversarial_samples.shape[1]))
    label_computer_nonRep = []
    index_non_rep = []  ####


    grey_adv = np.empty(shape=(0, adversarial_samples.shape[1]))
    grey_ORIGINAL_Y = np.empty(shape=(0, Y_test.shape[1]))
    original_grey_samples = np.empty(shape=(0, adversarial_samples.shape[1]))
    label_computer_grey=[]
    index_grey = [] ####

    adv_orig_BENIGN = np.empty(shape=(0, adversarial_samples.shape[1]))
    adv_orig_BEN_ORIGINAL_Y = np.empty(shape=(0, Y_test.shape[1]))
    label_computed_orig_BENIGN = []
    index_orig_BENIGN = []

    adv_orig_ATTACK = np.empty(shape=(0, adversarial_samples.shape[1]))
    adv_orig_ATTACK_ORIGINAL_Y = np.empty(shape=(0, Y_test.shape[1]))
    label_computed_orig_ATTACK = []
    index_orig_ATTACK = []

    print('Sig_adv init: ' + str(sig_adv.shape))
    label_rec = np.zeros(adversarial_samples.shape[0])

    dict_General_values['total'] = adversarial_samples.shape[0]
    for index in range(adversarial_samples.shape[0]):

        #score = User_item_score(index, 77, sim_matrix_30, similarity_matrix, oracle, 30)
        score = User_item_score(index, 77, sim_matrix_30, similarity_matrix, oracle, 15)

        #thresholds updated to 0.3,0.7
        if (score < 0.3):
            label_rec[index] = 0
        elif (score > 0.7):
            label_rec[index] = 1.0
        else:
            label_rec[index] = score

    print("Total sum: ", label_rec.sum(axis=0))

    label_count = abs(label_rec - Y_test[:, 0])

    mod = 0
    grey = 0
    originally_BENIGN_nr = 0
    originally_ATTACK_nr = 0

    #QUI AVVIENE LA DECOMPOSIZIONE DEI FILE. QUINDI QUI ANDREBBE ANCHE LA STRUTTURAZIONE E IL COPIA E INCOLLA DELLE FEATURE FILTRATE A MONTE
    for ind in range(label_count.shape[0]):
        if (label_count[ind] != 0):
            if (label_count[ind] == 1):
                mod += 1
                nonRep_adv = np.concatenate((nonRep_adv, np.array(adversarial_samples[ind]).reshape(1, adversarial_samples.shape[1])), axis=0)
                original_nonRep_samples = np.concatenate((original_nonRep_samples, np.array(X_test[ind]).reshape(1, adversarial_samples.shape[1])), axis=0)
                nonRep_ORIGINAL_Y = np.concatenate((nonRep_ORIGINAL_Y, np.array(Y_test[ind]).reshape(1, Y_test.shape[1])),axis=0)
                label_computer_nonRep.append(label_rec[ind])
                index_non_rep.append(ind)#######
                if(Y_test[ind,0]== 1):
                    originally_BENIGN_nr+=1
                    adv_orig_BENIGN = np.concatenate((adv_orig_BENIGN, np.array(adversarial_samples[ind]).reshape(1, adversarial_samples.shape[1])), axis=0)
                    adv_orig_BEN_ORIGINAL_Y = np.concatenate((adv_orig_BEN_ORIGINAL_Y, np.array(Y_test[ind]).reshape(1, Y_test.shape[1])),axis=0)
                    label_computed_orig_BENIGN.append(label_rec[ind])
                    index_orig_BENIGN.append(ind)#####
                else:
                    originally_ATTACK_nr+=1
                    adv_orig_ATTACK = np.concatenate((adv_orig_ATTACK, np.array(adversarial_samples[ind]).reshape(1, adversarial_samples.shape[1])),axis=0)
                    adv_orig_ATTACK_ORIGINAL_Y = np.concatenate((adv_orig_ATTACK_ORIGINAL_Y, np.array(Y_test[ind]).reshape(1, Y_test.shape[1])), axis=0)
                    label_computed_orig_ATTACK.append(label_rec[ind])
                    index_orig_ATTACK.append(ind)######

            else:
                grey += 1
                grey_adv = np.concatenate((grey_adv, np.array(adversarial_samples[ind]).reshape(1, adversarial_samples.shape[1])), axis=0)
                original_grey_samples = np.concatenate((original_grey_samples, np.array(X_test[ind]).reshape(1, adversarial_samples.shape[1])), axis=0)
                grey_ORIGINAL_Y = np.concatenate((grey_ORIGINAL_Y, np.array(Y_test[ind]).reshape(1, Y_test.shape[1])), axis=0)
                label_computer_grey.append(label_rec[ind])
                index_grey.append(ind)
        else:
            sig_adv = np.concatenate((sig_adv, np.array(adversarial_samples[ind]).reshape(1, adversarial_samples.shape[1])), axis=0)
            original_sig_samples = np.concatenate((original_sig_samples, np.array(X_test[ind]).reshape(1, adversarial_samples.shape[1])), axis=0)
            sig_ORIGINAL_Y = np.concatenate((sig_ORIGINAL_Y, np.array(Y_test[ind]).reshape(1, Y_test.shape[1])), axis=0)
            index_sig_adv.append(ind)

    print("% mod: ", mod / label_count.shape[0], "--- ", str(mod))
    print("% grey: ", grey / label_count.shape[0], "--- ", str(grey))

    dict_General_values['mod'] = mod
    dict_General_values['grey'] = grey
    dict_General_values['not_mod'] = sig_adv.shape[0]
    dict_General_values['originally_BENIGN_nr']=originally_BENIGN_nr
    dict_General_values['originally_ATTACK_nr'] = originally_ATTACK_nr



    print('Total not modified: ' + str(sig_adv.shape[0]))
    print(str(original_nonRep_samples.shape))



    dataframe_orig_BEN = np.concatenate((adv_orig_BENIGN,adv_orig_BEN_ORIGINAL_Y[:,0].reshape(adv_orig_BEN_ORIGINAL_Y.shape[0],1)),axis=1)
    dataframe_orig_BEN = np.concatenate((dataframe_orig_BEN, np.asarray(label_computed_orig_BENIGN).reshape(adv_orig_BEN_ORIGINAL_Y.shape[0],1)), axis=1)
    print("BENING :"+ str(dataframe_orig_BEN.shape))
    save_on_file_NR(dataframe_orig_BEN,file_name, "BENIGN", originalData, index_orig_BENIGN)

    dataframe_orig_ATTACK = np.concatenate((adv_orig_ATTACK, adv_orig_ATTACK_ORIGINAL_Y[:,0].reshape(adv_orig_ATTACK_ORIGINAL_Y.shape[0],1)), axis=1)
    dataframe_orig_ATTACK = np.concatenate((dataframe_orig_ATTACK, np.asarray(label_computed_orig_ATTACK).reshape(adv_orig_ATTACK_ORIGINAL_Y.shape[0],1)), axis=1)

    print("ATTACK :" + str(dataframe_orig_ATTACK.shape))
    save_on_file_NR(dataframe_orig_ATTACK, file_name, "ATTACK", originalData, index_orig_ATTACK)

    dataframe_orig_GREY = np.concatenate((grey_adv, grey_ORIGINAL_Y[:,0].reshape(grey_ORIGINAL_Y.shape[0],1)), axis=1)
    dataframe_orig_GREY = np.concatenate((dataframe_orig_GREY, np.asarray(label_computer_grey).reshape(grey_ORIGINAL_Y.shape[0],1)), axis=1)

    save_on_file_NR(dataframe_orig_GREY, file_name, "GREY", originalData, index_grey)
    print("GREY :" + str(dataframe_orig_GREY.shape))

    tempCopy = originalData.iloc[index_sig_adv].copy()
    dataframe = pd.DataFrame(sig_adv, columns=featuresNoLabel)
    tempCopy[featuresNoLabel] = dataframe[featuresNoLabel].values
    tempCopy.to_csv(base_dir + "/updatedOutput/adv_representative/" + file_name+".csv", header=originalFeatures,index=False)


    return sig_adv, original_sig_samples, sig_ORIGINAL_Y, nonRep_adv, index_sig_adv, index_non_rep

def save_on_file_NR(data, fileName, type, originalData, list_of_indexes): #VERIFICARE SE CONVIENE INCOLLARE LE FEATURE QUI O NEL METODO CHIAMANTE
    base_dir = str(Path().resolve().parent)
    tempCopy = originalData.iloc[list_of_indexes].copy() ##verificare funzione assign()

    dataframe = pd.DataFrame(data, columns=features_save_dataframe)

    tempCopy[featuresNoLabel] = dataframe[featuresNoLabel].values
    tempCopy.insert(len(originalFeatures), "Computed Label", dataframe["Computed_Label"].values)



    if(type =="BENIGN"):
        #dataframe.to_csv(base_dir+"/ADV_OUTPUT/adv_ORIG_BEN/"+fileName, header=features_save_dataframe, index=False)
        tempCopy.to_csv(base_dir + "/updatedOutput/adv_ORIG_BEN/" + fileName+".csv", header=originalFeatures+['Computed_Label'], index=False)
        print(tempCopy.shape)
        print(dataframe.shape)
    elif (type == "ATTACK"):
        tempCopy.to_csv(base_dir + "/updatedOutput/adv_ORIG_ATT/" + fileName+".csv", header=originalFeatures+['Computed_Label'], index=False)
        print(tempCopy.shape)
        print(dataframe.shape)
    else:
        #dataframe.to_csv(base_dir + "/ADV_OUTPUT/adv_grey/" + fileName, header=features_save_dataframe+['Computed_Label'], index=False)
        tempCopy.to_csv(base_dir + "/updatedOutput/adv_grey/" + fileName+".csv", header=originalFeatures + ['Computed_Label'], index=False)



def save_on_file(x_fil, nonRep):
    base_dir = str(Path().resolve())


    filtering_dataframe = pd.DataFrame(x_fil, columns=features)
    nonRep_dataframe = pd.DataFrame(nonRep, columns=features)

    nonRep_dataframe['Label'] = np.where(nonRep_dataframe['Label']==0,2, 3)


    total_dataframe = pd.concat([filtering_dataframe,nonRep_dataframe], axis=0)
    total_dataframe.index = range(0,filtering_dataframe.shape[0]+nonRep_dataframe.shape[0])

    total_dataframe.to_csv(base_dir+"/pcaGARR.csv", header=features, index=False)

def printResults(dictDT, dictRF, dictMLP, headerString=""):

    print(headerString)
    print('Accuracy of decision tree: ' + str(dictDT['accuracy']))
    print('Accuracy of random forest: ' + str(dictRF['accuracy']))
    print('Accuracy of MLP: ' + str(dictMLP['accuracy']))
    print()
    print('Precision of decision tree: ' + str(dictDT['precision']))
    print('Precision of random forest: ' + str(dictRF['precision']))
    print('Precision of MLP: ' + str(dictMLP['precision']))
    print()
    print('Recall of decision tree: ' + str(dictDT['recall']))
    print('Recall of random forest: ' + str(dictRF['recall']))
    print('Recall of MLP: ' + str(dictMLP['recall']))
    print()
    print('F1 of decision tree: ' + str(dictDT['f1']))
    print('F1 of random forest: ' + str(dictRF['f1']))
    print('F1 of MLP: ' + str(dictMLP['f1']))

def sannity_check_adv(not_scaled_samples, protocol_test, X_test, Y_test, originalData):
    '''
    adversarial_samples = np.concatenate((np.zeros(shape=(adversarial_samples.shape[0],1)), adversarial_samples), axis=1)
    not_scaled_samples = scaler.inverse_transform(adversarial_samples) #descaling

    # nuova implementazione
    rawCopy = originalData.copy()
    rawCopy[features_for_scaling] = not_scaled_samples[:,1:]

    rawCopy.to_csv(base_dir + "/updatedOutput/adv_RAW/" + file_name, header=originalFeatures, index=False)
    '''

    # qui vengono copiati i dati prima del sanity check. IN QUESTO PUNTO ANDREBBERO INSERIITE LE FEATURE CANCELLATE CHE ERANO ANDATE PERSE ALLA PRIMA LETTURA

    #vecchia implementazione
    #pd.DataFrame(not_scaled_to_copy, columns=features).to_csv(base_dir+"/ADV_OUTPUT/adv_RAW/"+file_name, header=features,index=False)

    print("RAW: " + str(not_scaled_samples.shape))


    list_of_int_index = []
    for feature in int_features:
        list_of_int_index.append(features.index(feature))

    for index in list_of_int_index:
        not_scaled_samples[:,index] = np.round(not_scaled_samples[:,index])

    for index_1 in range(0, len(to_update_feature)):
        feat = to_update_feature[index_1][0]
        '''
        if(len(to_update_feature[index_1]) == 4):
            #not_scaled_samples[:,features.index(to_update_feature[index_1][0])] = (not_scaled_samples[:,to_update_feature[index_1][1]] + not_scaled_samples[:,to_update_feature[index_1][2]]) / not_scaled_samples[:,to_update_feature[index_1][3]] * 1000
            not_scaled_samples[:, features.index(to_update_feature[index_1][0])] = (not_scaled_samples[:,features.index(to_update_feature[index_1][1])] + not_scaled_samples[:,features.index(to_update_feature[index_1][2])]) / not_scaled_samples[:,features.index(to_update_feature[index_1][3])] * 1000000

        if (len(to_update_feature[index_1]) == 3):
            #not_scaled_samples[:, features.index(to_update_feature[index_1][0])] = (not_scaled_samples[:,to_update_feature[index_1][1]]) / not_scaled_samples[:,to_update_feature[ index_1][2]] * 1000
            not_scaled_samples[:, features.index(to_update_feature[index_1][0])] = (not_scaled_samples[:,features.index(to_update_feature[index_1][1])]) / not_scaled_samples[:, features.index(to_update_feature[index_1][2])] * 1000000
        '''
        if feat == 'Fwd_Packet_Length_Mean':
            not_scaled_samples[:, features.index(feat)]=not_scaled_samples[:, features.index('Total_Length_of_Fwd_Packets')]/not_scaled_samples[:, features.index('Total_Fwd_Packets')]
        elif feat == 'Bwd_Packet_Length_Mean':
            not_scaled_samples[:, features.index(feat)] = not_scaled_samples[:, features.index('Total_Length_of_Bwd_Packets')] / not_scaled_samples[:, features.index('Total_Backward_Packets')]
        elif feat == 'Flow_Bytes/s':
            not_scaled_samples[:, features.index(feat)] = ((not_scaled_samples[:,features.index('Total_Length_of_Fwd_Packets')]) + (not_scaled_samples[:,features.index('Total_Length_of_Bwd_Packets')]))/ not_scaled_samples[:, features.index('Flow_Duration')] * 1000000
        elif feat == 'Flow_Packets/s':
            not_scaled_samples[:, features.index(feat)] = ((not_scaled_samples[:,features.index('Total_Fwd_Packets')]) + (not_scaled_samples[:, features.index('Total_Backward_Packets')])) / not_scaled_samples[:,features.index('Flow_Duration')] * 1000000
        #elif feat == 'Flow_IAT_Mean':
        #    not_scaled_samples[:, features.index(feat)] = ((not_scaled_samples[:,features.index('Fwd_IAT_Total')]) + (not_scaled_samples[:, features.index('Bwd_IAT_Total')])) / (not_scaled_samples[:,features.index('Total_Fwd_Packets')]+ not_scaled_samples[:,features.index('Total_Backward_Packets')]-2)
        #elif feat == 'Flow_IAT_Max':
        #    not_scaled_samples[:, features.index(feat)] = np.maximum(not_scaled_samples[:, features.index('Fwd_IAT_Max')],not_scaled_samples[:, features.index('Bwd_IAT_Max')])
        #elif feat == 'Flow_IAT_Min':
        #    not_scaled_samples[:, features.index(feat)] = np.minimum(not_scaled_samples[:, features.index('Fwd_IAT_Min')],not_scaled_samples[:, features.index('Bwd_IAT_Min')])
        elif feat == 'Fwd_IAT_Mean':
            not_scaled_samples[:, features.index(feat)] = not_scaled_samples[:, features.index('Fwd_IAT_Total')] / (not_scaled_samples[:, features.index('Total_Fwd_Packets')]-1)
        elif feat == 'Bwd_IAT_Mean':
            not_scaled_samples[:, features.index(feat)] = not_scaled_samples[:, features.index('Bwd_IAT_Total')] / (not_scaled_samples[:, features.index('Total_Backward_Packets')] - 1)
        elif feat == 'Fwd_Packets/s':
            not_scaled_samples[:, features.index(feat)] = (not_scaled_samples[:,features.index('Total_Fwd_Packets')]) / not_scaled_samples[:,features.index('Flow_Duration')] * 1000000
        elif feat == 'Bwd_Packets/s':
            not_scaled_samples[:, features.index(feat)] = (not_scaled_samples[:,features.index('Total_Backward_Packets')]) / not_scaled_samples[:,features.index('Flow_Duration')] * 1000000
        elif feat == 'Min_Packet_Length':
            not_scaled_samples[:, features.index(feat)] = np.minimum(not_scaled_samples[:, features.index('Fwd_Packet_Length_Min')],not_scaled_samples[:, features.index('Bwd_Packet_Length_Min')])
        elif feat == 'Max_Packet_Length':
            not_scaled_samples[:, features.index(feat)] = np.maximum(not_scaled_samples[:, features.index('Fwd_Packet_Length_Max')],not_scaled_samples[:, features.index('Bwd_Packet_Length_Max')])
        elif feat == 'Packet_Length_Mean':
            not_scaled_samples[:, features.index(feat)] = ((not_scaled_samples[:, features.index('Total_Length_of_Fwd_Packets')]) + (not_scaled_samples[:, features.index('Total_Length_of_Bwd_Packets')])) / (not_scaled_samples[:, features.index('Total_Fwd_Packets')] + not_scaled_samples[:, features.index('Total_Backward_Packets')])
        elif feat == 'Packet_Length_Std':
            not_scaled_samples[:, features.index(feat)] = np.sqrt(not_scaled_samples[:, features.index('Packet_Length_Variance')])
        elif feat == 'PSH_Flag_Count':
            not_scaled_samples[:, features.index(feat)] = not_scaled_samples[:, features.index('Fwd_PSH_Flags')] + not_scaled_samples[:, features.index('Bwd_PSH_Flags')]
        elif feat == 'URG_Flag_Count':
            not_scaled_samples[:, features.index(feat)] = not_scaled_samples[:, features.index('Fwd_URG_Flags')] + not_scaled_samples[:, features.index('Bwd_URG_Flags')]

        #duplicate features
        not_scaled_samples[:, features.index('Average_Packet_Size')] = not_scaled_samples[:, features.index('Packet_Length_Mean')]
        not_scaled_samples[:, features.index('Avg_Fwd_Segment_Size')] = not_scaled_samples[:,features.index('Fwd_Packet_Length_Mean')]
        not_scaled_samples[:, features.index('Avg_Bwd_Segment_Size')] = not_scaled_samples[:,features.index('Bwd_Packet_Length_Mean')]


    shape = not_scaled_samples.shape[0]

    i = 0
    while i < shape:
         if (np.isinf(not_scaled_samples[i]).any() or np.isnan(not_scaled_samples[i]).any()):
              not_scaled_samples = np.delete(not_scaled_samples, obj=i, axis=0)
              X_test = np.delete(X_test, obj=i, axis=0)
              Y_test = np.delete(Y_test, obj=i, axis=0)
              protocol_test = np.delete(protocol_test, obj= i, axis=0)
              originalData.drop(originalData.index[i], inplace=True) #########
              shape = shape - 1
         else:
            i += 1
    originalData.reset_index(drop=True, inplace=True)




    return not_scaled_samples, X_test, Y_test, protocol_test

def write_Results_On_File(fileName,dict_adv_raw, dict_X_Test,dict_Adv_Not_Filtered, dict_Original_Filtered, dict_Adv_Filtered):

    outputDirTXT = base_dir + '/updatedOutput/outputTXT_FILTERED_NotScaled'
    outputDirCSV = base_dir + '/updatedOutput/outputCSV_FILTERED_NotScaled'

    if (not os.path.exists(outputDirTXT)):
        os.mkdir(outputDirTXT)
    if (not os.path.exists(outputDirCSV)):
        os.mkdir(outputDirCSV)

    txtFile = outputDirTXT + '/' + fileName + '_Results.txt'
    csvFile = outputDirCSV + '/' + fileName + '_Results.csv'

    with open(txtFile, "w") as file1, open(csvFile,"w") as file2 :

        file1.write('Metric\t\tAdv Raw\t\tOriginal Test Set (X_test related to Sanitized)\t\tNot Filtered Adversarial Samples(adv_x sanitized)\t\tOriginal Test after filtering(original_sig_samples related to representative)\t\tFiltered Adversarial Samples(sig_adv_sample representative adv)\n')
        file1.write('Accuracy\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\n' % (str(dict_adv_raw['accuracy']),str(dict_X_Test['accuracy']), str(dict_Adv_Not_Filtered['accuracy']), str(dict_Original_Filtered['accuracy']), str(dict_Adv_Filtered['accuracy'])))
        file1.write('Precision\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\n' % (str(dict_adv_raw['precision']),str(dict_X_Test['precision']), str(dict_Adv_Not_Filtered['precision']), str(dict_Original_Filtered['precision']), str(dict_Adv_Filtered['precision'])))
        file1.write('Recall\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s\n' % (str(dict_adv_raw['recall']),str(dict_X_Test['recall']), str(dict_Adv_Not_Filtered['recall']), str(dict_Original_Filtered['recall']), str(dict_Adv_Filtered['recall'])))
        file1.write('F1\t\t%s\t\t%s\t\t%s\t\t%s\t\t%s' % (str(dict_adv_raw['f1']),str(dict_X_Test['f1']), str(dict_Adv_Not_Filtered['f1']), str(dict_Original_Filtered['f1']), str(dict_Adv_Filtered['f1'])))

        file1.write('\n##################\n')

        file1.write('Metric on Surrogate MLP\n')
        file1.write('\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1\n')
        file1.write('%s\t\t%s\t\t%s\t\t%s\n' % (str(dict_General_values['MLPAccuracy']),str(dict_General_values['MLPPrecision']),str(dict_General_values['MLPRecall']),str(dict_General_values['MLPF1'])))
        file1.write('Not significant elements\t\t%s\n' % (str(dict_General_values['mod'])))
        file1.write('Significant elements\t\t%s\n'%(str(dict_General_values['not_mod'])))
        file1.write('Uncertain elements\t\t%s\n' % (str(dict_General_values['grey'])))
        file1.write('Total elements\t\t%s\n' % (str(dict_General_values['total'])))
        file1.write('Originally BENIGN not representative\t\t%s\n' % (str(dict_General_values['originally_BENIGN_nr'])))
        file1.write('Originally ATTACK not representative\t\t%s\n' % (str(dict_General_values['originally_ATTACK_nr'])))

        file2.write('Metric,Adv Raw,Original Test Set (X_test related to Sanitized),Not Filtered Adversarial Samples(adv_x sanitized),Original Test after filtering(original_sig_samples related to representative)\t\tFiltered Adversarial Samples(sig_adv_sample representative adv)\n')
        file2.write('Accuracy,%s,%s,%s,%s,%s\n' % (str(dict_adv_raw['accuracy']), str(dict_X_Test['accuracy']), str(dict_Adv_Not_Filtered['accuracy']),str(dict_Original_Filtered['accuracy']), str(dict_Adv_Filtered['accuracy'])))
        file2.write('Precision,%s,%s,%s,%s,%s\n' % (str(dict_adv_raw['precision']), str(dict_X_Test['precision']), str(dict_Adv_Not_Filtered['precision']),str(dict_Original_Filtered['precision']), str(dict_Adv_Filtered['precision'])))
        file2.write('Recall,%s,%s,%s,%s,%s\n' % (str(dict_adv_raw['recall']), str(dict_X_Test['recall']), str(dict_Adv_Not_Filtered['recall']),str(dict_Original_Filtered['recall']), str(dict_Adv_Filtered['recall'])))
        file2.write('F1,%s,%s,%s,%s,%s' % (str(dict_adv_raw['f1']), str(dict_X_Test['f1']), str(dict_Adv_Not_Filtered['f1']),str(dict_Original_Filtered['f1']), str(dict_Adv_Filtered['f1'])))


if __name__ == '__main__':

    #flags.DEFINE_bool('backprop_through_attack', False,('If True, backprop through adversarial example ''construction process during adversarial training'))
    #qui i dati vengono gia' privati dei campi non necessari
    fileTrain = base_dir+'/updatedInput/ADV-training.csv' #VERSIONE DATI DEI BELGI
    datasetTrain,_ = get_data_from_file(fileTrain)

    fileTest = base_dir+'/updatedInput/ADVBASE1-baseAdv-test.csv' #VERSIONE DATI DEI BELGI
    datasetTest, originalTestDataset = get_data_from_file(fileTest)

    fileVal = base_dir + '/updatedInput/ADVBASE2-cfOracle.csv' #VERSIONE DATI DEI BELGI
    datasetVal,_ = get_data_from_file(fileVal)



    X_train, Y_train, _ = preprocess(datasetTrain, True)
    X_test, Y_test, _ = preprocess(datasetTest, True)
    X_FIL, Y_FIL, _ = preprocess(datasetVal, False)

    protocol_test = X_test[:, 0]
    #adv_params= {'clip_min': 0.0, 'clip_max': 1.1, 'eps': 0.01} #FGSM
    #adv_params = {'clip_min': 0.0, 'clip_max': 1.1, 'theta': 0.7, 'gamma': 0.3}  # JSMA
    #adv_params = {'clip_min': 0.0, 'clip_max': 1.1, 'nb_candidate': 2, 'nb_classes': 2, 'overshoot': 0.1, 'max_iter': 15}  # DeepFool
    adv_params = {'clip_min': 0.0, 'clip_max': 1.1, 'eps': 0.1, 'xi': 0.5, 'num_iterations': 1}  # VirtualADV


    algorithm = 'VirtualAdversarialMethod'
    #Generazione
    adv_samples, not_scaled_adv, X_test, Y_test, advScaler,  originalTestDataset, rawCopy, protocol_test = generate_adversarial_samples(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, adv_params=adv_params, originalTestDateset=originalTestDataset,algorithm=algorithm, protocol_test=protocol_test)

    print(Y_test.shape)
    #test di lettura adv raw file per test disaccoppiamento pipe
    #fileTest = base_dir+'/updatedOutput/adv_RAW/FastGradientSignMethod.csv' #VERSIONE DATI DEI BELGI
    #datasetTest, originalTestDataset = get_data_from_file(fileTest)
    #X_test, Y_test, _ = preprocess(datasetTest, True)
    #protocol_test = X_test[:, 0]
    #sanitize_adv, X_test, Y_test = get_sanitize_adversarial_samples(X_test, X_test, Y_test, protocol_test,originalTestDataset)
    ######
    dt, _, _, _, _, _ = decisionTree(gridSearch=False)
    rf, _, _, _, _, _ = randomForest(gridSearch=False)
    mlp = MultilayerPerceptron(input_dim=X_test.shape[1])

    print("RAW ADV VALUES")
    print(not_scaled_adv.shape)
    print(X_test.shape)
    print(Y_test.shape)
    prediction_RAW_DT = dt.predict(not_scaled_adv)  # results related to whole dataset of Original Samples
    prediction_RAW_RF = rf.predict(not_scaled_adv)
    prediction_RAW_MLP = mlp.predict(not_scaled_adv)

    dictionary_raw_DT = cmDT(Y_test, prediction_RAW_DT)  # .flatten per il random forest
    dictionary_raw_RF = cmRF(Y_test, prediction_RAW_RF)
    dictionary_raw_MLP = mlp.evaluate(prediction_RAW_MLP, Y_test)


    #Sanity Check
    sanitize_adv, X_test, Y_test, protocol_test= get_sanitize_adversarial_samples(not_scaled_adv, X_test, Y_test, protocol_test, originalTestDataset)

    print("SANITIZED VALUES")
    print(sanitize_adv.shape)
    print("ORIGINAL VALUES related to SANITIZED")
    print(X_test.shape)
    print(Y_test.shape)
    prediction_sanitized_DT = dt.predict(sanitize_adv)  # results related to Sanitized Samples
    prediction_sanitized_RF = rf.predict(sanitize_adv)
    prediction_sanitized_MLP = mlp.predict(sanitize_adv)

    dictionary_sanitized_DT = cmDT(Y_test, prediction_sanitized_DT)  # .flatten per il random forest
    dictionary_sanitized_RF = cmRF(Y_test, prediction_sanitized_RF)
    dictionary_sanitized_MLP = mlp.evaluate(prediction_sanitized_MLP, Y_test)

    prediction_Original_sanitized_DT = dt.predict(X_test)  # results related to Original Samples related to Sanitezed Adversarial ones
    prediction_Original_sanitized_RF = rf.predict(X_test)
    prediction_Original_sanitized_MLP = mlp.predict(X_test)

    dictionary_Original_sanitized_DT = cmDT(Y_test, prediction_Original_sanitized_DT)  # .flatten per il random forest
    dictionary_Original_sanitized_RF = cmRF(Y_test, prediction_Original_sanitized_RF)
    dictionary_Original_sanitized_MLP = mlp.evaluate(prediction_Original_sanitized_MLP, Y_test)

    #AGGIORNARE VALORE DEL FILE NAME PER EVITARE SOVRASCRIZIONI
    sig_adv, original_sig_samples, sig_ORIGINAL_Y, non_rep, index_sig, index_non_rep=get_representative_samples(X_FIL, Y_FIL, X_test, Y_test, adversarial_samples=sanitize_adv, adv_file_name=algorithm, originalTestDataset=originalTestDataset)

    prediction_orig_sig_DT = dt.predict(original_sig_samples)  # results related to Original Samples related to Representative Adversarial ones
    prediction_orig_sig_RF = rf.predict(original_sig_samples)
    prediction_orig_sig_MLP = mlp.predict(original_sig_samples)

    dictionary_orig_sig_DT = cmDT(sig_ORIGINAL_Y, prediction_orig_sig_DT)  # .flatten per il random forest
    dictionary_orig_sig_RF = cmRF(sig_ORIGINAL_Y, prediction_orig_sig_RF)
    dictionary_orig_sig_MLP = mlp.evaluate(prediction_orig_sig_MLP, sig_ORIGINAL_Y)

    prediction_sig_adv_DT = dt.predict(sig_adv)  # results related to Original Samples related to Sanitezed ones
    prediction_sig_adv_RF = rf.predict(sig_adv)
    prediction_sig_adv_MLP = mlp.predict(sig_adv)

    dictionary_sig_adv_DT = cmDT(sig_ORIGINAL_Y, prediction_sig_adv_DT)  # .flatten per il random forest
    dictionary_sig_adv_RF = cmRF(sig_ORIGINAL_Y, prediction_sig_adv_RF)
    dictionary_sig_adv_MLP = mlp.evaluate(prediction_sig_adv_MLP, sig_ORIGINAL_Y)

    write_Results_On_File(algorithm+"_DT",dictionary_raw_DT, dictionary_Original_sanitized_DT,dictionary_sanitized_DT,dictionary_orig_sig_DT,dictionary_sig_adv_DT)
    write_Results_On_File(algorithm + "_RF", dictionary_raw_RF, dictionary_Original_sanitized_RF,dictionary_sanitized_RF, dictionary_orig_sig_RF, dictionary_sig_adv_RF)
    write_Results_On_File(algorithm + "_MLP", dictionary_raw_MLP, dictionary_Original_sanitized_MLP,dictionary_sanitized_MLP, dictionary_orig_sig_MLP, dictionary_sig_adv_MLP)








