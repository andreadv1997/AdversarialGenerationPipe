from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging


#from mlp_test_filtered import MultilayerPerceptron ADDESTRAMENTO CON ESEMPI AVVERSARI
from mlp_test import MultilayerPerceptron #ADDESTRAMENTO NORMALE
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

from sklearn.metrics import recall_score
#from decisionTree_filtered_training import retrain, decisionTree, compute_metrics as cmDT, adversarialTraining as advTrainDT ADDESTRAMENTO CON ESEMPI AVVERSARI
from decisionTreeNEW import retrain, decisionTree, compute_metrics as cmDT, adversarialTraining as advTrainDT #ADDESTRAMENTO NORMALE
#from randomForest_filtered import randomForest, compute_metrics as cmRF, adversarialTraining as advTrainRF ADDESTRAMENTO CON ESEMPI AVVERSARI
from randomForest import randomForest, compute_metrics as cmRF, adversarialTraining as advTrainRF #ADDESTRAMENTO NORMALE
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances

FLAGS = flags.FLAGS
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

to_update_feature = [ ('Flow_Bytes/s',4,5,1) , ('Flow_Packets/s',2,3,1), ('Fwd_Packets/s',2,1), ('Bwd_Packets/s',3,1)]

dt = None
rf = None
mlp = None
algorithm_under_test = ""

def find_n_neighbours(df,n):
    order = np.argsort(df.values, axis=1)[:, :n] #controllare comportamento di questa istruzion. Questa istruzione non e' necessaria
    #df = df.apply(lambda x: pd.Series(x.sort_values(ascending=True).iloc[:n].index,index=['top{}'.format(i) for i in range(1, n + 1)]), axis=1)
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
    #fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1) #qui andrebbe tolta la moltiplicazione con x[correlation probabilmente]
    nume = fin['adg_score'].sum()


    #deno = fin['correlation'].sum() #e' giusto questo calcolo? Ossia al denumeratore non ci dovrebbero essere il numero di user comparati?
    deno = n

    #final_score = avg_user + (nume/deno)
    final_score =  (nume / deno)
    return final_score

def mnist_tutorial(train_start=0, train_end=8363, test_start=0,
                   test_end=2092, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64, num_threads=None, gridSearch=True):
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

    #qui i dati vengono ggia' privati dei campi non necessari
    #fileTrain = base_dir+'/bin/ADV-training.csv' VERSIONE DATI CICIDS
    fileTrain = base_dir+'/updatedInput/ADV-training.csv' #VERSIONE DATI DEI BELGI
    datasetTrain,_ = get_data_from_file(fileTrain)
    #fileTest = base_dir+'/bin/ADV-baseline-PARTE1.csv' VERSIONE DATI CICIDS
    fileTest = base_dir+'/updatedInput/ADVBASE1-baseAdv-test.csv' #VERSIONE DATI DEI BELGI
    datasetTest, originalTestDataset = get_data_from_file(fileTest)
    #fileVal = base_dir + '/bin/ADV-baseline-PARTE2.csv' VERSIONE DATI CICIDS
    fileVal = base_dir + '/updatedInput/ADVBASE2-cfOracle.csv' #VERSIONE DATI DEI BELGI
    datasetVal,_ = get_data_from_file(fileVal)

    advScaler = MaxAbsScaler()

    X_train, Y_train, _ = preprocess(datasetTrain, True)
    X_test, Y_test, _ = preprocess(datasetTest, True)
    X_FIL, Y_FIL, _ = preprocess(datasetVal, False)




    train_end = X_train.shape[0]
    test_end = X_test.shape[0]
    #y_target = np.empty((Y_test.shape[0], Y_test.shape[1]))


    protocol_train = X_train[:,0]
    protocol_test = X_test[:,0]


    advScaler = advScaler.fit(X_train)
    scaled_X_train = advScaler.transform(X_train)
    scaled_X_test = advScaler.transform(X_test)

    #cancellazione della colonna "Protocol"
    X_train_adv_mlp = np.delete(scaled_X_train, 0, 1)
    X_test_adv_mlp = np.delete(scaled_X_test, 0, 1)


    input_shape = X_train_adv_mlp.shape[1]
    # Use label smoothing
    print("Dimensione dell'input Train: "+str(X_train_adv_mlp.shape)+"\n")
    print("Dimensione dell'output Train: " + str(Y_train.shape) + "\n")

    print("Dimensione dell'input Test: " + str(X_test_adv_mlp.shape) + "\n")
    print("Dimensione dell'output Test: " + str(Y_test.shape) + "\n")

    print("Dimensione dell'input Val: " + str(X_FIL.shape) + "\n")
    print("Dimensione dell'output Val: " + str(Y_FIL.shape) + "\n")

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, input_shape))
    y = tf.placeholder(tf.float32, shape=(None, 2))




    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }


    global algorithm_under_test
    algorithm_under_test="FGSM_TEST_filtered_TEST___"
    adv_file_name ="FGSM_TEST_filt_.csv"

    ##NB RIESEGUIRE TRANING ADV DEEPFOOL PER RIGENERARE FILE ALTERATO

    #state of art test
    #fgsm_params = {'clip_min': 0.0, 'clip_max': 1.1, 'theta': 0.7, 'gamma': 0.7}  # JSMA
    fgsm_params = {'clip_min': 0.0, 'clip_max': 1.1, 'eps': 0.01}  # FGSM
    #fgsm_params = {'clip_min': 0.0, 'clip_max': 1.1, 'nb_candidate': 2, 'nb_classes': 2, 'overshoot': 0.1, 'max_iter': 1}  # DeepFool
    #fgsm_params = {'clip_min': 0.0, 'clip_max': 1.1, 'eps': 0.5, 'xi': 0.1, 'num_iterations': 15}  # VirtualADV
    #fgsm_params = {'clip_min': 0.0, 'clip_max': 1.1, 'confidence': 0.5, 'learning_rate': 0.1, 'batch_size': 1,'binary_search_steps': 10, 'max_iterations': 5, 'abort_early': False,'initial_const': 1}  # CW


    #Set of fist group of parameters (thesis)
    #fgsm_params = {'clip_min': 2.2250738585072014e-308, 'clip_max': 1.7976931348623157e+308, 'eps': 0.1} #FGSM
    #fgsm_params = {'clip_min': 2.2250738585072014e-308, 'clip_max': 1.7976931348623157e+308, 'theta': 0.1, 'gamma': 0.5}  # JSMA
    #fgsm_params = {'clip_min': 2.2250738585072014e-308, 'clip_max': 1.7976931348623157e+308, 'nb_candidate': 2,'nb_classes': 2, 'overshoot':0.9, 'max_iter':15}  # DeepFool
    #fgsm_params = {'clip_min':2.2250738585072014e-308,'clip_max': 1.7976931348623157e+308,'eps': 3.5,'xi':3.5,'num_iterations':1} #VirtualADV

    rng = np.random.RandomState([2017, 8, 30]) #riproducibilita'

    if clean_train:

        model = make_basic_mlp_non_conv(input_shape=(None,input_shape))
        preds = model.get_probs(x) #placeholder

        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test_adv_mlp, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)
        model_train(sess, x, y, preds, X_train_adv_mlp, Y_train, evaluate=evaluate, args=train_params, rng=rng, var_list=model.get_params())
        # Calculate training error
        if testing:
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_train_adv_mlp, Y_train, args=eval_params)
            report.train_clean_train_clean_eval = acc

        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        fgsm = FastGradientMethod(model, sess=sess)
        #fgsm = SaliencyMapMethod(model, sess=sess)
        #fgsm = DeepFool(model, sess=sess)
        #fgsm = CarliniWagnerL2(model, sess=sess)
        #fgsm = VirtualAdversarialMethod(model, sess=sess)

        adv_x = fgsm.generate(x, **fgsm_params)
        preds_adv = model.get_probs(adv_x)

        # Evaluate the accuracy of the MNIST model on adversarial examples
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test_adv_mlp, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        report.clean_train_adv_eval = acc

        #global
        dict_General_values['MLPaccuracy'] = acc

        #exit()

        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train_adv_mlp,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc


        ##COLLABORATIVE FILTERING CON 77 feature



        adversarial_s = adv_x.eval({x:X_test_adv_mlp}, session=sess)
        original_Y_Test = np.copy(Y_test)


        #####Solo  per DeepFool######
        shape = adversarial_s.shape[0]
        i = 0
        print("ADV s shape pre sanity\n\n")
        print(adversarial_s.shape)
        while i < shape:
            if (np.isinf(adversarial_s[i]).any() or np.isnan(adversarial_s[i]).any()):
                adversarial_s = np.delete(adversarial_s, obj=i, axis=0)
                X_test = np.delete(X_test, obj=i, axis=0)
                Y_test = np.delete(Y_test, obj=i, axis=0)
                protocol_test = np.delete(protocol_test, obj=i, axis=0) #riaggiorna dimensione dell'array di protocol per DeepFool
                originalTestDataset.drop(originalTestDataset.index[i], inplace= True)###
                shape = shape - 1
            else:
                i += 1

        print("ADV s shape pre sanity\n")
        print(adversarial_s.shape)
        print(originalTestDataset.shape)
        originalTestDataset.reset_index(drop=True, inplace=True)
        # QUI I SAMPLE SONO NON SCALATI
        adversarial_samples, X_test, Y_test = sannity_check_adv(adversarial_s, protocol_test, advScaler, X_test, Y_test, adv_file_name, originalTestDataset)
        print(adversarial_samples.shape)
        #pd.DataFrame(np.concatenate((adversarial_samples, Y_test[:,0].reshape(Y_test.shape[0],1)), axis=1)).to_csv(base_dir+"/adversarialsVAM_FILTERED_CASE.csv", index=False, header=features)
        #exit()


        global dt
        dt,_,_,_,_,_ = decisionTree(gridSearch)  # contiene il calcolo delle metriche per il file MOD-test
        global rf
        rf,_,_,_,_,_ = randomForest(gridSearch)  # contiene il calcolo delle metriche per il file MOD-test
        global mlp
        mlp = MultilayerPerceptron(input_dim=X_test.shape[1])

        print(X_test)

        #*****CALCOLO RISULTATI ORIGINALI (X TEST)*****
        prediction_original_DT = dt.predict(X_test)  # previous X_test
        prediction_original_RF = rf.predict(X_test)
        prediction_original_MLP = mlp.predict(X_test)

        dictionary_X_Test_DT=cmDT(Y_test, prediction_original_DT) #.flatten per il random forest
        dictionary_X_Test_RF = cmRF(Y_test, prediction_original_RF)
        dictionary_X_Test_MLP = mlp.evaluate(prediction_original_MLP, Y_test)

        printResults(dictionary_X_Test_DT, dictionary_X_Test_RF, dictionary_X_Test_MLP, headerString="*****CALCOLO RISULTATI ORIGINALI (X TEST)*****")



        #*****CALCOLO RISULTATI AVVERSARI NON FILTRATI (adversaria_samples)*****
        prediction_test_DT = dt.predict(adversarial_samples)
        prediction_test_RF = rf.predict(adversarial_samples)
        prediction_test_MLP = mlp.predict(adversarial_samples)

        dictionary_adversarial_samples_DT = cmDT(Y_test, prediction_test_DT)
        dictionary_adversarial_samples_RF = cmRF(Y_test, prediction_test_RF)
        dictionary_adversarial_samples_MLP = mlp.evaluate(prediction_test_MLP, Y_test)

        printResults(dictionary_adversarial_samples_DT,dictionary_adversarial_samples_RF,dictionary_adversarial_samples_MLP, headerString="*****CALCOLO RISULTATI AVVERSARI NON FILTRATI (adversaria_samples)*****")


        #CALCOLO ESEMPI SIGNIFICATIVI
        #QUI I DATI SONO NON SCALATI
        sig_adv, original_sig_samples, sig_ORIGINAL_Y, non_rep, index_sig, index_non_rep = collaborative_filtering(X_FIL, Y_FIL, X_test, Y_test, adversarial_samples,adv_file_name,originalTestDataset)

        #original_sig_samples = new_scaler.transform(notScaled_original_sig_samples)
        #sig_adv = new_scaler.transform(notScaled_sig_adv)

        #pd.DataFrame(np.concatenate((asig_adv, sig_ORIGINAL_Y[:, 0].reshape(sig_ORIGINAL_Y.shape[0], 1)), axis=1)).to_csv(base_dir + "/adversarial_samples/" + adv_file_name, header=features, index=False)

        #vecchia
        #pd.DataFrame(np.concatenate((sig_adv,sig_ORIGINAL_Y[:,0].reshape(sig_ORIGINAL_Y.shape[0],1)),axis=1)).to_csv(base_dir+"/ADV_OUTPUT/adv_representative/"+adv_file_name, header=features, index=False)
        tempCopy = originalTestDataset.iloc[index_sig].copy()

        dataframe = pd.DataFrame(sig_adv, columns=featuresNoLabel)
        tempCopy[featuresNoLabel] = dataframe[featuresNoLabel].values
        tempCopy.to_csv(base_dir + "/updatedOutput/adv_representative/" + adv_file_name,header=originalFeatures, index=False)


        #exit()

        print('Original: '+str(original_sig_samples.shape))
        print('ADV: ' + str(sig_adv.shape))

        #*****CALCOLO RISULTATI ORIGINALI POST FILTRAGGIO (X_Test associati ad avverari rappresentativi/original_sig_samples)*****
        predicition_original_post_fil_DT = dt.predict(original_sig_samples)
        predicition_original_post_fil_RF = rf.predict(original_sig_samples)
        prediction_original_post_fil_MLP = mlp.predict(original_sig_samples)

        dictionary_original_sig_samples_DT= cmDT(sig_ORIGINAL_Y, predicition_original_post_fil_DT)
        dictionary_original_sig_samples_RF = cmRF(sig_ORIGINAL_Y, predicition_original_post_fil_RF)
        dictionary_original_sig_samples_MLP =  mlp.evaluate(prediction_original_post_fil_MLP, sig_ORIGINAL_Y)

        printResults(dictionary_original_sig_samples_DT, dictionary_original_sig_samples_RF, dictionary_original_sig_samples_MLP, headerString="*****CALCOLO RISULTATI ORIGINALI POST FILTRAGGIO (X_Test associati ad avverari rappresentativi/original_sig_samples)*****")




        #*****CALCOLO RISULTATI AVVERSARI POST FILTRAGGIO (rappresentativi/sig_adv)*****
        prediction_adv_post_fil_DT = dt.predict(sig_adv)
        prediction_adv_post_fil_RF = rf.predict(sig_adv)
        prediction_adv_post_fil_MLP = mlp.predict(sig_adv)

        dictionary_sig_adv_DT = cmDT(sig_ORIGINAL_Y, prediction_adv_post_fil_DT)
        dictionary_sig_adv_RF = cmRF(sig_ORIGINAL_Y, prediction_adv_post_fil_RF)
        dictionary_sig_adv_MLP = mlp.evaluate(prediction_adv_post_fil_MLP, sig_ORIGINAL_Y)

        printResults(dictionary_sig_adv_DT, dictionary_sig_adv_RF, dictionary_sig_adv_MLP, headerString="*****CALCOLO RISULTATI AVVERSARI POST FILTRAGGIO (rappresentativi/sig_adv)*****")


        write_Results_On_File(algorithm_under_test + "_DecisionTree", dictionary_X_Test_DT,dictionary_adversarial_samples_DT, dictionary_original_sig_samples_DT,dictionary_sig_adv_DT)
        write_Results_On_File(algorithm_under_test + "_RandomForest", dictionary_X_Test_RF,dictionary_adversarial_samples_RF, dictionary_original_sig_samples_RF,dictionary_sig_adv_RF)
        write_Results_On_File(algorithm_under_test + "_MLP", dictionary_X_Test_MLP,dictionary_adversarial_samples_MLP, dictionary_original_sig_samples_MLP,dictionary_sig_adv_MLP)

        exit()


        #CALCOLO METRICHE CON CLASSIFICATORI ADDESTRATI SU ESEMPI RAPPRESENTATIVI
        #print("*******CALCOLO METRICHE CON CLASSIFICATORI ADDESTRATI SU ESEMPI RAPPRESENTATIVI NON FILTRATI******")
        #dt = retrain(adversarial_samples, Y_test)
        print("*******CALCOLO METRICHE CON CLASSIFICATORI ADDESTRATI SU ESEMPI AVVERSARI FILTRATI******")


        fgsm_params_rep_test = {'clip_min': 0.0, 'clip_max': 1.1, 'confidence': 0.5, 'learning_rate': 0.1, 'batch_size': 1,
                            'binary_search_steps': 10, 'max_iterations': 5, 'abort_early': False,
                            'initial_const': 1}  # CW
        fgsm_rep_test = CarliniWagnerL2(model, sess=sess)
        adv_x_rep_test = fgsm_rep_test.generate(x, **fgsm_params_rep_test)

        adv_x_rep_test = adv_x_rep_test.eval({x:X_test_adv_mlp}, session=sess)
        adversarial_samples_test, _, Y_rep_test = sannity_check_adv(adv_x_rep_test, protocol_test, advScaler, X_test_adv_mlp, original_Y_Test)
        adversarial_samples_test, _, Y_rep_test, non_rep, index_sig, index_non_rep = collaborative_filtering(X_FIL, Y_FIL, np.concatenate((protocol_test.reshape(protocol_test.shape[0], 1),X_test_adv_mlp),axis=1), Y_rep_test,adversarial_samples_test, originalTestDataset)
        #per il momento evitiamo di filtrare i non rappresentativi per semplici

        #prediction_rep = dt.predict(adversarial_samples_test)
        #dict_rep = cmDT(Y_rep_test, prediction_rep)
        #print('Accuracy of dt(non filtered): ' + str(dict_rep['accuracy']))
        #print('Precision of dt(non filtered): ' + str(dict_rep['precision']))
        #print('Recall of dt(non filtered): ' + str(dict_rep['recall']))
        #print('F1 of dt(non filtered): ' + str(dict_rep['f1']))

        '''
        fgsm_params_alg2 = {'clip_min': 0.0, 'clip_max': 1.8, 'theta': 0.1, 'gamma': 0.7}  # JSMA
        fgsm_alg2 = SaliencyMapMethod(model, sess=sess)
        adv_x_alg2 = fgsm_alg2.generate(x, **fgsm_params_alg2)
        adv_model_pred_alg2 = model.get_probs(adv_x_alg2)

        adv_model_pred_alg2 = adv_model_pred_alg2.eval({x: X_test_adv_mlp}, session=sess)
        adv_model_pred_alg2 = np.argmax(adv_model_pred_alg2, axis=1)
        adv_model_pred_alg2 = np.where(adv_model_pred_alg2 == 0, 1, 0)
        adv_model_pred_alg2 = adv_model_pred_alg2.reshape((adv_model_pred_alg2.shape[0], 1))

        adv_x_alg2 = adv_x_alg2.eval({x: X_test_adv_mlp}, session=sess)
        adversarial_samples_alg2, _, Y_alg2 = sannity_check_adv(adv_x_alg2, protocol_test, advScaler,X_test_adv_mlp, original_Y_Test)
        adversarial_samples_alg2, _,label_comp_alg2,_ = collaborative_filtering(X_FIL, Y_FIL, np.concatenate((protocol_test.reshape(protocol_test.shape[0], 1),X_test_adv_mlp),axis=1), Y_alg2,adversarial_samples_alg2)

        fgsm_params_alg3 = {'clip_min': 0.0, 'clip_max': 1.8, 'nb_candidate': 2, 'nb_classes': 2, 'overshoot': 0.1,
                            'max_iter': 1}  # DeepFool
        fgsm_alg3 = DeepFool(model, sess=sess)
        adv_x_alg3 = fgsm_alg3.generate(x, **fgsm_params_alg3)
        adv_model_pred_alg3 = model.get_probs(adv_x_alg3)

        adv_model_pred_alg3 = adv_model_pred_alg3.eval({x: X_test_adv_mlp}, session=sess)
        adv_model_pred_alg3 = np.argmax(adv_model_pred_alg3, axis=1)
        adv_model_pred_alg3 = np.where(adv_model_pred_alg3 == 0, 1, 0)
        adv_model_pred_alg3 = adv_model_pred_alg3.reshape((adv_model_pred_alg3.shape[0], 1))

        adv_x_alg3 = adv_x_alg3.eval({x: X_test_adv_mlp}, session=sess)
        adversarial_samples_alg3, _, Y_alg3 = sannity_check_adv(adv_x_alg3, protocol_test, advScaler,X_test_adv_mlp, original_Y_Test)
        adversarial_samples_alg3, _,label_comp_alg3, _ = collaborative_filtering(X_FIL, Y_FIL, np.concatenate((protocol_test.reshape(protocol_test.shape[0], 1),X_test_adv_mlp),axis=1), Y_alg3,adversarial_samples_alg3)

        train_adv_samples = np.concatenate((sig_adv, adversarial_samples_alg2), axis=0)
        train_adv_labels = np.concatenate((sig_ORIGINAL_Y, label_comp_alg2), axis=0)

        train_adv_samples = np.concatenate((train_adv_samples, adversarial_samples_alg3), axis=0)
        train_adv_labels = np.concatenate((train_adv_labels, label_comp_alg3), axis=0)
        '''

        dt = advTrainDT()
        rf = advTrainRF()
        mlp.adv_train()

        prediction_rep_DT = rf.predict(adversarial_samples_test)
        dict_rep_DT = cmRF(Y_rep_test, prediction_rep_DT)

        prediction_rep_RF = rf.predict(adversarial_samples_test)
        dict_rep_RF = cmRF(Y_rep_test, prediction_rep_RF)

        prediction_rep_MLP = mlp.predict(adversarial_samples_test)
        dict_rep_MLP = mlp.evaluate(prediction_rep_MLP, Y_rep_test)

        print('Accuracy of decision tree: ' + str(dict_rep_DT['accuracy']))
        print('Accuracy of random forest: ' + str(dict_rep_RF['accuracy']))
        print('Accuracy of MLP: ' + str(dict_rep_MLP['accuracy']))
        print()
        print('Precision of decision tree: ' + str(dict_rep_DT['precision']))
        print('Precision of random forest: ' + str(dict_rep_RF['precision']))
        print('Precision of MLP: ' + str(dict_rep_MLP['precision']))
        print()
        print('Recall of decision tree: ' + str(dict_rep_DT['recall']))
        print('Recall of random forest: ' + str(dict_rep_RF['recall']))
        print('Recall of MLP: ' + str(dict_rep_MLP['recall']))
        print()
        print('F1 of decision tree: ' + str(dict_rep_DT['f1']))
        print('F1 of random forest: ' + str(dict_rep_RF['f1']))
        print('F1 of MLP: ' + str(dict_rep_MLP['f1']))




    return report




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


        if (score < 0.2):
            label_rec[index] = 0
        elif (score > 0.8):
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


    return sig_adv, original_sig_samples, sig_ORIGINAL_Y, nonRep_adv, index_sig_adv, index_non_rep

def save_on_file_NR(data, fileName, type, originalData, list_of_indexes): #VERIFICARE SE CONVIENE INCOLLARE LE FEATURE QUI O NEL METODO CHIAMANTE
    base_dir = str(Path().resolve().parent)
    tempCopy = originalData.iloc[list_of_indexes].copy() ##verificare funzione assign()

    dataframe = pd.DataFrame(data, columns=features_save_dataframe)

    tempCopy[featuresNoLabel] = dataframe[featuresNoLabel].values
    tempCopy.insert(len(originalFeatures), "Computed Label", dataframe["Computed_Label"].values)



    if(type =="BENIGN"):
        #dataframe.to_csv(base_dir+"/ADV_OUTPUT/adv_ORIG_BEN/"+fileName, header=features_save_dataframe, index=False)
        tempCopy.to_csv(base_dir + "/updatedOutput/adv_ORIG_BEN/" + fileName, header=originalFeatures+['Computed_Label'], index=False)
        print(tempCopy.shape)
        print(dataframe.shape)
    elif (type == "ATTACK"):
        tempCopy.to_csv(base_dir + "/updatedOutput/adv_ORIG_ATT/" + fileName, header=originalFeatures+['Computed_Label'], index=False)
        print(tempCopy.shape)
        print(dataframe.shape)
    else:
        #dataframe.to_csv(base_dir + "/ADV_OUTPUT/adv_grey/" + fileName, header=features_save_dataframe+['Computed_Label'], index=False)
        tempCopy.to_csv(base_dir + "/updatedOutput/adv_grey/" + fileName, header=originalFeatures + ['Computed_Label'], index=False)



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

def sannity_check_adv(adversarial_samples, protocol_test, scaler, X_test, Y_test, file_name, originalData):

    adversarial_samples = np.concatenate((np.zeros(shape=(adversarial_samples.shape[0],1)), adversarial_samples), axis=1)
    not_scaled_samples = scaler.inverse_transform(adversarial_samples) #descaling

    # nuova implementazione
    rawCopy = originalData.copy()
    rawCopy[features_for_scaling] = not_scaled_samples[:,1:]  # controlla qui

    rawCopy.to_csv(base_dir + "/updatedOutput/adv_RAW/" + file_name, header=originalFeatures, index=False)



    protocol_test = protocol_test.reshape(protocol_test.shape[0],1)
    not_scaled_samples[:,0] = protocol_test[:,0]
    not_scaled_to_copy = np.concatenate((not_scaled_samples,Y_test[:,0].reshape(Y_test.shape[0],1)),axis=1)
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

        if(len(to_update_feature[index_1]) == 4):
            not_scaled_samples[:,features.index(to_update_feature[index_1][0])] = (not_scaled_samples[:,to_update_feature[index_1][1]] + not_scaled_samples[:,to_update_feature[index_1][2]]) / not_scaled_samples[:,to_update_feature[index_1][3]]

        if (len(to_update_feature[index_1]) == 3):
            not_scaled_samples[:, features.index(to_update_feature[index_1][0])] = (not_scaled_samples[:,to_update_feature[index_1][1]]) / not_scaled_samples[:,to_update_feature[ index_1][2]]


    shape = not_scaled_samples.shape[0]
    #protocol_test = protocol_test.reshape(protocol_test.shape[0], 1)
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
    #protocol_test = protocol_test.reshape(protocol_test.shape[0], 1)
    #not_scaled_samples[:,0] = protocol_test[:,0]




    return not_scaled_samples, X_test, Y_test

def computed_metrics_for_NonRep(nonRep_adv, original_nonRep_samples, nonRep_ORIGINAL_Y, label_computer_nonRep):
    prediction_nonRep_DT = dt.predict(nonRep_adv)
    prediction_nonRep_RF = rf.predict(nonRep_adv)
    prediction_nonRep_MLP = mlp.predict(nonRep_adv)

    print("***Prediction MLP***")
    print(prediction_nonRep_MLP)

    prediction_original_DT = dt.predict(original_nonRep_samples)
    prediction_original_RF = rf.predict(original_nonRep_samples)
    prediction_original_MLP = mlp.predict(original_nonRep_samples)

    dicionary_ORIGINAL_DT = cmDT(nonRep_ORIGINAL_Y, prediction_original_DT)
    dicionary_ORIGINAL_RF = cmRF(nonRep_ORIGINAL_Y, prediction_original_RF)
    dicionary_ORIGINAL_MLP = mlp.evaluate(prediction_original_MLP, nonRep_ORIGINAL_Y)

    dictionary_WITH_ORIGINAL_LABEL_DT = cmDT(nonRep_ORIGINAL_Y, prediction_nonRep_DT)
    dictionary_WITH_ORIGINAL_LABEL_RF = cmRF(nonRep_ORIGINAL_Y, prediction_nonRep_RF)
    dictionary_WITH_ORIGINAL_LABEL_MLP = mlp.evaluate(prediction_nonRep_MLP, nonRep_ORIGINAL_Y)

    dictionary_WITH_COMPUTED_LABEL_DT = cmDT(label_computer_nonRep, prediction_nonRep_DT)
    dictionary_WITH_COMPUTED_LABEL_RF = cmRF(label_computer_nonRep, prediction_nonRep_RF)
    dictionary_WITH_COMPUTED_LABEL_MLP = mlp.evaluate(prediction_nonRep_MLP, label_computer_nonRep)

    print("Accuracy Campioni Originali DT: "+str(dicionary_ORIGINAL_DT["accuracy"]))
    print("Accuracy Adv con Label originali DT: " + str(dictionary_WITH_ORIGINAL_LABEL_DT["accuracy"]))
    print("Accuracy Adv con Label Calcolate con Filtraggio DT: " + str(dictionary_WITH_COMPUTED_LABEL_DT["accuracy"]))
    print()
    print("Precision Campioni Originali DT: " + str(dicionary_ORIGINAL_DT["precision"]))
    print("Precision Adv con Label originali DT: " + str(dictionary_WITH_ORIGINAL_LABEL_DT["precision"]))
    print("Precision Adv con Label Calcolate con Filtraggio DT: " + str(dictionary_WITH_COMPUTED_LABEL_DT["precision"]))
    print()
    print("Recall Campioni Originali DT: " + str(dicionary_ORIGINAL_DT["recall"]))
    print("Recall Adv con Label originali DT: " + str(dictionary_WITH_ORIGINAL_LABEL_DT["recall"]))
    print("Recall Adv con Label Calcolate con Filtraggio DT: " + str(dictionary_WITH_COMPUTED_LABEL_DT["recall"]))
    print()
    print("TP Campioni Originali DT: " + str(dicionary_ORIGINAL_DT["true_pos"]))
    print("TP Adv con Label originali DT: " + str(dictionary_WITH_ORIGINAL_LABEL_DT["true_pos"]))
    print("TP Adv con Label Calcolate con Filtraggio DT: " + str(dictionary_WITH_COMPUTED_LABEL_DT["true_pos"]))
    print()
    print("TN Campioni Originali DT: " + str(dicionary_ORIGINAL_DT["true_neg"]))
    print("TN Adv con Label originali DT: " + str(dictionary_WITH_ORIGINAL_LABEL_DT["true_neg"]))
    print("TN Adv con Label Calcolate con Filtraggio DT: " + str(dictionary_WITH_COMPUTED_LABEL_DT["true_neg"]))
    print()

    write_NON_REP_RESULTS_On_File(algorithm_under_test+"_DecisionTree", dicionary_ORIGINAL_DT, dictionary_WITH_ORIGINAL_LABEL_DT, dictionary_WITH_COMPUTED_LABEL_DT)
    write_NON_REP_RESULTS_On_File(algorithm_under_test + "_RandomForest", dicionary_ORIGINAL_RF,dictionary_WITH_ORIGINAL_LABEL_RF, dictionary_WITH_COMPUTED_LABEL_RF)
    write_NON_REP_RESULTS_On_File(algorithm_under_test + "_MLP", dicionary_ORIGINAL_MLP,dictionary_WITH_ORIGINAL_LABEL_MLP, dictionary_WITH_COMPUTED_LABEL_MLP)




def main(argv=None):

    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters, gridSearch=FLAGS.gridSearch)


def write_NON_REP_RESULTS_On_File(fileName, dict_ORIGINAL, dict_WITH_ORIGINAL_LABEL, dict_WITH_COMPUTED_LABEL):

    outputDirTXT = base_dir + '/outputTXT_FILTERED___RESULTS_ON_FILTERED_DATA'


    if (not os.path.exists(outputDirTXT)):
        os.mkdir(outputDirTXT)

    txtFile = outputDirTXT + '/' + fileName + '_results_on_filtered.txt'


    with open(txtFile, "w") as file1:

        file1.write('Metric\t\tOriginal Samples\t\tAdversarial Samples and ORIGINAL LABEL\t\tAdversarial Samples and COMPUTED LABEL\n')
        file1.write('Accuracy\t\t%s\t\t%s\t\t%s\n' % (str(dict_ORIGINAL['accuracy']), str(dict_WITH_ORIGINAL_LABEL['accuracy']), str(dict_WITH_COMPUTED_LABEL['accuracy'])))
        file1.write('Precision\t\t%s\t\t%s\t\t%s\n' % (str(dict_ORIGINAL['precision']), str(dict_WITH_ORIGINAL_LABEL['precision']), str(dict_WITH_COMPUTED_LABEL['precision'])))
        file1.write('Recall\t\t%s\t\t%s\t\t%s\n' % (str(dict_ORIGINAL['recall']), str(dict_WITH_ORIGINAL_LABEL['recall']), str(dict_WITH_COMPUTED_LABEL['recall'])))
        file1.write('F1\t\t%s\t\t%s\t\t%s' % (str(dict_ORIGINAL['f1']), str(dict_WITH_ORIGINAL_LABEL['f1']), str(dict_WITH_COMPUTED_LABEL['f1'])))
        file1.write('\n##################\n')









def write_Results_On_File(fileName, dict_X_Test, dict_Adv_Not_Filtered, dict_Original_Filtered, dict_Adv_Filtered):

    outputDirTXT = base_dir + '/updatedOutput/outputTXT_FILTERED_NotScaled'
    outputDirCSV = base_dir + '/updatedOutput/outputCSV_FILTERED_NotScaled'

    if (not os.path.exists(outputDirTXT)):
        os.mkdir(outputDirTXT)
    if (not os.path.exists(outputDirCSV)):
        os.mkdir(outputDirCSV)

    txtFile = outputDirTXT + '/' + fileName + '_Results.txt'
    csvFile = outputDirCSV + '/' + fileName + '_Results.csv'

    with open(txtFile, "w") as file1, open(csvFile,"w") as file2 :

        file1.write('Metric\t\tOriginal Test Set (X_test)\t\tNot Filtered Adversarial Samples(adv_x)\t\tOriginal Test after filtering(original_sig_samples)\t\tFiltered Adversarial Samples(sig_adv_sample)\n')
        file1.write('Accuracy\t\t%s\t\t%s\t\t%s\t\t%s\n' % (str(dict_X_Test['accuracy']), str(dict_Adv_Not_Filtered['accuracy']), str(dict_Original_Filtered['accuracy']), str(dict_Adv_Filtered['accuracy'])))
        file1.write('Precision\t\t%s\t\t%s\t\t%s\t\t%s\n' % (str(dict_X_Test['precision']), str(dict_Adv_Not_Filtered['precision']), str(dict_Original_Filtered['precision']), str(dict_Adv_Filtered['precision'])))
        file1.write('Recall\t\t%s\t\t%s\t\t%s\t\t%s\n' % (str(dict_X_Test['recall']), str(dict_Adv_Not_Filtered['recall']), str(dict_Original_Filtered['recall']), str(dict_Adv_Filtered['recall'])))
        file1.write('F1\t\t%s\t\t%s\t\t%s\t\t%s' % (str(dict_X_Test['f1']), str(dict_Adv_Not_Filtered['f1']), str(dict_Original_Filtered['f1']), str(dict_Adv_Filtered['f1'])))

        file1.write('\n##################\n')

        file1.write('MLP Accuray\t\t%s\n' % (str(dict_General_values['MLPaccuracy'])))
        file1.write('Not significant elements\t\t%s\n' % (str(dict_General_values['mod'])))
        file1.write('Significant elements\t\t%s\n'%(str(dict_General_values['not_mod'])))
        file1.write('Uncertain elements\t\t%s\n' % (str(dict_General_values['grey'])))
        file1.write('Total elements\t\t%s\n' % (str(dict_General_values['total'])))
        file1.write('Originally BENIGN not representative\t\t%s\n' % (str(dict_General_values['originally_BENIGN_nr'])))
        file1.write('Originally ATTACK not representative\t\t%s\n' % (str(dict_General_values['originally_ATTACK_nr'])))

        file2.write('Metric, Original Test Set (X_test), Not Filtered Adversarial Samples(adv_x), Original Test after filtering(original_sig_samples), Filtered Adversarial Samples(sig_adv_sample)\n')
        file2.write('Accuracy, %s, %s, %s, %s\n' % (str(dict_X_Test['accuracy']), str(dict_Adv_Not_Filtered['accuracy']), str(dict_Original_Filtered['accuracy']), str(dict_Adv_Filtered['accuracy'])))
        file2.write('Precision, %s, %s, %s, %s\n' % (str(dict_X_Test['precision']), str(dict_Adv_Not_Filtered['precision']), str(dict_Original_Filtered['precision']), str(dict_Adv_Filtered['precision'])))
        file2.write('Recall, %s, %s, %s, %s\n' % (str(dict_X_Test['recall']), str(dict_Adv_Not_Filtered['recall']), str(dict_Original_Filtered['recall']), str(dict_Adv_Filtered['recall'])))
        file2.write('F1, %s, %s, %s, %s' % (str(dict_X_Test['f1']), str(dict_Adv_Not_Filtered['f1']), str(dict_Original_Filtered['f1']), str(dict_Adv_Filtered['f1'])))
        file2.write('MLP Accuray, %s\n' % (str(dict_General_values['MLPaccuracy'])))
        file2.write('Not significant elements, %s\n' % (str(dict_General_values['mod'])))
        file2.write('Significant elements, %s\n' % (str(dict_General_values['not_mod'])))
        file2.write('Uncertain elements, %s\n' % (str(dict_General_values['grey'])))
        file2.write('Total elements, %s' % (str(dict_General_values['total'])))




if __name__ == '__main__':

    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 5, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.01, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))
    flags.DEFINE_bool('gridSearch', False, 'Execute grid search for decision tree')

    tf.app.run()
