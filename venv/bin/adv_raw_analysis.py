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

#from data import preprocess, get_data_from_file
from dataPrep import preprocess, get_data_from_file
from pathlib import Path
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import os

from sklearn.metrics import recall_score
from decisionTreeNEW import retrain, decisionTree, compute_metrics as cmDT, adversarialTraining as advTrainDT #ADDESTRAMENTO NORMALE
from randomForest import randomForest, compute_metrics as cmRF, adversarialTraining as advTrainRF #ADDESTRAMENTO NORMALE
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import pairwise_distances

base_dir = str(Path().resolve().parent)

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

def write_Results_On_File(fileName, dict_X_Test):

    outputDirTXT = base_dir + '/updatedOutput/outputTXT_FILTERED_NotScaled_ADV_RAW'
    outputDirCSV = base_dir + '/updatedOutput/outputCSV_FILTERED_NotScaled_ADV_RAW'

    if (not os.path.exists(outputDirTXT)):
        os.mkdir(outputDirTXT)
    if (not os.path.exists(outputDirCSV)):
        os.mkdir(outputDirCSV)

    txtFile = outputDirTXT + '/' + fileName + '_ADV_RAW_Results.txt'
    csvFile = outputDirCSV + '/' + fileName + '_ADV_RAW_Results.csv'

    with open(txtFile, "w") as file1, open(csvFile,"w") as file2 :

        file1.write('Metric\t\tAdversarial Samples Before Sanity Check (adv_raw)\n')
        file1.write('Accuracy\t\t%s\n' % (str(dict_X_Test['accuracy'])))
        file1.write('Precision\t\t%s\n' % (str(dict_X_Test['precision'])))
        file1.write('Recall\t\t%s\n' % (str(dict_X_Test['recall'])))
        file1.write('F1\t\t%s' % (str(dict_X_Test['f1'])))


        file2.write('Metric, Adversarial Samples Before Sanity Check (adv_raw)\n')
        file2.write('Accuracy, %s\n' % (str(dict_X_Test['accuracy'])))
        file2.write('Precision, %s\n' % (str(dict_X_Test['precision'])))
        file2.write('Recall, %s\n' % (str(dict_X_Test['recall'])))
        file2.write('F1, %s' % (str(dict_X_Test['f1'])))


# SCRIPT PER IL CALCOLO INDIVIDUALE DELLE METRICHE PER ADV RAW
if __name__ == "__main__":

    file_test = base_dir + '/updatedInput/MOD-test.csv'
    dataset_test,_ = get_data_from_file(file_test)
    X_test, Y_test, _ = preprocess(dataset_test,True)

    file_test_adv = base_dir + '/updatedInput/ADVBASE1-baseAdv-test.csv'
    dataset_test_adv, _ = get_data_from_file(file_test_adv)
    X_adv, Y_adv, _ = preprocess(dataset_test_adv, True)


    input_dim = X_test.shape[1]  # restituisce le dimensioni della matrice. Qui restituiamo il numero di colonne
    # creaiamo istanza di MultiLayerAutoencoder

    mlp = MultilayerPerceptron(input_dim=input_dim)
    mlp.summary()
    dt, _, _, _, _, _ = decisionTree(gridSearch=False)  # contiene il calcolo delle metriche per il file MOD-test
    rf, _, _, _, _, _ = randomForest(gridSearch=False)  # contiene il calcolo delle metriche per il file MOD-test



    mypath=base_dir + '/updatedOutput/adv_RAW/'
    #read all files in updatedOutput
    fileList = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

    for fileName in fileList:
        print("Working on file: "+fileName)
        adv_raw_file = mypath+fileName
        dataset_ADV_RAW, _ = get_data_from_file(adv_raw_file)
        X_ADV_RAW, Y_ADV_RAW, _ = preprocess(dataset_ADV_RAW, True)

        prediction_adv_raw_DT = dt.predict(X_ADV_RAW)  # previous X_test
        prediction_adv_raw_RF = rf.predict(X_ADV_RAW)
        prediction_adv_raw_MLP = mlp.predict(X_ADV_RAW)

        dictionary_X_adv_raw_DT = cmDT(Y_ADV_RAW, prediction_adv_raw_DT)  # .flatten per il random forest
        dictionary_X_adv_raw_RF = cmRF(Y_ADV_RAW, prediction_adv_raw_RF)
        dictionary_X_adv_raw_MLP = mlp.evaluate(prediction_adv_raw_MLP, Y_ADV_RAW)

        fileNamePrefix = fileName[:len(fileName) - 4] # drop .csv suffix

        write_Results_On_File(fileNamePrefix + "_DecisionTree", dictionary_X_adv_raw_DT)
        write_Results_On_File(fileNamePrefix + "_RandomForest", dictionary_X_adv_raw_RF)
        write_Results_On_File(fileNamePrefix + "_MLP", dictionary_X_adv_raw_MLP)


    # *****CALCOLO RISULTATI ORIGINALI (X TEST)*****
    prediction_original_DT = dt.predict(X_test)  # previous X_test
    prediction_original_RF = rf.predict(X_test)
    prediction_original_MLP = mlp.predict(X_test)

    dictionary_X_Test_DT = cmDT(Y_test, prediction_original_DT)  # .flatten per il random forest
    dictionary_X_Test_RF = cmRF(Y_test, prediction_original_RF)
    dictionary_X_Test_MLP = mlp.evaluate(prediction_original_MLP, Y_test)

    printResults(dictionary_X_Test_DT, dictionary_X_Test_RF, dictionary_X_Test_MLP, headerString="*****CALCOLO RISULTATI MOD-test*****")

    prediction_adv_DT = dt.predict(X_adv)
    prediction_adv_RF = rf.predict(X_adv)
    prediction_adv_MLP = mlp.predict(X_adv)

    dictionary_X_adv_DT = cmDT(Y_adv, prediction_adv_DT)  # .flatten per il random forest
    dictionary_X_adv_RF = cmRF(Y_adv, prediction_adv_RF)
    dictionary_X_adv_MLP = mlp.evaluate(prediction_adv_MLP, Y_adv)

    printResults(dictionary_X_adv_DT, dictionary_X_adv_RF, dictionary_X_adv_MLP, headerString="*****CALCOLO RISULTATI ORIGINALI ADVBASE1-baseAdv-test*****")




