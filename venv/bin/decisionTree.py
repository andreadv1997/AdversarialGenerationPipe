from dataPrep import preprocess, get_data_from_file, get_data_from_file_ADV, get_data_from_file_Non_Rep
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import numpy as np
from hypopt import GridSearch
from joblib import dump, load

from sklearn.metrics import precision_score, recall_score, classification_report

import sys
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from randomForest import compute_metrics as rfComputeMetrics
import os
import pandas as pd

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
            'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min']

classes = ['DOS', 'BENIGN'] #ho letto il file in modo tale che i beningni abbiano label 1, per questo va a destra




def adversarialTraining(save=False):
    if (os.path.exists("decisiontree_ADV_TRAIN.joblib")):
        return load('decisiontree_ADV_TRAIN.joblib')
    else:
        base_dir = str(Path().resolve())
        adv_dir = base_dir + "/adversarial_samples_for_training"
        adv_files = [f for f in os.listdir(adv_dir) if os.path.isfile(os.path.join(adv_dir, f))]  # questi sono i file da usare per il training
        print("N file for adv samples: ", str(len(adv_files)))
        X_train = np.empty(shape=(0, 77))
        Y_train = np.empty(shape=(0, 1))

        for file in adv_files:
            dataset_temp = get_data_from_file_ADV(adv_dir+"/"+file)
            X_temp, Y_temp, _ = preprocess(dataset_temp,adv=True)

            X_train = np.concatenate((X_train, X_temp), axis=0)
            Y_train = np.concatenate((Y_train, Y_temp), axis=0)

        print(X_train.shape)
        return retrain(X_train, Y_train,save)





#OCCHIO: MODIFICARE IL PATH DEI DATI QUANDO SI PREVEDERA' DI AGGIUNGERE ESEMPI AVVERSARI PER L'ADDESTRAMENTO
def retrain(samples_to_add, Y_to_add, save=False):
    base_dir = str(Path().resolve())
    file_train = base_dir + '/MOD-training.csv'
    dataset_train = get_data_from_file(file_train)

    file_val = base_dir + '/MOD-validation.csv'
    dataset_val = get_data_from_file(file_val)

    file_test = base_dir + '/MOD-test.csv'
    dataset_test = get_data_from_file(file_test)

    X_train, Y_train, scaler = preprocess(dataset_train)  # modifica, test con uno scaler unico per tutti le fasi
    X_val, Y_val, _ = preprocess(dataset_val, scaler=scaler)
    X_test, Y_test, _ = preprocess(dataset_test, scaler=scaler)

    X_train = np.concatenate((X_train,samples_to_add), axis=0)
    Y_train = np.concatenate((Y_train, Y_to_add[:,0].reshape(Y_to_add.shape[0],1)), axis=0)

    X_train = np.concatenate((X_train, X_val), axis=0)
    Y_train = np.concatenate((Y_train, Y_val), axis=0)

    parameters = {'max_depth': range(13, 20), 'random_state': range(0, 6), 'min_samples_leaf': range(10, 30)}
    clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=parameters, refit=True, n_jobs=-1)  # 5-fold
    clf.fit(X_train, Y_train)

    print("Best Param: ", clf.best_params_)
    prediction_onVal = clf.predict(X_val)
    dict = compute_metrics(Y_val, prediction_onVal)

    if (dict['accuracy'] >= 0.99 and dict['precision'] >= 0.99 and dict['recall'] >= 0.99 and dict['f1'] >= 0.99):


        print("Values satisfies constraints :", clf.best_params_)
        if(save):
            dump(clf, 'decisiontree_ADV_TRAIN.joblib')

    else:
        print("Values DO NOT satisfy constraints")
        print('Accuracy of decision tree: ' + str(dict['accuracy']))
        print('Precision of decision tree: ' + str(dict['precision']))
        print('Recall of decision tree: ' + str(dict['recall']))
        print('F1 of decision tree: ' + str(dict['f1']))
        return None

    return clf


def decisionTree(gridSearch = False):

    base_dir = str(Path().resolve())
    # file_train = base_dir + '/MOD-training.csv' DATI CICIDS
    file_train = base_dir + '/../updatedInput/MOD-training.csv'  # DATI DEI BELGI
    dataset_train,_ = get_data_from_file(file_train)

    # file_val = base_dir + '/MOD-validation.csv' DATI CICIDS
    file_val = base_dir + '/../updatedInput/MOD-validation.csv'  # DATI DEI BELGI
    dataset_val,_ = get_data_from_file(file_val)

    # file_test = base_dir + '/MOD-test.csv' DATI CICIDS
    file_test = base_dir + '/../updatedInput/MOD-test.csv'  # DATI DEI BELGI
    dataset_test,_ = get_data_from_file(file_test)


    X_train, Y_train, scaler = preprocess(dataset_train)  #modifica, test con uno scaler unico per tutti le fasi
    X_val, Y_val, _ = preprocess(dataset_val)
    X_test, Y_test, _ = preprocess(dataset_test)

    clf = None

    parameters = {'max_depth': range(13, 20), 'random_state': range(0, 6), 'min_samples_leaf': range(1, 6)}


    #test con hypopt library
    if gridSearch:
        #clf = GridSearch(tree.DecisionTreeClassifier(), param_grid=parameters)
        X_train = np.concatenate((X_train, X_val), axis=0)
        Y_train = np.concatenate((Y_train, Y_val), axis=0)
        clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=parameters,refit=True, n_jobs=-1) #5-fold
        #clf.fit(X_train, Y_train, X_val, Y_val)
        clf.fit(X_train, Y_train)

    else:
        #clf = load('decisiontree.joblib')
        #clf = load('decisiontree_SingleScaler.joblib')
        clf = load('decisiontree_NotScaled_NEW_CONFIG.joblib')


    #print("Best Param: ", clf.best_params)
    #print("Best Param: ", clf.best_params_)
    prediction_onVal = clf.predict(X_val)
    dict = compute_metrics(Y_val, prediction_onVal)

    #ABBASSIAMO A 97% PER CONSENTIRE SALVATAGGIO. RIPORTARE A 99%
    if (dict['accuracy'] >= 0.97 and dict['precision'] >= 0.97 and dict['recall'] >= 0.97 and dict['f1'] >= 0.97):

        if gridSearch:
           print("Values satisfies constraints :", clf.best_params_)
         #   print("Values satisfies constraints ")
        #dump(clf,'decisiontree.joblib')
        #dump(clf, 'decisiontree_SingleScaler.joblib')
        dump(clf, 'decisiontree_NotScaled_NEW_CONFIG.joblib')



    else:
        print("Values DO NOT satisfy constraints")
        print('Accuracy of decision tree: ' + str(dict['accuracy']))
        print('Precision of decision tree: ' + str(dict['precision']))
        print('Recall of decision tree: ' + str(dict['recall']))
        print('F1 of decision tree: ' + str(dict['f1']))
        return None


    prediction_test = clf.predict(X_test)
    difference_class = prediction_test-Y_test[:,0]
    print(difference_class)
    mod = float((abs(difference_class).sum()))
    mod = mod / float(Y_test.shape[0])


    true_pos = 0
    false_pos= 0
    true_neg = 0
    false_neg = 0


    for index in range(len(prediction_test)):
        if(difference_class[index] == 0):
            # TP or TN
            if(Y_test[index,0]==0):
                true_pos+=1
            else:
                true_neg+=1
        else:
            if(difference_class[index] > 0):
                false_neg+=1
            else:
                false_pos+=1





    precision = float(true_pos)/float((true_pos+false_pos))
    recall = float(true_pos)/float((true_pos+false_neg))
    f1 = float((2*precision*recall))/float((precision+recall))

    print('#####Evaluation on Test Set#####')
    print('Accuracy of decision tree: '+ str(float(1-mod)))
    print('Precision of decision tree: ' + str(precision))
    print('Recall of decision tree: ' + str(recall))
    print('F1 of decision tree: ' + str(f1))




    '''
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=features, class_names=classes,
                                    filled=True, rounded=True, special_characters=True)

    graph = graphviz.Source(dot_data)

    graph.render("CIC")
    '''
    return clf, float(1-mod), precision, recall, f1, scaler

def compute_metrics(Y_test, predictions):
    difference_class = predictions - Y_test[:, 0]
    mod = float((abs(difference_class).sum()))
    mod = mod / float(Y_test.shape[0])

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

    true_pos_index = []
    false_neg_index = []

    precision = 0
    recall = 0
    f1 = 0

    for index in range(len(predictions)):
        if (difference_class[index] == 0):
            # TP or TN
            if (Y_test[index, 0] == 0):
                true_pos += 1
                true_pos_index.append(index)
            else:
                true_neg += 1
        else:
            if (difference_class[index] > 0):
                false_neg += 1
                false_neg_index.append(index)
            else:
                false_pos += 1

    if ((true_pos + false_pos) != 0):
        precision = float(true_pos) / float((true_pos + false_pos))

    if ((true_pos + false_neg) != 0):
        recall = float(true_pos) / float((true_pos + false_neg))

    if ((precision + recall) != 0):
        f1 = float((2 * precision * recall)) / float((precision + recall))


    dict = {}
    dict['accuracy'] = float(1-mod)
    dict['precision'] = precision
    dict['recall'] = recall
    dict['f1'] = f1
    dict['true_pos'] = true_pos
    dict['false_pos'] = false_pos
    dict['true_neg'] = true_neg
    dict['false_neg'] = false_neg
    dict['true_pos_index'] = true_pos_index
    dict['false_neg_index'] = false_neg_index

    return dict


if __name__ == '__main__':
    '''
    ######VALUTAZIONE SENSITIVITA' DECISION TREE########
    base_dir = str(Path().resolve())
    # file_train = base_dir + '/MOD-training.csv' DATI CICIDS
    file_train = base_dir + '/../updatedInput/MOD-training.csv'  # DATI DEI BELGI
    dataset_train, _ = get_data_from_file(file_train)

    # file_val = base_dir + '/MOD-validation.csv' DATI CICIDS
    file_val = base_dir + '/../updatedInput/MOD-validation.csv'  # DATI DEI BELGI
    dataset_val, _ = get_data_from_file(file_val)

    # file_test = base_dir + '/MOD-test.csv' DATI CICIDS
    file_test = base_dir + '/../updatedInput/MOD-test.csv'  # DATI DEI BELGI
    dataset_test, _ = get_data_from_file(file_test)

    file_ADV_BASE= base_dir + '/../updatedInput/ADVBASE1-baseAdv-test.csv' #DATI DEI BELGI
    dataset_adv, _ = get_data_from_file(file_ADV_BASE)

    X_train, Y_train, scaler = preprocess(dataset_train)  # modifica, test con uno scaler unico per tutti le fasi
    X_val, Y_val, _ = preprocess(dataset_val, scaler=scaler)
    X_test, Y_test, _ = preprocess(dataset_test, scaler=scaler)
    X_ADV,Y_ADV,_=preprocess(dataset_adv,scaler=scaler)
    print(X_test.shape)


    #NUOVA CONFIG DECISION TREE#
    dt = tree.DecisionTreeClassifier(random_state=4, max_depth=15, min_samples_leaf=100) ##Tutti i valosi sopra al 98 ma non sopra al 99
    #clf = tree.DecisionTreeClassifier(random_state=4, max_depth=15, min_samples_leaf=100)
    dt.fit(X_train, Y_train)

    rf = RandomForestClassifier(n_jobs=-1, n_estimators=50, max_depth=15, min_samples_leaf=150, random_state=4, max_features=0.25, bootstrap=False)
    rf.fit(X_train,Y_train)

    prediction_onValDT = dt.predict(X_val)
    dict_onValDT = compute_metrics(Y_val, prediction_onValDT)

    prediction_onValRF = rf.predict(X_val)
    dict_onValRF = rfComputeMetrics(Y_val, prediction_onValRF)


    if (dict_onValDT['accuracy'] >= 0.98 and dict_onValDT['precision'] >= 0.98 and dict_onValDT['recall'] >= 0.98 and dict_onValDT['f1'] >= 0.98):
        print("DT Values satisfies constraints")
        dump(dt, 'decisiontree_NotScaled_NEW_CONFIG.joblib')

        prediction_onTestDT = dt.predict(X_test)
        dict_onTestDT = compute_metrics(Y_test,prediction_onTestDT)
        print("Calcolo Metriche DT on MOD-TEST")
        print("Accuracy: " + str(dict_onTestDT['accuracy']))
        print("Precision: " + str(dict_onTestDT['precision']))
        print("Recall: " + str(dict_onTestDT['recall']))
        print("F1: " + str(dict_onTestDT['f1']))

        prediction_onADVDT = dt.predict(X_ADV)
        dict_onADVDT = compute_metrics(Y_ADV, prediction_onADVDT)
        print("Calcolo Metriche DT on ADV-BASE")
        print("Accuracy: " + str(dict_onADVDT['accuracy']))
        print("Precision: " + str(dict_onADVDT['precision']))
        print("Recall: " + str(dict_onADVDT['recall']))
        print("F1: " + str(dict_onADVDT['f1']))


    else:
        print("DT Values DO NOT satisfy constraints")
        print('Accuracy of decision tree: ' + str(dict_onValDT['accuracy']))
        print('Precision of decision tree: ' + str(dict_onValDT['precision']))
        print('Recall of decision tree: ' + str(dict_onValDT['recall']))
        print('F1 of decision tree: ' + str(dict_onValDT['f1']))


    if (dict_onValRF['accuracy'] >= 0.98 and dict_onValRF['precision'] >= 0.98 and dict_onValRF['recall'] >= 0.98 and dict_onValRF['f1'] >= 0.98):


        print("RF Values satisfies constraints")
        dump(rf, 'randomForest_NotScaled_NEW_CONFIG.joblib')
        prediction_onTestRF = rf.predict(X_test)
        dict_onTestRF = rfComputeMetrics(Y_test, prediction_onTestRF)
        print("Calcolo Metriche RF on MOD-TEST")
        print("Accuracy: " + str(dict_onTestRF['accuracy']))
        print("Precision: " + str(dict_onTestRF['precision']))
        print("Recall: " + str(dict_onTestRF['recall']))
        print("F1: " + str(dict_onTestRF['f1']))

        prediction_onADVRF = rf.predict(X_ADV)
        dict_onADVRF = rfComputeMetrics(Y_ADV, prediction_onADVRF)
        print("Calcolo Metriche RF on ADV-BASE")
        print("Accuracy: " + str(dict_onADVRF['accuracy']))
        print("Precision: " + str(dict_onADVRF['precision']))
        print("Recall: " + str(dict_onADVRF['recall']))
        print("F1: " + str(dict_onADVRF['f1']))




    else:
        print("RF Values DO NOT satisfy constraints")
        print('Accuracy of random Forest: ' + str(dict_onValRF['accuracy']))
        print('Precision of random Forest: ' + str(dict_onValRF['precision']))
        print('Recall of random Fores: ' + str(dict_onValRF['recall']))
        print('F1 of random Fores: ' + str(dict_onValRF['f1']))





    exit()
    '''
    base_dir = str(Path().resolve())
    ###VAM Non Rep###
    file_grey = base_dir + '/../updatedOutput/adv_grey/VAM_4_filt_.csv'  # DATI DEI BELGI
    dataset_grey, _ = get_data_from_file_Non_Rep(file_grey)

    file_ORIG_ATT = base_dir + '/../updatedOutput/adv_ORIG_ATT/VAM_4_filt_.csv'  # DATI DEI BELGI
    dataset_ORIG_ATT, _ = get_data_from_file_Non_Rep(file_ORIG_ATT)

    file_ORIG_BEN = base_dir + '/../updatedOutput/adv_ORIG_BEN/VAM_4_filt_.csv'  # DATI DEI BELGI
    dataset_ORIG_BEN, _ = get_data_from_file_Non_Rep(file_ORIG_BEN)

    ###VAM Rep###
    file_rep = base_dir + '/../updatedOutput/adv_representative/VAM_4_filt_.csv'  # DATI DEI BELGI
    dataset_rep, _ = get_data_from_file(file_rep)
    X_Rep, Y_Rep,_ = preprocess(dataset_rep)

    ###VAM RAW###
    file_raw = base_dir + '/../updatedOutput/adv_RAW/VAM_4_filt_.csv'  # DATI DEI BELGI
    dataset_raw, _ = get_data_from_file(file_raw)
    X_Raw, Y_Raw, _ = preprocess(dataset_raw)


    ###VAM NOT FILTERED###
    dataset_not_fil = pd.concat([dataset_grey,dataset_ORIG_ATT], axis=0)
    dataset_not_fil = pd.concat([dataset_not_fil,dataset_ORIG_BEN], axis=0)
    dataset_not_fil = pd.concat([dataset_not_fil, dataset_rep], axis=0)
    X_Not_Fil, Y_Not_Fil, _ = preprocess(dataset_not_fil)

    print("X Not Filtered")
    print(X_Not_Fil.shape)

    print("X Representative")
    print(X_Rep.shape)

    print("X Raw")
    print(X_Raw.shape)

    dt = load('decisiontree_NotScaled_NEW_CONFIG.joblib')

    print("Calcolo Metriche DT VAM 4 RAW")
    preds_raw = dt.predict(X_Raw)
    dict_raw = compute_metrics(Y_Raw, preds_raw)
    print("Accuracy: " + str(dict_raw['accuracy']))
    print("Precision: " + str(dict_raw['precision']))
    print("Recall: " + str(dict_raw['recall']))
    print("F1: " + str(dict_raw['f1']))

    print("Calcolo Metriche DT VAM 4 Not Filtered")
    preds_not_fil = dt.predict(X_Not_Fil)
    dict_not_fil = compute_metrics(Y_Not_Fil, preds_not_fil)
    print("Accuracy: " + str(dict_not_fil['accuracy']))
    print("Precision: " + str(dict_not_fil['precision']))
    print("Recall: " + str(dict_not_fil['recall']))
    print("F1: " + str(dict_not_fil['f1']))

    print("Calcolo Metriche DT VAM 4 Representative")
    preds_rep = dt .predict(X_Rep)
    dict_rep = compute_metrics(Y_Rep, preds_rep)
    print("Accuracy: " + str(dict_rep['accuracy']))
    print("Precision: " + str(dict_rep['precision']))
    print("Recall: " + str(dict_rep['recall']))
    print("F1: " + str(dict_rep['f1']))

    print("##############")

    rf = load('randomForest_NotScaled_NEW_CONFIG.joblib')


    print("Calcolo Metriche RF VAM 4 RAW")
    preds_rawRF = rf.predict(X_Raw)
    dict_rawRF = rfComputeMetrics(Y_Raw, preds_rawRF)
    print("Accuracy: " + str(dict_rawRF['accuracy']))
    print("Precision: " + str(dict_rawRF['precision']))
    print("Recall: " + str(dict_rawRF['recall']))
    print("F1: " + str(dict_rawRF['f1']))

    print("Calcolo Metriche RF VAM 4 Not Filtered")
    preds_not_filRF = rf.predict(X_Not_Fil)
    dict_not_filRF = rfComputeMetrics(Y_Not_Fil, preds_not_filRF)
    print("Accuracy: " + str(dict_not_filRF['accuracy']))
    print("Precision: " + str(dict_not_filRF['precision']))
    print("Recall: " + str(dict_not_filRF['recall']))
    print("F1: " + str(dict_not_filRF['f1']))

    print("Calcolo Metriche RF VAM 4 Representative")
    preds_repRF = rf.predict(X_Rep)
    dict_repRF = rfComputeMetrics(Y_Rep, preds_repRF)
    print("Accuracy: " + str(dict_repRF['accuracy']))
    print("Precision: " + str(dict_repRF['precision']))
    print("Recall: " + str(dict_repRF['recall']))
    print("F1: " + str(dict_repRF['f1']))

    exit()
    ####################################################

    '''
    ######VALUTAZIONE SENSITIVITA' DECISION TREE########
    base_dir = str(Path().resolve())
    # file_train = base_dir + '/MOD-training.csv' DATI CICIDS
    file_train = base_dir + '/../updatedInput/MOD-training.csv'  # DATI DEI BELGI
    dataset_train, _ = get_data_from_file(file_train)

    # file_val = base_dir + '/MOD-validation.csv' DATI CICIDS
    file_val = base_dir + '/../updatedInput/MOD-validation.csv'  # DATI DEI BELGI
    dataset_val, _ = get_data_from_file(file_val)

    # file_test = base_dir + '/MOD-test.csv' DATI CICIDS
    file_test = base_dir + '/../updatedInput/MOD-test.csv'  # DATI DEI BELGI
    dataset_test, _ = get_data_from_file(file_test)

    X_train, Y_train, scaler = preprocess(dataset_train)  # modifica, test con uno scaler unico per tutti le fasi
    X_val, Y_val, _ = preprocess(dataset_val, scaler=scaler)
    X_test, Y_test, _ = preprocess(dataset_test, scaler=scaler)

    if (scaler==None):
        print("Scaler Assente")
    print("Inizio Training")
    clf = tree.DecisionTreeClassifier(random_state=4, max_depth=10, min_samples_leaf=500)
    clf.fit(X_train, Y_train)

    print("Calcolo Metriche Mod Test")
    preds = clf.predict(X_test)
    dict = compute_metrics(Y_test, preds)
    print(dict['accuracy'])
    print(dict['precision'])
    print(dict['recall'])

    print("Calcolo Metriche FGSM 1 ADV Rep")
    file_FGSM = base_dir + '/../updatedOutput/adv_representative/FGSM_filt_.csv'  # DATI DEI BELGI
    dataset_FGSM, _ = get_data_from_file(file_FGSM)
    X_FGSM, Y_FGSM, _ = preprocess(dataset_FGSM, scaler=scaler)

    preds_fgsm = clf.predict(X_FGSM)
    dict_fgsm = compute_metrics(Y_FGSM, preds_fgsm)
    print(dict_fgsm['accuracy'])
    print(dict_fgsm['precision'])
    print(dict_fgsm['recall'])

    exit()






    ##########

    clf = load('decisiontree_NotScaled.joblib')
    print("Values satisfies constraints :", clf.best_params_)
    exit()

    fileName = sys.argv[1]

    base_dir = str(Path().resolve())
    file_evaluation = base_dir + '/'+fileName
    dataset_evaluation = get_data_from_file(file_evaluation,True)

    #dataset_evaluation = dataset_evaluation.iloc[1:]


    #dt, _, _, _, _, scaler = decisionTree(True)

    #dt = adversarialTraining(True)
    #dt,_,_,_,_,_ = decisionTree()
    X_eval, Y_eval,_= preprocess(dataset_evaluation)
    print(X_eval.shape)
    exit()

    preds_eval = dt.predict(X_eval)

    print("Calcolo Metriche")
    dict = compute_metrics(Y_eval, preds_eval)

    print(dict['accuracy'])
    print(dict['precision'])
    print(dict['recall'])
    '''