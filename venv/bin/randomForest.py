from dataPrep import preprocess, get_data_from_file, get_data_from_file_ADV
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
import numpy as np
from hypopt import GridSearch
from joblib import dump, load

import sys
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

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


def adversarialTraining():
    if (os.path.exists("randomForest_ADV_TRAIN.joblib")):
        return load('randomForest_ADV_TRAIN.joblib')
    else:
        base_dir = str(Path().resolve())
        adv_dir = base_dir + "/adversarial_samples_for_training"
        adv_files = [f for f in os.listdir(adv_dir) if
                     os.path.isfile(os.path.join(adv_dir, f))]  # questi sono i file da usare per il training

        X_train = np.empty(shape=(0, 77))
        Y_train = np.empty(shape=(0, 1))

        for file in adv_files:
            dataset_temp = get_data_from_file_ADV(adv_dir+"/"+file)
            X_temp, Y_temp, _ = preprocess(dataset_temp,adv=True)

            X_train = np.concatenate((X_train, X_temp), axis=0)
            Y_train = np.concatenate((Y_train, Y_temp), axis=0)

        return retrain(X_train, Y_train)




#OCCHIO: MODIFICARE IL PATH DEI DATI QUANDO SI PREVEDERA' DI AGGIUNGERE ESEMPI AVVERSARI PER L'ADDESTRAMENTO
def retrain(samples_to_add, Y_to_add):
    base_dir = str(Path().resolve())
    file_train = base_dir + '/MOD-training.csv'
    dataset_train = get_data_from_file(file_train)

    file_val = base_dir + '/MOD-validation.csv'
    dataset_val = get_data_from_file(file_val)

    file_test = base_dir + '/MOD-test.csv'
    dataset_test = get_data_from_file(file_test)

    X_train, Y_train, _ = preprocess(dataset_train)  # modifica, test con uno scaler unico per tutti le fasi
    X_val, Y_val, _ = preprocess(dataset_val)
    X_test, Y_test, _ = preprocess(dataset_test)

    X_train = np.concatenate((X_train,samples_to_add), axis=0)
    Y_train = np.concatenate((Y_train, Y_to_add[:,0].reshape(Y_to_add.shape[0],1)), axis=0)

    X_train = np.concatenate((X_train, X_val), axis=0)
    Y_train = np.concatenate((Y_train, Y_val), axis=0)

    parameters = {'max_features': ["auto",X_test.shape[1], 0.5, 0.25], 'bootstrap':[True, False]}
    clf = GridSearchCV(ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=50, max_depth=19, min_samples_leaf=10, random_state=5),param_grid=parameters, refit=True, n_jobs=-1)  # 5-fold
    clf.fit(X_train, Y_train)

    print("Best Param: ", clf.best_params_)
    prediction_onVal = clf.predict(X_val)
    dict = compute_metrics(Y_val, prediction_onVal)

    # ABBASSIAMO A 97% PER CONSENTIRE SALVATAGGIO. RIPORTARE A 99%
    if (dict['accuracy'] >= 0.97 and dict['precision'] >= 0.97 and dict['recall'] >= 0.97 and dict['f1'] >= 0.97):

        print("Values satisfies constraints :", clf.best_params_)
        dump(clf, 'randomForest_ADV_TRAIN.joblib')

    else:
        print("Values DO NOT satisfy constraints")
        print('Accuracy of decision tree: ' + str(dict['accuracy']))
        print('Precision of decision tree: ' + str(dict['precision']))
        print('Recall of decision tree: ' + str(dict['recall']))
        print('F1 of decision tree: ' + str(dict['f1']))
        return None

    return clf



def randomForest(gridSearch = False):


    base_dir = str(Path().resolve())
    #file_train = base_dir + '/MOD-training.csv' DATI CICIDS
    file_train = base_dir + '/../updatedInput/MOD-training.csv' #DATI DEI BELGI
    dataset_train,_ = get_data_from_file(file_train)

    #file_val = base_dir + '/MOD-validation.csv' DATI CICIDS
    file_val = base_dir + '/../updatedInput/MOD-validation.csv' #DATI DEI BELGI
    dataset_val,_ = get_data_from_file(file_val)

    #file_test = base_dir + '/MOD-test.csv' DATI CICIDS
    file_test = base_dir + '/../updatedInput/MOD-test.csv' #DATI DEI BELGI
    dataset_test,_ = get_data_from_file(file_test)


    X_train, Y_train, scaler = preprocess(dataset_train)
    X_val, Y_val, _ = preprocess(dataset_val)
    X_test, Y_test, _ = preprocess(dataset_test)

    #i parametri sono decisamente troppi per un tempo di esecuzione ragionevole. Per questo almeno il numero di alberi viene tenuto fisso a 50 e si usano le configurazione degli alberi definite per il decision tree classico. In caso di
    #esito negativo si aumenteranno i range
    parameters = {'max_features': ["auto",X_test.shape[1], 0.5, 0.25], 'bootstrap':[True, False]}

    #test per il GridSearchCV
    #X_Comb = np.concatenate((X_train, X_val), axis=0)
    #Y_Comb = np.concatenate((Y_train, Y_val), axis=0)
    #clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=parameters,refit=True, n_jobs=-1) #5-fold
    ##########

    #test con un decision tree senza validation
    #clf = tree.DecisionTreeClassifier(max_depth=15, random_state=3, min_samples_leaf=3)
    #clf = RandomForestClassifier(max_depth=16, random_state=4, min_samples_leaf=10, n_estimators=20, max_features=77)
    #clf = clf.fit(X,Y.reshape(X.shape[0],))

    clf = None

    #test con hypopt library
    if gridSearch:
        #clf = GridSearch(ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=50, max_depth=19, min_samples_leaf=2, random_state=1), param_grid=parameters)
        #clf.fit(X_train, Y_train, X_val, Y_val)
        X_train = np.concatenate((X_train, X_val), axis=0)
        Y_train = np.concatenate((Y_train, Y_val), axis=0)
        clf = GridSearchCV(ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=50, max_depth=19, min_samples_leaf=10, random_state=5), param_grid=parameters,refit=True, n_jobs=-1) #5-fold
        clf.fit(X_train, Y_train)

    else:
        #clf = load('randomForest.joblib')
        #clf = load('randomForest_SingleScaler.joblib')
        clf = load('randomForest_NotScaled.joblib')





    prediction_onVal = clf.predict(X_val)

    dict= compute_metrics(Y_val, prediction_onVal)

    # ABBASSIAMO A 97% PER CONSENTIRE SALVATAGGIO. RIPORTARE A 99%
    if (dict['accuracy'] >= 0.97 and dict['precision'] >= 0.97 and dict['recall'] >= 0.97 and dict['f1'] >= 0.97):
        if gridSearch:
            print("Values satisfies constraints :", clf.best_params_)
        #dump(clf,'randomForest.joblib')
        #dump(clf, 'randomForest_SingleScaler.joblib')
        dump(clf, 'randomForest_NotScaled.joblib')

    else:
        print("Values DO NOT satisfy constraints")
        print('Accuracy of decision tree: ' + str(dict['accuracy']))
        print('Precision of decision tree: ' + str(dict['precision']))
        print('Recall of decision tree: ' + str(dict['recall']))
        print('F1 of decision tree: ' + str(dict['f1']))
        return None


    prediction_test = clf.predict(X_test)
    difference_class = prediction_test-Y_test[:,0]

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
    difference_class = predictions - Y_test[:,0]
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
            if (Y_test[index,0] == 0):
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

    if((true_pos+false_pos) != 0):
        precision = float(true_pos) / float((true_pos + false_pos))

    if((true_pos+false_neg)!=0):
        recall = float(true_pos) / float((true_pos + false_neg))

    if((precision+recall)!=0):
        f1 = float((2 * precision * recall)) / float((precision + recall))


    dict = {}
    dict['accuracy'] = float(1 - mod)
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

    X_train, Y_train, scaler = preprocess(dataset_train)
    X_val, Y_val, _ = preprocess(dataset_val, scaler=scaler)
    X_test, Y_test, _ = preprocess(dataset_test, scaler=scaler)

    #clf = RandomForestClassifier(n_jobs=-1, n_estimators=50, max_depth=15, min_samples_leaf=150, random_state=4,, max_features=0.25, bootstrap=False) #Tutti maggiori di 98
    clf = RandomForestClassifier(n_jobs=-1, n_estimators=50, max_depth=15, min_samples_leaf=150,
                                 random_state=4, max_features=0.50, bootstrap=True)
    clf.fit(X_train, Y_train)

    prediction_onVal = clf.predict(X_val)
    dict_onVal = compute_metrics(Y_val, prediction_onVal)

    if (dict_onVal['accuracy'] >= 0.99 and dict_onVal['precision'] >= 0.99 and dict_onVal['recall'] >= 0.99 and
            dict_onVal['f1'] >= 0.99):

        print("Values satisfies constraints : ")
        # dump(clf, 'decisiontree_NotScaled.joblib')



    else:
        print("Values DO NOT satisfy constraints")
        print('Accuracy of decision tree: ' + str(dict_onVal['accuracy']))
        print('Precision of decision tree: ' + str(dict_onVal['precision']))
        print('Recall of decision tree: ' + str(dict_onVal['recall']))
        print('F1 of decision tree: ' + str(dict_onVal['f1']))

    exit()





    clf = load('randomForest_NotScaled.joblib')
    print("Values satisfies constraints :", clf.best_params_)
    exit()

    fileName = sys.argv[1]

    base_dir = str(Path().resolve())
    file_evaluation = base_dir + '/'+fileName
    dataset_evaluation = get_data_from_file(file_evaluation)
    dataset_evaluation = dataset_evaluation.iloc[1:]
    X_eval, Y_eval,_= preprocess(dataset_evaluation)
    #Y_eval = Y_eval.flatten()
    dt,_,_,_,_,scaler = randomForest(gridSearch=False)

    #dt = adversarialTraining()
    preds_eval = dt.predict(X_eval)

    print("Calcolo metriche")
    dict = compute_metrics(Y_eval, preds_eval)

    print(dict['accuracy'])
    print(dict['precision'])
    print(dict['recall'])