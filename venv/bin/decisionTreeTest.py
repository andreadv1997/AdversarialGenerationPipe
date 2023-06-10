from dataPrep import preprocess, get_data_from_file
from sklearn import tree
from sklearn.tree import export
import numpy as np
import graphviz
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


def decisionTree():


    base_dir = str(Path().resolve().parent)
    file = base_dir + '/bin/MOD-training.csv'
    dataset = get_data_from_file(file)

    X, Y, _ = preprocess(dataset)

    file2 = base_dir + '/bin/MOD-test.csv'
    dataset2 = get_data_from_file(file2)

    X_test, Y_test, _ = preprocess(dataset2)

    file3 = base_dir + '/bin/MOD-validation.csv'
    dataset3 = get_data_from_file(file3)

    X_val, Y_val, _ = preprocess(dataset3)

    print("Train: " + str(X.shape) + "\n")
    print("Test: " + str(X_test.shape) + "\n")
    print("Val: " + str(X_val.shape) + "\n")



    count = 0

    print(str(X_test.shape))
    for i in range(Y.shape[0]):
        if(Y[i,0] == 1):
            count = count +1

    #print(Y.shape)
    #print(count)
    clf = tree.DecisionTreeClassifier(max_depth=15, random_state=3, min_samples_leaf=3)
    #clf = RandomForestClassifier(max_depth=16, random_state=4, min_samples_leaf=10, n_estimators=20, max_features=77)
    #clf = clf.fit(X,Y.reshape(X.shape[0],))
    clf = clf.fit(X, Y)
    prediction_test = clf.predict(X_test)
    difference_class = prediction_test-Y_test[:,0]
    mod = float((abs(difference_class).sum()))
    mod = mod / float(Y_test.shape[0])

    true_pos = 0
    false_pos= 0
    true_neg = 0
    false_neg = 0

    print(str(len(prediction_test)))
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
    return clf

def compute_metrics(Y_test, predictions):
    difference_class = predictions - Y_test[:, 0]
    mod = float((abs(difference_class).sum()))
    mod = mod / float(Y_test.shape[0])

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0


    for index in range(len(predictions)):
        if (difference_class[index] == 0):
            # TP or TN
            if (Y_test[index, 0] == 0):
                true_pos += 1
            else:
                true_neg += 1
        else:
            if (difference_class[index] > 0):
                false_neg += 1
            else:
                false_pos += 1

    precision = float(true_pos) / float((true_pos + false_pos))
    recall = float(true_pos) / float((true_pos + false_neg))
    f1 = float((2 * precision * recall)) / float((precision + recall))

    print('Accuracy of decision tree: ' + str(float(1 - mod)))
    print('Precision of decision tree: ' + str(precision))
    print('Recall of decision tree: ' + str(recall))
    print('F1 of decision tree: ' + str(f1))




if __name__ == '__main__':


    decisionTree()