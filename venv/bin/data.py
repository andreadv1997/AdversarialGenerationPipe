# -*- coding: iso-8859-15 -*-
from __future__ import unicode_literals
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

import pandas as pd
import numpy as np
import sys

np.random.seed(101)

names = ['Flow_ID', 'Source_IP', 'Source_Port', 'Destination_IP', 'Destination_Port', 'Protocol', 'Timestamp',
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
         'Fwd_Header_Length', 'Fwd_Avg_Bytes/Bulk', 'Fwd_Avg_Packets/Bulk', 'Fwd_Avg_Bulk_Rate', 'Bwd_Avg_Bytes/Bulk',
         'Bwd_Avg_Packets/Bulk', 'Bwd_Avg_Bulk_Rate', 'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes', 'Subflow_Bwd_Packets',
         'Subflow_Bwd_Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
         'min_seg_size_forward',
         'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min',
         'Label']


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
            'Fwd_Header_Length', 'Fwd_Avg_Bytes/Bulk', 'Fwd_Avg_Packets/Bulk', 'Fwd_Avg_Bulk_Rate',
            'Bwd_Avg_Bytes/Bulk',
            'Bwd_Avg_Packets/Bulk', 'Bwd_Avg_Bulk_Rate', 'Subflow_Fwd_Packets', 'Subflow_Fwd_Bytes',
            'Subflow_Bwd_Packets',
            'Subflow_Bwd_Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
            'min_seg_size_forward',
            'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min']

def get_data_from_file(file):
    #print(file)
    df = pd.read_csv(file, names=names, sep=',', dtype='unicode',skiprows=1) #creaimao dataframe: tabella
    df['Label'] = np.where(df['Label']=='BENIGN',1,0) #seleziono colonna dataframe marcata label. Quando trovi Benign, metti 0, altimenti metti 1

    return df

    #utilizzo separatore virgola e carattere unicode. Uso pandas come libreria. 01:35 valore di ritorno di df è una tabella: ossia valore tabellato del dataset

def preprocess(df_train, test_size=0.2):  #prendo dataframe destinato al training e il parametro che corrisponde al test size(dimensione del test set è 0.2)
    random_seed= 101
    x_train, x_test = train_test_split(df_train, test_size=test_size, random_state=random_seed) #dato l'interno dataset, df_train lo usi per fare training. Un'altra porzione corrispondente a test_size per test
          #random_seed split 80--20 viene preso sempre allo stesso modo. Uso questa cosa per avere riproducibilità tra esecuzioni successive
          #per convenione si sceglie un intero o 42 o 101. Faccio in modo che le partizioni siano sempre le stesse tra le diverse esecuzioni
    #x_train = x_train[x_train['Label']== 0]# dato il training set, seleziono solo i record che hanno 0 nella colonna Label
    y_train = x_train['Label']
    y_train_neg = np.where(y_train == 1,0,1) #seleziono colonna dataframe marcata label. Quando trovi Benign, metti 0, altimenti metti 1

    y_train = np.expand_dims(y_train,axis=1)
    y_train_neg = np.expand_dims(y_train_neg, axis=1)

    y_train = np.concatenate((y_train,y_train_neg),axis=1)

    #print(temp)



    x_train = x_train.drop(['Label'], axis=1) #axis = 1 stiamo dicendo che stiamo togliendo la colonna selezionata


    y_test = x_test['Label']
    y_test_neg = np.where(y_test == 1, 0, 1)  # seleziono colonna dataframe marcata label. Quando trovi Benign, metti 0, altimenti metti 1

    y_test = np.expand_dims(y_test, axis=1)
    y_test_neg = np.expand_dims(y_test_neg, axis=1)

    y_test = np.concatenate((y_test, y_test_neg), axis=1)

    x_test = x_test.drop(['Label'], axis=1)

    x_train = x_train[features]
    x_test= x_test[features]

    #trasformo tabella/dataframe in vettore
    x_train = x_train.values
    x_test = x_test.values

    #Normalizzazione sui dati
    #scaler = MaxAbsScaler()
    #x_train_scaled = scaler.fit_transform(x_train) #fitto e calcolo i parametri, incapsulati nella variabile scaler
    #x_test_scaled = scaler.transform(x_test) #qui uso direttamente scaler


    #x_test = np.delete(x_test,(0),axis=0)
    #x_train = np.delete(x_train, (0), axis=0)


    #print(x_test[0:1000])
    #print("°°°°°°")
    #np.set_printoptions(threshold=np.inf)
    #print(x_train)



    x_test = x_test.astype(np.float)

    nanArr = np.isnan(x_test)
    infArr = np.isinf(x_test)
    negInfArr = np.isneginf(x_test)
    print(x_test.shape)
    x_test = np.delete(x_test, np.where(nanArr), 0)
    y_test = np.delete(y_test, np.where(nanArr), 0)

    x_test = np.delete(x_test, np.where(infArr), 0)
    y_test = np.delete(y_test, np.where(infArr), 0)

    x_test = np.delete(x_test, np.where(negInfArr), 0)
    y_test = np.delete(y_test, np.where(negInfArr), 0)
    print(x_test.shape)

    x_train = x_train.astype(np.float)

    nanArr = np.isnan(x_train)
    infArr = np.isinf(x_train)
    negInfArr = np.isneginf(x_train)

    print(x_train.shape)
    x_train = np.delete(x_train, np.where(nanArr), 0)
    y_train = np.delete(y_train, np.where(nanArr), 0)

    x_train = np.delete(x_train, np.where(infArr), 0)
    y_train = np.delete(y_train, np.where(infArr), 0)

    x_train = np.delete(x_train, np.where(negInfArr), 0)
    y_train = np.delete(y_train, np.where(negInfArr), 0)

    print(x_train.shape)

    delete_x  = -1
    rangeT = x_test.shape[0]
    i = 0;
    while i < rangeT:
        if "DoS Hulk" in x_test[i,:]:
            print("Trovata Stringa in x_test :"+str(i)+" -->"+ str(x_test[i]))
            delete_x = i
            x_test = np.delete(x_test, delete_x, 0)
            y_test = np.delete(y_test, delete_x, 0)
            rangeT = x_test.shape[0]
            continue
        if np.isnan(np.sum(x_test[i, :])) or np.isinf(np.sum(x_test[i, :])):
            #print("Trovata Nan in x_test :" + str(i) + " -->" + str(x_test[i]))
            delete_x = i
            x_test = np.delete(x_test, delete_x, 0)
            y_test = np.delete(y_test, delete_x, 0)
            rangeT = x_test.shape[0]

        i+=1

    #x_test = x_test.astype(np.float)

    x_train = x_train.astype(np.float)
    rangeT = x_train.shape[0]
    j = 0;
    while j < rangeT:
        if "DoS Hulk" in x_train[j,:]:
            print("Trovata Stringa in x_train:"+str(j1)+" -->"+ str(x_train[j1]))
            delete_x=j
            x_train = np.delete(x_train, delete_x, 0)
            y_test = np.delete(y_train, delete_x, 0)
            rangeT = x_train.shape[0]
            continue
        if np.isnan(np.sum(x_train[j, :])) or np.isinf(np.sum(x_train[j, :])):
           # print("Trovata Nan in x_train :" + str(j) + " -->" + str(x_train[j]))
            delete_x = j
            x_train = np.delete(x_train, delete_x, 0)
            y_train = np.delete(y_train, delete_x, 0)
            rangeT = x_train.shape[0]

        j += 1

    x_train = x_train.astype(np.float)

    '''
    for k in range(y_test.shape[0]):
        if "DoS Hulk" in y_test[k,:]:
            print("Trovata Stringa in y_test:"+str(k)+" -->"+ str(y_test[k]))
        if np.isnan(np.sum(y_test[k, :])):
            print("Trovata Nan in y_test :" + str(k) + " -->" + str(y_test[k]))

    for z in range(y_train.shape[0]):
        if "DoS Hulk" in y_train[z,:]:
            print("Trovata Stringa in y_train:"+str(z)+" -->"+ str(y_train[z]))
        if np.isnan(np.sum(y_train[z, :])):
            print("Trovata Nan in y_train :" + str(z) + " -->" + str(y_train[z]))
    '''
    # Normalizzazione sui dati
    #scaler = MaxAbsScaler()
    #scaler = StandardScaler()
    #x_train_scaled = scaler.fit_transform(x_train) #fitto e calcolo i parametri, incapsulati nella variabile scaler
    #x_test_scaled = scaler.transform(x_test) #qui uso direttamente scaler


    #return x_train_scaled, x_test_scaled, y_train, y_test
    return x_train, x_test, y_train, y_test