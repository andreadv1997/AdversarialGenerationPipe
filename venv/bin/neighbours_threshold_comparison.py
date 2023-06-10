from __future__ import unicode_literals
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from dataPrep import get_data_from_file, preprocess
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

base_dir = str(Path().resolve().parent)


featuresLabel = ['Protocol', 'Flow_Duration', 'Total_Fwd_Packets', 'Total_Backward_Packets', 'Total_Length_of_Fwd_Packets',
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
            'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min','Label']

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

def find_n_neighbours(df,n):
    order = np.argsort(df.values, axis=1)[:, :n] #controllare comportamento di questa istruzion. Questa istruzione non e' necessaria
    #in questo caso ascending e' True perche' la matrice definisce la distanza e non la similarita'. I vicini meno distanti sono i piu' simili
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=True).iloc[:n].index,index=['top{}'.format(i) for i in range(1, n + 1)]), axis=1)
    #df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False).iloc[:n].index,index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    # il dataset restituito e' quello contenente gli indici dei valori top n
    return df

def User_item_score(user,item, sim_user_n, similarity_matrix, oracle, n):
    a = sim_user_n[sim_user_n.index==user].values #definisce gli indici degli n oggetti piu' simili
    b = a.squeeze().tolist() #lista degli indici
    b=b[0:n] # in questo modo calcolo la matrice di similarita' e i vicini piu' prossimi una sola volta e uso il valore di n per calcolare l'indice
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


if __name__ ==  "__main__":


    fileVal = base_dir + '/bin/ADV-baseline-PARTE2.csv'
    datasetVal = get_data_from_file(fileVal)
    X_FIL, Y_FIL, _ = preprocess(datasetVal, False)

    adversarials = pd.read_csv(base_dir+"/adversarialsFGSM2.csv", names=featuresLabel, sep=',', dtype='unicode', header=0)
    adversarials_Label = adversarials['Label'].values.astype(np.float)

    adversarials = adversarials[features]
    adversarials = adversarials.values.astype(np.float)

    filteringDataframe = pd.DataFrame(X_FIL)
    oracle = pd.DataFrame(np.concatenate((X_FIL, Y_FIL), axis=1))
    print("Generazione dataFrame eseguito")

    #per fare un test utilizzeremo la distanza euclidea
    cosine = euclidean_distances(adversarials, filteringDataframe)
    #cosine = cosine_similarity(adversarials, filteringDataframe)
    print("Calcolo della matrice di similarita' eseguito")
    print(cosine.shape)
    print(str(adversarials.shape))
    similarity_matrix = pd.DataFrame(cosine, index=range(adversarials.shape[0]))
    similarity_matrix.columns = filteringDataframe.index

    sim_matrix_30 = find_n_neighbours(similarity_matrix, 30)

    #print(str(adv_x.shape[1]))




    dict_General_values = {}
    dict_General_values['total'] = adversarials.shape[0]

    threshold = np.arange(0.1, 0.5, 0.1)
    neighbours = np.arange(10, 35, 5)


    combination = list(itertools.product(neighbours, threshold))
    #print(combination)

    for n_neigh, th in combination:
        label_rec = np.zeros(adversarials.shape[0])

        for index in range(adversarials.shape[0]):


            score = User_item_score(index, 77, sim_matrix_30, similarity_matrix, oracle, n_neigh)

            if (score < 0+th):
                 label_rec[index] = 0
            elif (score > 1-th):
                 label_rec[index] = 1.0
            else:
                label_rec[index] = score

        print("Total sum: ", label_rec.sum(axis=0))

        label_count = abs(label_rec - adversarials_Label)

        mod = 0
        grey = 0
        sig = 0
        originally_BENIGN_nr = 0
        originally_ATTACK_nr = 0
        for ind in range(label_count.shape[0]):
             if (label_count[ind] != 0):
                 if (label_count[ind] == 1):
                     mod += 1

                     if(adversarials_Label[ind]== 1):
                         originally_BENIGN_nr+=1
                     else:
                        originally_ATTACK_nr+=1
                 else:
                    grey += 1

             else:
                 sig+=1

        dict_General_values[(n_neigh,th)] = (float(mod)/float(adversarials.shape[0])*100,float(grey)/float(adversarials.shape[0])*100,float(sig)/float(adversarials.shape[0])*100)

    print(dict_General_values)
    rows = 1
    columns = 3

    ax = []
    title = ["Mod","Grey","Sig"]
    fig = plt.figure(figsize=(10, 15))
    for i in range(columns * rows):

        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title(str(title[i]))  # set title
        #ax[-1].set_title("Neighbours:" + str(neighbours[i]))  # set title



    y_mod = {}
    y_grey = {}
    y_sig = {}
    for n in neighbours:
        y_mod_n=[]
        y_grey_n=[]
        y_sig_n=[]
        for t in threshold:
            y_mod_n.append(dict_General_values[(n,t)][0])
            y_grey_n.append(dict_General_values[(n, t)][1])
            y_sig_n.append(dict_General_values[(n, t)][2])
        y_mod[n]=y_mod_n
        y_grey[n] = y_grey_n
        y_sig[n] = y_sig_n

    for n in neighbours:
        ax[0].plot(threshold, y_mod[n], label=str(n)+" Neighbours")
        ax[0].set_xlabel("Th")
        ax[0].legend()
        ax[0].grid(True)

        ax[1].plot(threshold, y_grey[n], label=str(n)+" Neighbours")
        ax[1].set_xlabel("Th")
        ax[1].legend()
        ax[1].grid(True)

        ax[2].plot(threshold, y_sig[n], label=str(n)+" Neighbours")
        ax[2].set_xlabel("Th")
        ax[2].legend()
        ax[2].grid(True)



    plt.show()

















