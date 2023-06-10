import numpy as np
import pandas as pd
from cleverhans.utils_tf import model_loss
from pathlib import Path
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import keras as k
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input, Dense, Activation
from keras import regularizers, initializers
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataPrep import preprocess, get_data_from_file, preprocess_autoencoder, get_data_from_file_ADV
from joblib import dump, load
import os
import sys
np.random.seed(101)

class MLPException(Exception):
    pass


class MultilayerPerceptron():

    def __init__(self, input_dim):   #metodo costruttore . Definisce il costruttore per la classe che stiamo definendo


         if (os.path.exists("mlp_filtered_NotScaled.joblib")):
            self.mlp = k.models.load_model("mlp_filtered_NotScaled.joblib")


         else:
            input_layer = Input(shape=(input_dim,)) #creo layer di input con una certa dimensione
            layer_1 = Dense(100, activation='relu') (input_layer) #creao layer neurale con input e primo livello nascosto densamente connesso. Passo il numero di neuoroni del lievello e funzione di attivazione(tangente iperbolica). Metto regolarizzatore per evitare overfitting
            layer_2 = Dense(100, activation='tanh') (layer_1)

            layer_3 = Dense(100, activation='relu') (layer_2)
            layer_4 = Dense(100, activation='sigmoid')(layer_3)
            layer_5 = Dense(100, activation='relu')(layer_4)
            layer_6 = Dense(2, activation='linear')(layer_5)
            output_layer = Activation(activation='softmax') (layer_6) #numero di uscite del livello e' il primo parametro. Pari al numero di input

            #costruisco rete
            self.mlp = Model(inputs = input_layer, output = output_layer)

            base_dir = str(Path().resolve())
            file_train = base_dir + '/MOD-training-correct-samples.csv'
            dataset_train,_ = get_data_from_file(file_train,True)

            file_val = base_dir + '/MOD-validation.csv'
            dataset_val,_ = get_data_from_file(file_val)

            file_test = base_dir + '/MOD-test.csv'
            dataset_test,_ = get_data_from_file(file_test)

            X_train, Y_train, scaler = preprocess(dataset_train, True)
            X_val, Y_val, _ = preprocess(dataset_val, True,scaler=scaler)
            X_test, Y_test, _ = preprocess(dataset_test, True,scaler=scaler)

            history = self.train(X_train, Y_train)

            #self.mlp.save("mlp.joblib")
            pred = self.predict(X_val)
            dict = self.evaluate(pred, Y_val)
            if (dict['accuracy'] >= 0.91 and dict['precision'] >= 0.91 and dict['recall'] >= 0.91 and dict['f1'] >= 0.91):

                    print("Values satisfies constraints")
                    print('Accuracy: ' + str(dict['accuracy']))
                    print('Precision: ' + str(dict['precision']))
                    print('Recall: ' + str(dict['recall']))
                    print('F1: ' + str(dict['f1']))

                    self.mlp.save("mlp_filtered_NotScaled.joblib")




            else:
                print("Values DO NOT satisfy constraints")
                print('Accuracy of decision tree: ' + str(dict['accuracy']))
                print('Precision of decision tree: ' + str(dict['precision']))
                print('Recall of decision tree: ' + str(dict['recall']))
                print('F1 of decision tree: ' + str(dict['f1']))






    def adv_train(self):
        if (os.path.exists("mlp_filtered_ADV_TRAIN.joblib")):
            self.mlp = k.models.load_model("mlp_filtered_ADV_TRAIN.joblib")
        else:
            base_dir = str(Path().resolve())
            adv_dir = base_dir + "/adversarial_samples_for_training"
            adv_files = [f for f in os.listdir(adv_dir) if os.path.isfile(os.path.join(adv_dir, f))]  # questi sono i file da usare per il training


            file_original = base_dir + '/MOD-training-correct-samples.csv'
            dataset_original = get_data_from_file(file_original,True)



            X_original, Y_original, _ = preprocess(dataset_original, True)

            X_train = np.empty(shape=(0, 77))
            Y_train = np.empty(shape=(0, 2))

            for file in adv_files:
                dataset_temp = get_data_from_file_ADV(base_dir+"/adversarial_samples_for_training/"+file)
                X_temp, Y_temp, _ = preprocess(dataset_temp,adv=True, twoCol=True)

                X_train = np.concatenate((X_train, X_temp), axis=0)
                Y_train = np.concatenate((Y_train, Y_temp), axis=0)

            X_train = np.concatenate((X_original, X_train), axis=0)
            Y_train = np.concatenate((Y_original, Y_train), axis=0)

            base_dir = str(Path().resolve())
            file_val = base_dir + '/MOD-validation.csv'
            dataset_val = get_data_from_file(file_val)
            X_val, Y_val, _ = preprocess(dataset_val, True)

            history = self.train(X_train, Y_train)

            # self.mlp.save("mlp.joblib")
            pred = self.predict(X_val)
            dict = self.evaluate(pred, Y_val)
            if (dict['accuracy'] >= 0.91 and dict['precision'] >= 0.91 and dict['recall'] >= 0.91 and dict['f1'] >= 0.91):

                print("Values satisfies constraints")
                print('Accuracy: ' + str(dict['accuracy']))
                print('Precision: ' + str(dict['precision']))
                print('Recall: ' + str(dict['recall']))
                print('F1: ' + str(dict['f1']))
                self.mlp.save("mlp_filtered_ADV_TRAIN.joblib")

            else:
                print("Values DO NOT satisfy constraints")
                print('Accuracy of decision tree: ' + str(dict['accuracy']))
                print('Precision of decision tree: ' + str(dict['precision']))
                print('Recall of decision tree: ' + str(dict['recall']))
                print('F1 of decision tree: ' + str(dict['f1']))

    def summary(self,):
        self.mlp.summary()


    def train(self, x, y):

        epoch = 200 #numero di epoche di apprendimento

        batch_size = 64 #divido training set in batch
        validation_split = 0.01 #pezzetto dei record benigni per l'addestramento

        #adadelta

        optimizer = k.optimizers.Adadelta(lr = 0.1)
        #self.mlp.compile(optimizer='adadelta', loss='mean_squared_error')  #ottimizzatore. Loss definisce il tipo di perdita
        self.mlp.compile(optimizer=optimizer,loss='mean_squared_error')  # ottimizzatore. Loss definisce il tipo di perdita


        #compile fa un compile della rete
        #self.mlp.compile(optimizer='adadelta',loss='categorical_crossentropy')  # ottimizzatore. Loss definisce il tipo di perdita

        history = self.mlp.fit(x,y ,epochs= epoch, batch_size=batch_size,shuffle=True,validation_split=validation_split, verbose=2)  #shuffle mischia i dati e verbose definisce il livello di verbosita' con cui ci fa vedere come si sta comportando il training

        # ---
        df_history = pd.DataFrame(history.history)

        return df_history

    def predict(self,x_test):
        predictions = self.mlp.predict(x_test)
        return predictions


    def evaluate(self,pred, Y_test):
        vectPred = np.argmax(pred,axis=1)
        vectOr = np.argmax(Y_test,axis=1)

        difference_class = vectPred - vectOr
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

        for index in range(len(pred)):
            if (difference_class[index] == 0):
                # TP or TN
                if (Y_test[index,0] == 0):
                    true_pos += 1
                    true_pos_index.append(index)
                else:
                    true_neg += 1
            else:
                if (difference_class[index] < 0):
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










if __name__ == "__main__":

    base_dir = str(Path().resolve())
    file_test = base_dir + '/MOD-test.csv'
    dataset_test = get_data_from_file(file_test)

    fileName = sys.argv[1]
    file_evaluation = base_dir + '/' + fileName
    dataset_evaluation = get_data_from_file_ADV(file_evaluation)
    dataset_evaluation = dataset_evaluation.iloc[1:]
    X_eval, Y_eval, _ = preprocess(dataset_evaluation,True)


    X_test, Y_test, _ = preprocess(dataset_test,True)


    input_dim = X_test.shape[1]  # restituisce le dimensioni della matrice. Qui restituiamo il numero di colonne

    # creaiamo istanza di MultiLayerAutoencoder

    from mlp_test import MultilayerPerceptron as oldMLP

    mlp = oldMLP(input_dim=input_dim)
    #mlp = MultilayerPerceptron(input_dim=input_dim)
    mlp.summary()

    #mlp.adv_train() #QUI SI ESEGUE  L'ADDESTRAMENTO SUGLI ESEMPI AVVERSARI
    #history= mlp.train(X_train,Y_train)  # devo ricostruire input e output che sono uguali. Mostro all'autoencoder in ingresso e in uscita la stessa cosa, per potre ricostruire bene tutti i vettori

    prediction_test = mlp.predict(X_test)
    predictions_VAL = mlp.predict(X_eval)

    dict_test = mlp.evaluate(prediction_test, Y_test)
    dict_val = mlp.evaluate(predictions_VAL, Y_eval)


    print("Test")
    print(dict_test['accuracy'])
    print(dict_test['precision'])
    print(dict_test['recall'])
    print("*******")
    print("Eval")
    print(dict_val['accuracy'])
    print(dict_val['precision'])
    print(dict_val['recall'])
