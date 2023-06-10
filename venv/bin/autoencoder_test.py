import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Input, Dense
from keras import regularizers, initializers
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataPrep import preprocess, get_data_from_file, preprocess_autoencoder


np.random.seed(101)
class MultilayerAutoEncoder():

    def __init__(self, input_dim):   #metodo costruttore . Definisce il costruttore per la classe che stiamo definendo
        input_layer = Input(shape=(input_dim,)) #creo layer di input con una certa dimensione
        layer_1 = Dense(100, activation='tanh',
                        activity_regularizer=regularizers.l1(10e-5),
                        kernel_initializer= initializers.RandomNormal()) (input_layer) #creao layer neurale con input e primo livello nascosto densamente connesso. Passo il numero di neuoroni del lievello e funzione di attivazione(tangente iperbolica). Metto regolarizzatore per evitare overfitting
        layer_2 = Dense(90, activation='tanh',
                        activity_regularizer=regularizers.l1(10e-5),
                        kernel_initializer= initializers.RandomNormal()) (layer_1)

        layer_3 = Dense(10, activation='tanh',
                        activity_regularizer=regularizers.l1(10e-5),
                        kernel_initializer= initializers.RandomNormal()) (layer_2)

        layer_4 = Dense(90, activation='tanh',
                        activity_regularizer=regularizers.l1(10e-5),
                        kernel_initializer=initializers.RandomNormal())(layer_3)
        layer_5 = Dense(100, activation='tanh',
                        activity_regularizer=regularizers.l1(10e-5),
                        kernel_initializer=initializers.RandomNormal())(layer_4)
        output_layer = Dense(input_dim, activation='relu') (layer_3) #numero di uscite del livello e' il primo parametro. Pari al numero di input

        #costruisco rete
        self.autoencoder = Model(inputs = input_layer, output = output_layer)

    def summary(self,):
        self.autoencoder.summary()


    def train(self, x, y):

        epoch = 300 #numero di epoche di apprendimento

        batch_size = 50 #divido training set in batch
        validation_split = 0.03 #pezzetto dei record benigni per l'addestramento

        self.autoencoder.compile(optimizer='adagrad', loss='mean_squared_error')  #ottimizzatore. Loss definisce il tipo di perdita
            #compile fa un compile della rete

        history = self.autoencoder.fit(x,y ,epochs= epoch, batch_size=batch_size,shuffle=True,validation_split=validation_split, verbose=2)  #shuffle mischia i dati e verbose definisce il livello di verbosita' con cui ci fa vedere come si sta comportando il training

        # --- determine treshold ---
        #definisco soglia con validation set
        x_val = x[x.shape[0]-(int)(x.shape[0]*validation_split):x.shape[0]-1, :]
        #02:23
        val_predictions = self.autoencoder.predict(x_val) # predizione sul validation set. Fa predizione sull'input

        val_mse = np.mean(np.power(x_val - val_predictions, 2), axis=1)

        threshold = np.mean ( val_mse ) + 3*np.std ( val_mse )
        # ---
        df_history = pd.DataFrame(history.history)

        return df_history, threshold


    def evaluate(self,x_test, y_test, threshold):
        predictions = self.autoencoder.predict(x_test)
        mse = np.mean(np.power(x_test-predictions, 2), axis =1) #errore di ricostruzione
        df_error = pd.DataFrame({'reconstruction_error': mse, 'true_class': y_test})

        plot_reconstruction_error(df_error, threshold)
        pred = compute(df_error, threshold)
        compute_metrics(y_test, pred)

def compute_metrics(Y_test, predictions):
    difference_class = predictions - Y_test
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
            if (Y_test[index] == 0):
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

    print('Accuracy of decision tree: ' + str(dict['accuracy']))
    print('Precision of decision tree: ' + str(dict['precision']))
    print('Recall of decision tree: ' + str(dict['recall']))
    print('F1 of decision tree: ' + str(dict['f1']))
    print('True pos: ', str(true_pos))
    print('False pos: ', str(false_pos))
    print('True neg: ', str(true_neg))
    print('False neg: ', str(false_neg))

    return dict



from pathlib import Path
base_dir = str(Path().resolve().parent)
def plot_reconstruction_error(errors, threshold):
    groups = errors.groupby('true_class')
    fig, ax = plt.subplots(figsize=(15, 7))
    right = 0
    for name, group in groups:
        if max(group.index) > right: right = max(group.index)

        ax.plot(group.index, group.reconstruction_error, marker = 'o',
                ms = 5, linestyle = '', markeredgecolor = 'black', #alpha = 0.5,
                 label = 'Attack' if int(name) == 0 else 'Normal', color = 'green' if int(name) == 1 else 'red')
    ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors = 'red', zorder = 100, label = 'Threshold',linewidth=4,linestyles='dashed')
    #ax.semilogy()
    ax.legend()
    plt.xlim(left = 0, right = right)
    plt.title('Reconstruction error for different classes')
    plt.grid(True)
    plt.ylabel('Reconstruction error')
    plt.xlabel('Data point index')
    plt.savefig(base_dir + '/reconstruction_error.png', bbox_inches = 'tight', dpi = 500)
    plt.show()


def compute(df_error, threshold):
    y_pred = [0 if e > threshold else 1 for e in df_error.reconstruction_error.values]
    conf_matrix = confusion_matrix(df_error.true_class, y_pred, labels=[1,0])

    tn, fp, fn, tp = conf_matrix.ravel()

    print(tn)
    print(fp)
    print(fn)
    print(tp)
    recall = float(tp) / float((tp+fn))
    precision = float(tp) / float((tp+fp))
    f1 = 2 *  float(precision*recall) / float((precision+recall) )

    print('R  = ', recall)
    print('P  = ', precision)
    print('F1 = ', f1)

    sns.heatmap(conf_matrix, xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'], annot=True,fmt='d');
    plt.title('Confusion matrix')
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig(base_dir + '/confusion_matrix.png', bbox_inches='tight', dpi=500)
    plt.show()
    return y_pred






if __name__ == "__main__":
    base_dir = str(Path().resolve())
    file_train = base_dir + '/MOD-training.csv'
    dataset_train = get_data_from_file(file_train)

    file_val = base_dir + '/MOD-validation.csv'
    dataset_val = get_data_from_file(file_val)

    file_test = base_dir + '/MOD-test.csv'
    dataset_test = get_data_from_file(file_test)

    X_train, Y_train= preprocess_autoencoder(dataset_train)
    X_val, Y_val, _ = preprocess(dataset_val)
    X_test, Y_test, _ = preprocess(dataset_test)

    input_dim = X_train.shape[1]  # restituisce le dimensioni della matrice. Qui restituiamo il numero di colonne

    # creaiamo istanza di MultiLayerAutoencoder

    autoencoder = MultilayerAutoEncoder(input_dim=input_dim)
    autoencoder.summary()

    history, threshold = autoencoder.train(X_train,X_train)  # devo ricostruire input e output che sono uguali. Mostro all'autoencoder in ingresso e in uscita la stessa cosa, per potre ricostruire bene tutti i vettori

    autoencoder.evaluate(X_test, Y_test[:,0], threshold)







