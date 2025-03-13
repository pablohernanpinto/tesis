import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay, balanced_accuracy_score
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer
import tensorflow as tf
from tensorflow import keras
from keras import regularizers
from keras.optimizers import Adam
#from keras.backend import expand_dims
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import Sequential
from keras.constraints import MaxNorm
from keras.layers import Activation, Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization, SpatialDropout1D
import os
from imblearn.over_sampling import SMOTE


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def entrenamiento_base(y_train):
    scaler=Normalizer(norm='max')
    sc_X = scaler
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    sample_size = X_train.shape[0] # numero de muestras en el set de datos
    time_steps  = X_train.shape[1] # numero de atributos en el set de datos
    input_dimension = 1            #

    X_train_reshaped = X_train.reshape(sample_size,time_steps,input_dimension)
    X_test_reshaped = X_test.reshape(X_test.shape[0],X_test.shape[1],1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.000001)
    early_st = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    n_timesteps = X_train_reshaped.shape[1] #
    n_features  = X_train_reshaped.shape[2] #

    model = Sequential(name="Modelo_s_aureus_ciprofloxacin")
    init_mode = 'normal'
    model.add(Conv1D(filters=(64), kernel_size=(17), input_shape = (n_timesteps,n_features), name='Conv_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D_1"))

    model.add(Conv1D(filters=(128), kernel_size=(9),kernel_initializer=init_mode, kernel_regularizer=regularizers.l2(0.0001),  name='Conv_2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D_2"))

    model.add(Conv1D(filters=(256), kernel_size=(5),kernel_initializer=init_mode,kernel_regularizer=regularizers.l2(0.0001),   name='Conv_3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D_3"))

    model.add(Conv1D(filters=(256), kernel_size=(5),kernel_initializer=init_mode, kernel_regularizer=regularizers.l2(0.0001),   name='Conv_4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2, name="MaxPooling1D_4"))

    model.add(Flatten())
    model.add(Dropout(0.65))
    model.add(Dense(256, activation='relu',kernel_initializer=init_mode, kernel_regularizer=regularizers.l2(0.0001), name="fully_connected_0"))
    model.add(Dense(64, activation='relu',kernel_initializer=init_mode, kernel_regularizer=regularizers.l2(0.0001), name="fully_connected_1"))
    model.add(Dense(64, activation='relu',kernel_initializer=init_mode, kernel_regularizer=regularizers.l2(0.0001),  name="fully_connected_2"))
    model.add(Dense(n_features, activation='sigmoid', name="OUT_Layer"))

    model.compile(optimizer = Adam(learning_rate=0.0001), loss = 'binary_crossentropy',  metrics=METRICS)
    model.summary()
    history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=10, verbose=1, validation_split=0.1, callbacks=[reduce_lr,early_st])
    y_pred  = model.predict(X_test_reshaped)
    y_pred = (y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
    return model,'Entrenamiento base',cm,y_pred,X_train_reshaped

def Aplicar_Smote(X_train, X_test, y_train, y_test):
    smote = SMOTE(random_state=42)
    X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)
    print(pd.Series(y_train).value_counts())

def columnas_bacterias_fun(df):
    vocales = ['a','e','i','o','u']
    columnas_bacterias = []
    for i in vocales:
        for j in df.columns:
            if i in j:
                columnas_bacterias.append(j)
    columnas_bacterias = list(set(columnas_bacterias))
    return columnas_bacterias

def inscripcion_resultados(model,archivo,bacteria,cm,y_test, y_pred,tipo_entrenamiento,X_test_reshaped,X_train_reshaped):
    with open('resultados/resultados.txt', 'a') as archivo_:
        # Redirige la salida estándar al archivo
        print('-----------------------------------------------------\n\n','nombre de archivo:', archivo, '\nBacteria:', bacteria, '\n\nconfusion_matrix:\n', cm,"\n\nbalanced acuracy:", balanced_accuracy_score(y_test, y_pred),"\n\nbalanced acuracy:",tipo_entrenamiento, file=archivo_)
        target_names=["0","1"]
        print('\n\n',classification_report(y_test, y_pred, target_names=target_names), file=archivo_)

        train_predictions_baseline = model.predict(X_train_reshaped, batch_size=10)
        test_predictions_baseline = model.predict(X_test_reshaped, batch_size=10)
        print('\n\n')
        baseline_results = model.evaluate(X_test_reshaped, y_test, verbose=0)
        for name, value in zip(model.metrics_names, baseline_results):
            print(name, ': ', value, file=archivo_) 



files_list = os.listdir('SetDatos/')
for archivo in files_list:
    print(archivo)
    df = pd.read_csv('SetDatos/'+archivo)
    df = df.drop(columns=['code','species'])
    df.dropna(axis=0, how="any", inplace=True)
    columnas_bacterias = columnas_bacterias_fun(df)
    for bacteria in columnas_bacterias:
        try:
            print('Archivo:',archivo,'Bacteria:',bacteria)
            columnas_bacterias_sin_bacteria = [b for b in columnas_bacterias if b != bacteria]
            df_bacteria = df.drop(columns = columnas_bacterias_sin_bacteria)
            bacteria = df_bacteria.columns[-1]
            X = df_bacteria.iloc[:, 0:-2].values  # variables independientes (espectros de masa)
            y = df_bacteria.iloc[:, -1].values    # variable dependientes (resistencia a ciprofloxacin)
            X = np.asarray(X).astype(np.float32)
            y = np.asarray(y).astype(np.float32)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify=y)
            

            model,tipo_entrenamiento,y_pred,cm,X_test_reshaped,X_train_reshaped = entrenamiento_base(y_train)
            inscripcion_resultados(model,archivo,bacteria,cm,y_test, y_pred,tipo_entrenamiento,X_test_reshaped,X_train_reshaped)

            # Abre el archivo en modo apéndice
            
        except ValueError as e:
            with open('resultados/resultados.txt', 'a') as archivo_:
                print("Error:",e,file = archivo_)
