import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Deshabilitar GPU

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
from keras.models import Sequential,Model
from keras.constraints import MaxNorm
from keras.layers import Activation, Dense, Conv1D, Flatten, MaxPooling1D, Dropout, BatchNormalization, SpatialDropout1D,Lambda,Input
from imblearn.over_sampling import SMOTE
import gc
from tensorflow.keras.losses import mse
import torch
import torch.nn as nn
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

# Deshabilitar GPU en TensorFlow
tf.config.set_visible_devices([], 'GPU')



torch.manual_seed(42)
np.random.seed(42)

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

def Crear_modelo(X_train_reshaped,y_train):
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
    return model

def normalizacion(X_train, X_test):
    scaler=Normalizer(norm='max')
    sc_X = scaler
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    sample_size = X_train.shape[0] # numero de muestras en el set de datos
    time_steps  = X_train.shape[1] # numero de atributos en el set de datos
    input_dimension = 1            #

    X_train_reshaped = X_train.reshape(sample_size,time_steps,input_dimension)
    X_test_reshaped = X_test.reshape(X_test.shape[0],X_test.shape[1],1)
    return X_train_reshaped,X_test_reshaped


def entrenamiento_base(X_train, X_test, y_train, y_test):
    X_train_reshaped,X_test_reshaped = normalizacion(X_train, X_test)
    
    model = Crear_modelo(X_train_reshaped,y_train)

    y_pred  = model.predict(X_test_reshaped)
    y_pred = (y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
    #model,tipo_entrenamiento,cm,y_pred,X_test_reshaped,X_train_reshaped
    return model,'Entrenamiento base',cm,y_pred,X_test_reshaped,X_train_reshaped

def Aplicar_Smote(X_train, X_test, y_train, y_test):
    smote = SMOTE(random_state=42)
    X_resampled_smote, y_resampled_smote = smote.fit_resample(X_train, y_train)
    X_train_reshaped,X_test_reshaped = normalizacion(X_resampled_smote, X_test)

    model = Crear_modelo(X_train_reshaped,y_resampled_smote)
    y_pred  = model.predict(X_test_reshaped)

    y_pred = (y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
    #model,tipo_entrenamiento,cm,y_pred,X_test_reshaped,X_train_reshaped
    return model,'Entrenamiento Smote',cm,y_pred,X_test_reshaped,X_train_reshaped

def Aplicar_VAE(df_bacteria,bacteria,X_train, X_test, y_train, y_test):
    minority_class = df_bacteria[df_bacteria[bacteria] == 1].drop(columns=[bacteria])
    scaler = MinMaxScaler()
    X_minority_scaled = scaler.fit_transform(minority_class)
    # Dimensiones
    input_dim = X_minority_scaled.shape[1]
    latent_dim = 2  # Espacio latente

    # Encoder
    inputs = Input(shape=(input_dim,))
    hidden = Dense(16, activation='relu')(inputs)
    z_mean = Dense(latent_dim, name='z_mean')(hidden)
    z_log_var = Dense(latent_dim, name='z_log_var')(hidden)

    # Sampling
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # Decoder
    decoder_hidden = Dense(16, activation='relu')
    decoder_output = Dense(input_dim, activation='sigmoid')

    hidden_decoded = decoder_hidden(z)
    outputs = decoder_output(hidden_decoded)

    # Modelo VAE
    vae = Model(inputs, outputs)

    # Pérdida personalizada
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    vae.summary()
    vae.fit(X_minority_scaled, X_minority_scaled, epochs=200, batch_size=32, verbose=1)
    # Construir el generador (Decoder independiente)
    decoder_input = Input(shape=(latent_dim,))
    hidden_decoded_2 = decoder_hidden(decoder_input)
    output_decoded = decoder_output(hidden_decoded_2)
    generator = Model(decoder_input, output_decoded)

    # Generar datos sintéticos
    print(pd.Series(y_train).value_counts())
    num_samples = pd.Series(y_train).value_counts()[0]-pd.Series(y_train).value_counts()[1]
    latent_points = np.random.normal(size=(num_samples, latent_dim))
    synthetic_data = generator.predict(latent_points)


    # Escalar de vuelta a los valores originales
    synthetic_data_original = scaler.inverse_transform(synthetic_data)
    X_train_balanced = np.concatenate([X_train, synthetic_data_original])
    y_train_balanced = np.concatenate([y_train, np.ones(num_samples)])

    X_train_reshaped,X_test_reshaped = normalizacion(X_train_balanced, X_test)
    
    model = Crear_modelo(X_train_reshaped,y_train_balanced)

    y_pred  = model.predict(X_test_reshaped)
    y_pred = (y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
    #model,tipo_entrenamiento,cm,y_pred,X_test_reshaped,X_train_reshaped
    return model,'Entrenamiento VAE',cm,y_pred,X_test_reshaped,X_train_reshaped


def Aplicar_DifussionModel(df_bacteria,bacteria,X_train, X_test, y_train, y_test):
    # Preprocesamiento
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X_train)
    # Modelo de Difusión
    class DiffusionModel(nn.Module):
        def __init__(self, input_dim):
            super(DiffusionModel, self).__init__()
            self.model = nn.Sequential( 
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(p=0.2),  # Regularización Dropout
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(p=0.2),  # Regularización Dropout
                nn.Linear(32, input_dim)
            )
        def forward(self, x):
            return self.model(x)
    # Función de ruido (Scheduler)
    def add_noise(data, timesteps, noise_scale=1.0):
        noise = np.random.normal(0, noise_scale, data.shape) * np.sqrt(timesteps / 100)
        noisy_data = data + noise
        return noisy_data, noise
    
    # Configuración del modelo
    input_dim = scaled_data.shape[1]
    model = DiffusionModel(input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.SmoothL1Loss()  # O Huber Loss

    # Scheduler de tasa de aprendizaje
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # Entrenamiento
    scaled_data_tensor = torch.tensor(scaled_data, dtype=torch.float32)
    epochs = 1000
    losses = []  # Para guardar la pérdida por época

    for epoch in range(epochs):
        timesteps = np.random.randint(1, 100)
        noisy_data, noise = add_noise(scaled_data, timesteps)
        noisy_data_tensor = torch.tensor(noisy_data, dtype=torch.float32)
        noise_tensor = torch.tensor(noise, dtype=torch.float32)

        optimizer.zero_grad()
        predicted_noise = model(noisy_data_tensor)
        loss = loss_fn(predicted_noise, noise_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Actualiza la tasa de aprendizaje

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item()}")

        # Generación de Datos Sintéticos
    def generate_synthetic_data(model, num_samples, input_dim):
        model.eval()
        with torch.no_grad():
            synthetic_data = np.random.normal(0, 1, (num_samples, input_dim))
            for t in range(100, 0, -1):  # Reverse diffusion
                synthetic_data = synthetic_data - model(torch.tensor(synthetic_data, dtype=torch.float32)).numpy() * (t / 100)
            return synthetic_data
        
        
    synthetic_data = generate_synthetic_data(model, pd.Series(y_train).value_counts()[0]-pd.Series(y_train).value_counts()[1], input_dim)
    synthetic_data_rescaled = scaler.inverse_transform(synthetic_data)

        
    # Cambiar el tipo de datos a float32
    synthetic_samples_numpy = synthetic_data_rescaled.astype(np.float32)

    # Mostrar las muestras generadas
    synthetic_samples_numpy.shape

    X_train_resampled = np.concatenate([X_train,synthetic_samples_numpy])

    ones_array = np.ones(pd.Series(y_train).value_counts()[0]-pd.Series(y_train).value_counts()[1])
    y_train_resampled = np.concatenate([y_train,ones_array])

    #termino de oversampling
    X_train_reshaped,X_test_reshaped = normalizacion(X_train_resampled, X_test)
    
    model = Crear_modelo(X_train_reshaped,y_train_resampled)

    y_pred  = model.predict(X_test_reshaped)
    y_pred = (y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
    #model,tipo_entrenamiento,cm,y_pred,X_test_reshaped,X_train_reshaped
    return model,'Entrenamiento difussion model',cm,y_pred,X_test_reshaped,X_train_reshaped

def Aplicar_Copulas(df_bacteria,bacteria,X_train, X_test, y_train, y_test):
    minority_class = df_bacteria[df_bacteria[bacteria] == 1].drop(columns=[bacteria])

    # Crear un objeto Metadata para el dataset
    metadata = SingleTableMetadata()

    # Detectar automáticamente los tipos de datos del DataFrame
    metadata.detect_from_dataframe(minority_class)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data=minority_class)

    synthetic_data = synthesizer.sample(pd.Series(y_train).value_counts()[0]-pd.Series(y_train).value_counts()[1])
    
    # Cambiar el tipo de datos a float32
    synthetic_samples_numpy_copula = synthetic_data.astype(np.float32)

    # Mostrar las muestras generadas
    synthetic_samples_numpy_copula.shape
    X_train_resampled = np.concatenate([X_train,synthetic_samples_numpy_copula])

    ones_array = np.ones(int((pd.Series(y_train).value_counts()[0]-pd.Series(y_train).value_counts()[1])))
    y_train_resampled = np.concatenate([y_train,ones_array])


    X_train_reshaped,X_test_reshaped = normalizacion(X_train_resampled, X_test)
    
    model = Crear_modelo(X_train_reshaped,y_train_resampled)

    y_pred  = model.predict(X_test_reshaped)
    y_pred = (y_pred>0.5)
    cm = confusion_matrix(y_test, y_pred)
    #model,tipo_entrenamiento,cm,y_pred,X_test_reshaped,X_train_reshaped
    return model,'Entrenamiento Copulas',cm,y_pred,X_test_reshaped,X_train_reshaped


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
        print('-----------------------------------------------------\n\n','nombre de archivo:', archivo, '\nBacteria:', bacteria, "\n\nTipo de entrenamiento:",tipo_entrenamiento,'\n\nconfusion_matrix:\n', cm, file=archivo_)
        target_names=["0","1"]
        print('\n\n',classification_report(y_test, y_pred, target_names=target_names), file=archivo_)

        train_predictions_baseline = model.predict(X_train_reshaped, batch_size=10)
        test_predictions_baseline = model.predict(X_test_reshaped, batch_size=10)
        print('\n\n')
        baseline_results = model.evaluate(X_test_reshaped, y_test, verbose=0)
        for name, value in zip(model.metrics_names, baseline_results):
            print(name, ': ', value, file=archivo_)  


files_list = os.listdir('SetDatos/')
for archivo in ['e_coli_driams_b_2000_20000Da_v2 (1).csv']:#files_list
    print(archivo)
    df = pd.read_csv('SetDatos/'+archivo)
    df = df.drop(columns=['code','species'])
    df.dropna(axis=0, how="any", inplace=True)
    columnas_bacterias = columnas_bacterias_fun(df)
    
    for bacteria in ['Cefepime']: #columnas_bacterias

        try:
            print('Archivo:',archivo,'Bacteria:',bacteria)
            columnas_bacterias_sin_bacteria = [b for b in columnas_bacterias if b != bacteria]
            df_bacteria = df.drop(columns = columnas_bacterias_sin_bacteria)
            bacteria = df_bacteria.columns[-1]
            X = df_bacteria.iloc[:, 0:-1].values  # variables independientes (espectros de masa)
            y = df_bacteria.iloc[:, -1].values    # variable dependientes (resistencia a ciprofloxacin)
            X = np.asarray(X).astype(np.float32)
            y = np.asarray(y).astype(np.float32)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify=y)
 
            #resultado sin oversampling
            model,tipo_entrenamiento,cm,y_pred,X_test_reshaped,X_train_reshaped = entrenamiento_base(X_train, X_test, y_train, y_test)
            inscripcion_resultados(model,archivo,bacteria,cm,y_test, y_pred,tipo_entrenamiento,X_test_reshaped,X_train_reshaped)
        
            # Liberar memoria
            del model, tipo_entrenamiento, y_pred, cm, X_test_reshaped, X_train_reshaped
            gc.collect()  # Forzar recolección de basura
            
            #resultado con smote
            model,tipo_entrenamiento,cm,y_pred,X_test_reshaped,X_train_reshaped = Aplicar_Smote(X_train, X_test, y_train, y_test)
            inscripcion_resultados(model,archivo,bacteria,cm,y_test, y_pred,tipo_entrenamiento,X_test_reshaped,X_train_reshaped)

            # Liberar memoria
            del model, tipo_entrenamiento, y_pred, cm, X_test_reshaped, X_train_reshaped
            gc.collect()  # Forzar recolección de basura
            # 
            #resultado con VAE
            model,tipo_entrenamiento,cm,y_pred,X_test_reshaped,X_train_reshaped = Aplicar_VAE(df_bacteria,bacteria,X_train, X_test, y_train, y_test)
            inscripcion_resultados(model,archivo,bacteria,cm,y_test, y_pred,tipo_entrenamiento,X_test_reshaped,X_train_reshaped)
            
            # Liberar memoria
            del model, tipo_entrenamiento, y_pred, cm, X_test_reshaped, X_train_reshaped
            gc.collect()  # Forzar recolección de basura

            
            #resultado con Difussion model
            model,tipo_entrenamiento,cm,y_pred,X_test_reshaped,X_train_reshaped = Aplicar_DifussionModel(df_bacteria,bacteria,X_train, X_test, y_train, y_test)
            inscripcion_resultados(model,archivo,bacteria,cm,y_test, y_pred,tipo_entrenamiento,X_test_reshaped,X_train_reshaped)
            
            # Liberar memoria
            del model, tipo_entrenamiento, y_pred, cm, X_test_reshaped, X_train_reshaped
            gc.collect()  # Forzar recolección de basura

            #resultado con Difussion model
            model,tipo_entrenamiento,cm,y_pred,X_test_reshaped,X_train_reshaped = Aplicar_Copulas(df_bacteria,bacteria,X_train, X_test, y_train, y_test)
            inscripcion_resultados(model,archivo,bacteria,cm,y_test, y_pred,tipo_entrenamiento,X_test_reshaped,X_train_reshaped)
            
            # Liberar memoria
            del model, tipo_entrenamiento, y_pred, cm, X_test_reshaped, X_train_reshaped
            gc.collect()  # Forzar recolección de basura

        except ValueError as e:
            print(e)
            with open('resultados/resultados.txt', 'a') as archivo_:
                print("Error:",e,file = archivo_)

