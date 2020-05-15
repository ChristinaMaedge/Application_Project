# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:39:21 2020

@author: Marco Landt-Hayen, Christina Mädge
"""
# Vorbereitung / Setup
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, Adagrad, Nadam, RMSprop
from tensorflow.keras.utils import normalize
from sklearn import datasets

# Lade Trainings- und Testdaten aus R Export
path_to_task = "/Users/Marco Landt-Hayen/Documents/R/Application_Project/data"

train_WG1 = np.loadtxt(join(path_to_task,'df_MLP_train_WG1.csv'), delimiter=',') 
train_WG2 = np.loadtxt(join(path_to_task,'df_MLP_train_WG2.csv'), delimiter=',') 
train_WG3 = np.loadtxt(join(path_to_task,'df_MLP_train_WG3.csv'), delimiter=',') 
train_WG4 = np.loadtxt(join(path_to_task,'df_MLP_train_WG4.csv'), delimiter=',') 
train_WG5 = np.loadtxt(join(path_to_task,'df_MLP_train_WG5.csv'), delimiter=',') 

test_WG1 = np.loadtxt(join(path_to_task,'df_MLP_test_WG1.csv'), delimiter=',') 
test_WG2 = np.loadtxt(join(path_to_task,'df_MLP_test_WG2.csv'), delimiter=',') 
test_WG3 = np.loadtxt(join(path_to_task,'df_MLP_test_WG3.csv'), delimiter=',') 
test_WG4 = np.loadtxt(join(path_to_task,'df_MLP_test_WG4.csv'), delimiter=',') 
test_WG5 = np.loadtxt(join(path_to_task,'df_MLP_test_WG5.csv'), delimiter=',') 

# prüfe exemplarisch Typ und Struktur
type(train_WG1)
np.shape(train_WG1)

# Die Daten enthalten keine Spaltenüberschriften. Die Spalten sind:
# 0: Datum
# 1: Jahr
# 2: Warengruppe
# 3: Umsatz
# 4: KielerWoche
# 5: SommerferienSH
# 6: Feiertag
# 7: Silvester_ext
# 8: Montag
# 9: Dienstag
#10: Mittwoch
#11: Donnerstag
#12: Freitag
#13: Samstag
#14: Sonntag
#15: Januar
#16: Februar
#17: März
#18: April
#19: Mai 
#20: Juni
#21: Juli
#22: August
#23: September
#24: Oktober
#25: November
#26: Dezember
#27: Temp_eis
#28: Temp_kalt
#29: Temp_warm
#30: Temp_heiss

# Wir wollen zwei Modelle betrachten: Ein einfaches (mod1), dass nur die Variablen SommerferienSH, Feiertag und die Wochentage enthält.
# Und dann bauen wir noch ein komplexes (mod2) mit allen Variablen.

###################
# Modell 1 (mod1) #
###################

# Input-Variablen für Trainings- und Testdaten
mod1_train_WG1_input = train_WG1[:,[5,6,8,9,10,11,12,13,14]]
mod1_train_WG2_input = train_WG2[:,[5,6,8,9,10,11,12,13,14]]
mod1_train_WG3_input = train_WG3[:,[5,6,8,9,10,11,12,13,14]]
mod1_train_WG4_input = train_WG4[:,[5,6,8,9,10,11,12,13,14]]
mod1_train_WG5_input = train_WG5[:,[5,6,8,9,10,11,12,13,14]]

mod1_test_WG1_input = test_WG1[:,[5,6,8,9,10,11,12,13,14]]
mod1_test_WG2_input = test_WG2[:,[5,6,8,9,10,11,12,13,14]]
mod1_test_WG3_input = test_WG3[:,[5,6,8,9,10,11,12,13,14]]
mod1_test_WG4_input = test_WG4[:,[5,6,8,9,10,11,12,13,14]]
mod1_test_WG5_input = test_WG5[:,[5,6,8,9,10,11,12,13,14]]

# Target-Variablen für Trainings- und Testdaten ist der Umsatz
mod1_train_WG1_target = train_WG1[:,3]
mod1_train_WG2_target = train_WG2[:,3]
mod1_train_WG3_target = train_WG3[:,3]
mod1_train_WG4_target = train_WG4[:,3]
mod1_train_WG5_target = train_WG5[:,3]

mod1_test_WG1_target = test_WG1[:,3]
mod1_test_WG2_target = test_WG2[:,3]
mod1_test_WG3_target = test_WG3[:,3]
mod1_test_WG4_target = test_WG4[:,3]
mod1_test_WG5_target = test_WG5[:,3]

# Anzahl der Input-Variablen
input_dim = mod1_train_WG1_input.shape[1]

# Definiere die Struktur des Multilayer Perceptrons (MLP)
num_inputs = input_dim
num_hidden1 = 20 # Anzahl der units im ersten hidden layer
num_outputs = 1

# verwende sigmoide Aktivierung im hidden layer und lineare Aktivierung (>= 0) 'relu' für den output
mod1_WG1_input_layer = Input(shape=(num_inputs,), name='input')
mod1_WG1_hidden_1 = Dense(units=num_hidden1, activation="sigmoid", name='hidden1')(mod1_WG1_input_layer)
mod1_WG1_out = Dense(units=num_outputs, activation="relu", name="output")(mod1_WG1_hidden_1) 
  
mod1_WG2_input_layer = Input(shape=(num_inputs,), name='input')
mod1_WG2_hidden_1 = Dense(units=num_hidden1, activation="sigmoid", name='hidden1')(mod1_WG2_input_layer)
mod1_WG2_out = Dense(units=num_outputs, activation="relu", name="output")(mod1_WG2_hidden_1) 
  
mod1_WG3_input_layer = Input(shape=(num_inputs,), name='input')
mod1_WG3_hidden_1 = Dense(units=num_hidden1, activation="sigmoid", name='hidden1')(mod1_WG3_input_layer)
mod1_WG3_out = Dense(units=num_outputs, activation="relu", name="output")(mod1_WG3_hidden_1) 
  
mod1_WG4_input_layer = Input(shape=(num_inputs,), name='input')
mod1_WG4_hidden_1 = Dense(units=num_hidden1, activation="sigmoid", name='hidden1')(mod1_WG4_input_layer)
mod1_WG4_out = Dense(units=num_outputs, activation="relu", name="output")(mod1_WG4_hidden_1) 
  
mod1_WG5_input_layer = Input(shape=(num_inputs,), name='input')
mod1_WG5_hidden_1 = Dense(units=num_hidden1, activation="sigmoid", name='hidden1')(mod1_WG5_input_layer)
mod1_WG5_out = Dense(units=num_outputs, activation="relu", name="output")(mod1_WG5_hidden_1) 
    
# Erstelle das Modell für die unterschiedlichen Warengruppen
mod1_WG1 = Model(mod1_WG1_input_layer, mod1_WG1_out)
mod1_WG2 = Model(mod1_WG2_input_layer, mod1_WG2_out)
mod1_WG3 = Model(mod1_WG3_input_layer, mod1_WG3_out)
mod1_WG4 = Model(mod1_WG4_input_layer, mod1_WG4_out)
mod1_WG5 = Model(mod1_WG5_input_layer, mod1_WG5_out)

# show how the model looks
mod1_WG1.summary()

# Kompiliere das Modell:
# Wir verwenden das bekannte stochastic gradient descent (SGD) Verfahren.
# Die Lernreate muss festgelegt werden. Als loss-Funktion verwenden wir mean squared error (mse) als Standard.
opt = SGD(learning_rate=0.01) 
# model.compile(optimizer=opt,loss="mean_absolute_percentage_error",metrics=['mean_absolute_percentage_error'])
mod1_WG1.compile(optimizer=opt,loss="mse",metrics=['mae', 'mse'])
mod1_WG2.compile(optimizer=opt,loss="mse",metrics=['mae', 'mse'])
mod1_WG3.compile(optimizer=opt,loss="mse",metrics=['mae', 'mse'])
mod1_WG4.compile(optimizer=opt,loss="mse",metrics=['mae', 'mse'])
mod1_WG5.compile(optimizer=opt,loss="mse",metrics=['mae', 'mse'])

# Starte mit zufällig gewählten kleinen größen für die Gewichte und Schwellwerte
initial_weights = mod1_WG2.layers[-1].get_weights() 
initial_weights

# Trainiere das Modell:
# Die Anzahl der Trainings-Epochen und die Batchgröße sind festzulegen.
num_epochs = 50
batch_size = 10

mod1_WG1.fit(x=mod1_train_WG1_input, y=mod1_train_WG1_target, batch_size=batch_size, epochs=num_epochs, verbose=True)
mod1_WG2.fit(x=mod1_train_WG2_input, y=mod1_train_WG2_target, batch_size=batch_size, epochs=num_epochs, verbose=True)
mod1_WG3.fit(x=mod1_train_WG3_input, y=mod1_train_WG3_target, batch_size=batch_size, epochs=num_epochs, verbose=True)
mod1_WG4.fit(x=mod1_train_WG4_input, y=mod1_train_WG4_target, batch_size=batch_size, epochs=num_epochs, verbose=True)
mod1_WG5.fit(x=mod1_train_WG5_input, y=mod1_train_WG5_target, batch_size=batch_size, epochs=num_epochs, verbose=True)
final_weights = mod1_WG1.layers[-1].get_weights() 
final_weights

# Wende das trainierte Modell nun auf die Test-Daten an
mod1_test_WG1_pred = mod1_WG1.predict(mod1_test_WG1_input)
mod1_test_WG2_pred = mod1_WG2.predict(mod1_test_WG2_input)
mod1_test_WG3_pred = mod1_WG3.predict(mod1_test_WG3_input)
mod1_test_WG4_pred = mod1_WG4.predict(mod1_test_WG4_input)
mod1_test_WG5_pred = mod1_WG5.predict(mod1_test_WG5_input)
mod1_test_WG2_pred

# exportiere Modellvorhersagen als csv
np.savetxt(join(path_to_task,'df_MLP_test_mod1_WG1_pred.csv'), mod1_test_WG1_pred, delimiter=',') 
np.savetxt(join(path_to_task,'df_MLP_test_mod1_WG2_pred.csv'), mod1_test_WG2_pred, delimiter=',') 
np.savetxt(join(path_to_task,'df_MLP_test_mod1_WG3_pred.csv'), mod1_test_WG3_pred, delimiter=',') 
np.savetxt(join(path_to_task,'df_MLP_test_mod1_WG4_pred.csv'), mod1_test_WG4_pred, delimiter=',') 
np.savetxt(join(path_to_task,'df_MLP_test_mod1_WG5_pred.csv'), mod1_test_WG5_pred, delimiter=',') 


###################
# Modell 2 (mod2) #
###################

# Input-Variablen für Trainings- und Testdaten: Alle außer Datum, Jahr, Warengruppe, Umsatz
mod2_train_WG1_input = train_WG1[:,4:31]
mod2_train_WG2_input = train_WG2[:,4:31]
mod2_train_WG3_input = train_WG3[:,4:31]
mod2_train_WG4_input = train_WG4[:,4:31]
mod2_train_WG5_input = train_WG5[:,4:31]

mod2_test_WG1_input = test_WG1[:,4:31]
mod2_test_WG2_input = test_WG2[:,4:31]
mod2_test_WG3_input = test_WG3[:,4:31]
mod2_test_WG4_input = test_WG4[:,4:31]
mod2_test_WG5_input = test_WG5[:,4:31]

# Target-Variablen für Trainings- und Testdaten ist der Umsatz
mod2_train_WG1_target = train_WG1[:,3]
mod2_train_WG2_target = train_WG2[:,3]
mod2_train_WG3_target = train_WG3[:,3]
mod2_train_WG4_target = train_WG4[:,3]
mod2_train_WG5_target = train_WG5[:,3]

mod2_test_WG1_target = test_WG1[:,3]
mod2_test_WG2_target = test_WG2[:,3]
mod2_test_WG3_target = test_WG3[:,3]
mod2_test_WG4_target = test_WG4[:,3]
mod2_test_WG5_target = test_WG5[:,3]

# Anzahl der Input-Variablen
input_dim = mod2_train_WG1_input.shape[1]

# Definiere die Struktur des Multilayer Perceptrons (MLP)
num_inputs = input_dim
num_hidden1 = 50 # Anzahl der units im ersten hidden layer
num_outputs = 1

# verwende sigmoide Aktivierung im hidden layer und lineare Aktivierung (>= 0) 'relu' für den output
mod2_WG1_input_layer = Input(shape=(num_inputs,), name='input')
mod2_WG1_hidden_1 = Dense(units=num_hidden1, activation="sigmoid", name='hidden1')(mod2_WG1_input_layer)
mod2_WG1_out = Dense(units=num_outputs, activation="relu", name="output")(mod2_WG1_hidden_1) 
  
mod2_WG2_input_layer = Input(shape=(num_inputs,), name='input')
mod2_WG2_hidden_1 = Dense(units=num_hidden1, activation="sigmoid", name='hidden1')(mod2_WG2_input_layer)
mod2_WG2_out = Dense(units=num_outputs, activation="relu", name="output")(mod2_WG2_hidden_1) 
  
mod2_WG3_input_layer = Input(shape=(num_inputs,), name='input')
mod2_WG3_hidden_1 = Dense(units=num_hidden1, activation="sigmoid", name='hidden1')(mod2_WG3_input_layer)
mod2_WG3_out = Dense(units=num_outputs, activation="relu", name="output")(mod2_WG3_hidden_1) 
  
mod2_WG4_input_layer = Input(shape=(num_inputs,), name='input')
mod2_WG4_hidden_1 = Dense(units=num_hidden1, activation="sigmoid", name='hidden1')(mod2_WG4_input_layer)
mod2_WG4_out = Dense(units=num_outputs, activation="relu", name="output")(mod2_WG4_hidden_1) 
  
mod2_WG5_input_layer = Input(shape=(num_inputs,), name='input')
mod2_WG5_hidden_1 = Dense(units=num_hidden1, activation="sigmoid", name='hidden1')(mod2_WG5_input_layer)
mod2_WG5_out = Dense(units=num_outputs, activation="relu", name="output")(mod2_WG5_hidden_1) 
    
# Erstelle das Modell für die unterschiedlichen Warengruppen
mod2_WG1 = Model(mod2_WG1_input_layer, mod2_WG1_out)
mod2_WG2 = Model(mod2_WG2_input_layer, mod2_WG2_out)
mod2_WG3 = Model(mod2_WG3_input_layer, mod2_WG3_out)
mod2_WG4 = Model(mod2_WG4_input_layer, mod2_WG4_out)
mod2_WG5 = Model(mod2_WG5_input_layer, mod2_WG5_out)

# show how the model looks
mod2_WG1.summary()

# Kompiliere das Modell:
# Wir verwenden das bekannte stochastic gradient descent (SGD) Verfahren.
# Die Lernreate muss festgelegt werden. Als loss-Funktion verwenden wir mean squared error (mse) als Standard.
opt = SGD(learning_rate=0.01) 
# model.compile(optimizer=opt,loss="mean_absolute_percentage_error",metrics=['mean_absolute_percentage_error'])
mod2_WG1.compile(optimizer=opt,loss="mse",metrics=['mae', 'mse'])
mod2_WG2.compile(optimizer=opt,loss="mse",metrics=['mae', 'mse'])
mod2_WG3.compile(optimizer=opt,loss="mse",metrics=['mae', 'mse'])
mod2_WG4.compile(optimizer=opt,loss="mse",metrics=['mae', 'mse'])
mod2_WG5.compile(optimizer=opt,loss="mse",metrics=['mae', 'mse'])

# Starte mit zufällig gewählten kleinen größen für die Gewichte und Schwellwerte
initial_weights = mod2_WG2.layers[-1].get_weights() 
initial_weights

# Trainiere das Modell:
# Die Anzahl der Trainings-Epochen und die Batchgröße sind festzulegen.
num_epochs = 100
batch_size = 10

mod2_WG1.fit(x=mod2_train_WG1_input, y=mod2_train_WG1_target, batch_size=batch_size, epochs=num_epochs, verbose=True)
mod2_WG2.fit(x=mod2_train_WG2_input, y=mod2_train_WG2_target, batch_size=batch_size, epochs=num_epochs, verbose=True)
mod2_WG3.fit(x=mod2_train_WG3_input, y=mod2_train_WG3_target, batch_size=batch_size, epochs=num_epochs, verbose=True)
mod2_WG4.fit(x=mod2_train_WG4_input, y=mod2_train_WG4_target, batch_size=batch_size, epochs=num_epochs, verbose=True)
mod2_WG5.fit(x=mod2_train_WG5_input, y=mod2_train_WG5_target, batch_size=batch_size, epochs=num_epochs, verbose=True)
final_weights = mod2_WG1.layers[-1].get_weights() 
final_weights

# Wende das trainierte Modell nun auf die Test-Daten an
mod2_test_WG1_pred = mod2_WG1.predict(mod2_test_WG1_input)
mod2_test_WG2_pred = mod2_WG2.predict(mod2_test_WG2_input)
mod2_test_WG3_pred = mod2_WG3.predict(mod2_test_WG3_input)
mod2_test_WG4_pred = mod2_WG4.predict(mod2_test_WG4_input)
mod2_test_WG5_pred = mod2_WG5.predict(mod2_test_WG5_input)
mod2_test_WG2_pred

# exportiere Modellvorhersagen als csv
np.savetxt(join(path_to_task,'df_MLP_test_mod2_WG1_pred.csv'), mod2_test_WG1_pred, delimiter=',') 
np.savetxt(join(path_to_task,'df_MLP_test_mod2_WG2_pred.csv'), mod2_test_WG2_pred, delimiter=',') 
np.savetxt(join(path_to_task,'df_MLP_test_mod2_WG3_pred.csv'), mod2_test_WG3_pred, delimiter=',') 
np.savetxt(join(path_to_task,'df_MLP_test_mod2_WG4_pred.csv'), mod2_test_WG4_pred, delimiter=',') 
np.savetxt(join(path_to_task,'df_MLP_test_mod2_WG5_pred.csv'), mod2_test_WG5_pred, delimiter=',') 


###################
# Modell 3 (mod3) #
###################

# Input-Variablen für Trainings- und Testdaten: SommerferienSH, Feiertag, Silvester_ext, Samstag, Sonntag, Juli, August
mod3_train_WG2_input = train_WG2[:,[5,6,7,13,14,21,22]]
mod3_test_WG2_input = test_WG2[:,[5,6,7,13,14,21,22]]

# Target-Variablen für Trainings- und Testdaten ist der Umsatz
mod3_train_WG2_target = train_WG2[:,3]
mod3_test_WG2_target = test_WG2[:,3]

# rescale target to have values in the range [0,1]
# max(mod3_test_WG2_target)
# mod3_test_WG2_target_rescale = mod3_test_WG2_target/1000
# mod3_train_WG2_target_rescale = mod3_train_WG2_target/1000

# Anzahl der Input-Variablen
input_dim = mod3_train_WG2_input.shape[1]

# Definiere die Struktur des Multilayer Perceptrons (MLP)
num_inputs = input_dim
num_hidden1 = 100 # Anzahl der units im ersten hidden layer
num_hidden2 = 50
num_outputs = 1

# verwende sigmoide Aktivierung im hidden layer und lineare Aktivierung (>= 0) 'relu' für den output
mod3_WG2_input_layer = Input(shape=(num_inputs,), name='input')
mod3_WG2_hidden_1 = Dense(units=num_hidden1, activation="relu", name='hidden1')(mod3_WG2_input_layer)
mod3_WG2_hidden_2 = Dense(units=num_hidden2, activation="relu", name='hidden2')(mod3_WG2_hidden_1)
mod3_WG2_out = Dense(units=num_outputs, activation="relu", name="output")(mod3_WG2_hidden_2) 
 
 
# Erstelle das Modell für die unterschiedlichen Warengruppen
mod3_WG2 = Model(mod3_WG2_input_layer, mod3_WG2_out)

# show how the model looks
mod3_WG2.summary()

# Kompiliere das Modell:
# Wir verwenden das bekannte stochastic gradient descent (SGD) Verfahren.
# Die Lernreate muss festgelegt werden. Als loss-Funktion verwenden wir mean squared error (mse) als Standard.
#opt = SGD(learning_rate=0.01) 
opt = Adam(learning_rate=0.01) 
# model.compile(optimizer=opt,loss="mean_absolute_percentage_error",metrics=['mean_absolute_percentage_error'])
mod3_WG2.compile(optimizer=opt,loss="mse",metrics=['mae', 'mse'])

# Starte mit zufällig gewählten kleinen größen für die Gewichte und Schwellwerte
initial_weights = mod3_WG2.layers[-1].get_weights() 
initial_weights

# Trainiere das Modell:
# Die Anzahl der Trainings-Epochen und die Batchgröße sind festzulegen.
num_epochs = 10
batch_size = 3

mod3_WG2.fit(x=mod3_train_WG2_input, y=mod3_train_WG2_target_rescale, batch_size=batch_size, epochs=num_epochs, verbose=True)

final_weights = mod3_WG2.layers[-1].get_weights() 
final_weights

# Wende das trainierte Modell nun auf die Test-Daten an
mod3_test_WG2_pred = mod3_WG2.predict(mod3_test_WG2_input)
mod3_test_WG2_pred

# exportiere Modellvorhersagen als csv
np.savetxt(join(path_to_task,'df_MLP_test_mod3_WG2_pred.csv'), mod3_test_WG2_pred, delimiter=',') 


###################
# Modell 4 (mod4) WG2#
###################

# Lade Trainings- und Testdaten aus R Export
path_to_task = "/Users/Marco Landt-Hayen/Documents/R/Application_Project/data"

train_WG2 = np.loadtxt(join(path_to_task,'df_SVM_train_WG2.csv'), delimiter=',') 
test_WG2 = np.loadtxt(join(path_to_task,'df_SVM_test_WG2.csv'), delimiter=',') 

np.shape(train_WG2)

# Input-Variablen für Trainings- und Testdaten: Alle außer Datum, Jahr, Warengruppe, Umsatz
mod4_train_WG2_input = train_WG2[:,4:46]
mod4_test_WG2_input = test_WG2[:,4:46]

# Target-Variablen für Trainings- und Testdaten ist der Umsatz
mod4_train_WG2_target = train_WG2[:,2]
mod4_test_WG2_target = test_WG2[:,2]

# Anzahl der Input-Variablen
input_dim = mod4_train_WG2_input.shape[1]

input_dim

# Modell aufsetzen
num_inputs = input_dim
num_hidden = [100,50] # for each hidden layer: number of hidden units in form of a python list   # FIX!!!
num_outputs = 1 # predict single number in case of Eckerle4

activation = 'relu' # activation of hidden layers   # FIX!!!
dropout = 0 # 0 if no dropout, else fraction of dropout units (e.g. 0.2)   # FIX!!!

# Sequential network structure.
model = Sequential()

if len(num_hidden) == 0:
  print("Error: Must at least have one hidden layer!")
  sys.exit()  

# add first hidden layer connecting to input layer
model.add(Dense(num_hidden[0], input_dim=num_inputs, activation=activation))

if dropout: 
  # dropout of fraction dropout of the neurons and activation layer.
  model.add(Dropout(dropout))
  # model.add(Activation("linear"))

# potentially further hidden layers
for i in range(1, len(num_hidden)):
  # add hidden layer with len[i] neurons
  model.add(Dense(num_hidden[i], activation=activation))
#  model.add(Activation("linear"))

  if dropout:
  # dropout of fraction dropout of the neurons and activation layer.
    model.add(Dropout(dropout))
  #  model.add(Activation("linear"))

# output layer
model.add(Dense(1))

# show how the model looks
model.summary()

# compile model
#opt = SGD(learning_rate=0.01) # FIX!!!
#opt = Adadelta(learning_rate=0.001) # FIX!!!
opt = Adam(learning_rate=0.001) # FIX!!!
model.compile(loss='mse', optimizer=opt, metrics=["mae", "mse"])# FIX!!!

# Training model with train data. Fixed random seed:
np.random.seed(3)
num_epochs = 20   # FIX !!!
batch_size = 1 # FIX !!! 
history = model.fit(mod4_train_WG2_input, mod4_train_WG2_target, epochs=num_epochs, batch_size=batch_size, verbose=2)

# Modell auf Testdaten anwenden
mod4_test_WG2_pred = model.predict(mod4_test_WG2_input)
mod4_test_WG2_pred

# exportiere Modellvorhersagen als csv
np.savetxt(join(path_to_task,'df_MLP_test_mod4_WG2_pred.csv'), mod4_test_WG2_pred, delimiter=',') 



###################
# Modell 4 (mod4) WG4#
###################

# Lade Trainings- und Testdaten aus R Export
path_to_task = "/Users/Marco Landt-Hayen/Documents/R/Application_Project/data"

train_WG4 = np.loadtxt(join(path_to_task,'df_SVM_train_WG4.csv'), delimiter=',') 
test_WG4 = np.loadtxt(join(path_to_task,'df_SVM_test_WG4.csv'), delimiter=',') 

np.shape(train_WG4)

# Input-Variablen für Trainings- und Testdaten: Alle außer Datum, Jahr, Warengruppe, Umsatz
mod4_train_WG4_input = train_WG4[:,4:46]
mod4_test_WG4_input = test_WG4[:,4:46]

# Target-Variablen für Trainings- und Testdaten ist der Umsatz
mod4_train_WG4_target = train_WG4[:,2]
mod4_test_WG4_target = test_WG4[:,2]

# Anzahl der Input-Variablen
input_dim = mod4_train_WG4_input.shape[1]

input_dim

# Modell aufsetzen
num_inputs = input_dim
num_hidden = [100,50] # for each hidden layer: number of hidden units in form of a python list   # FIX!!!
num_outputs = 1 # predict single number in case of Eckerle4

activation = 'relu' # activation of hidden layers   # FIX!!!
dropout = 0 # 0 if no dropout, else fraction of dropout units (e.g. 0.2)   # FIX!!!

# Sequential network structure.
model = Sequential()

if len(num_hidden) == 0:
  print("Error: Must at least have one hidden layer!")
  sys.exit()  

# add first hidden layer connecting to input layer
model.add(Dense(num_hidden[0], input_dim=num_inputs, activation=activation))

if dropout: 
  # dropout of fraction dropout of the neurons and activation layer.
  model.add(Dropout(dropout))
  # model.add(Activation("linear"))

# potentially further hidden layers
for i in range(1, len(num_hidden)):
  # add hidden layer with len[i] neurons
  model.add(Dense(num_hidden[i], activation=activation))
#  model.add(Activation("linear"))

  if dropout:
  # dropout of fraction dropout of the neurons and activation layer.
    model.add(Dropout(dropout))
  #  model.add(Activation("linear"))

# output layer
model.add(Dense(1))

# show how the model looks
model.summary()

# compile model
#opt = SGD(learning_rate=0.01) # FIX!!!
#opt = Adadelta(learning_rate=0.001) # FIX!!!
opt = Adam(learning_rate=0.001) # FIX!!!
model.compile(loss='mse', optimizer=opt, metrics=["mae", "mse"])# FIX!!!

# Training model with train data. Fixed random seed:
np.random.seed(3)
num_epochs = 20   # FIX !!!
batch_size = 1 # FIX !!! 
history = model.fit(mod4_train_WG4_input, mod4_train_WG4_target, epochs=num_epochs, batch_size=batch_size, verbose=2)

# Modell auf Testdaten anwenden
mod4_test_WG4_pred = model.predict(mod4_test_WG4_input)
mod4_test_WG4_pred

# exportiere Modellvorhersagen als csv
np.savetxt(join(path_to_task,'df_MLP_test_mod4_WG4_pred.csv'), mod4_test_WG4_pred, delimiter=',') 








