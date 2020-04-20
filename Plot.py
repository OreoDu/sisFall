# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:19:27 2020

@author: dujing1107
"""

import pickle
import matplotlib.pyplot as plt

# Loading saved model & model history (before adding filter)
print("Loading model's history (before filter)...")
historyBF = pickle.load(open("HistoryBF_RNN.pickle", "rb"))

print("Plotting graphs...")
# Plot training & validation accuracy values
fig1 = plt.figure(1)
plt.plot(historyBF['accuracy'])
plt.plot(historyBF['val_accuracy'])
plt.title('Model Accuracy Before Filter')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='center right')
plt.savefig('SisFall_Accuracy(BF).png')

# Plot training & validation loss values
plt.figure(2)
plt.plot(historyBF['loss'])
plt.plot(historyBF['val_loss'])
plt.title('Model Loss Before Filter')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='center right')
plt.savefig('SisFall_Loss(BF).png')

