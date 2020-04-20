# SisFall Detection

## Overview

SisFall is a <u>[research](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5298771/)</u> about fall and movement detection with wearable devices and researchers have collected a large amount of data from  both young and old people during the research. In this little project, I only select four classes of data and train them by using RNN.

### Data set
The whole dataset is available in <u>[here](http://sistemic.udea.edu.co/en/research/projects/english-falls/)</u>.

The four classes I chosen:
- F01	Fall forward while walking caused by a slip	
- F02	Fall backward while walking caused by a slip
- D01	Walking slowly
- D02	Walking quickly

### Data process and Feature selection
This process can be seen in LoadFeatures.py . 
- Data:
Generally, I choose the data measured by the sensor ADXL345 and convert the acceleration data into gravity, using this equation: 

Acceleration [g]: [(2·Range)/(2^Resolution)]·data.

According to the research, the effect of filtering is not stable and doesn't work on every features so I didn't use the filtered data before applying the features.

- Feature:
I choose threee features in the training process.(They are three of five features that best performed in the research.)

C2:Sum vector magnitude on horizontal plane

```
def vectorsum (Acc_x, Acc_z):
    x = []
    z = []
    x = [i*i for i in Acc_x]
    z = [i*i for i in Acc_z]
    sum = [a+b for a,b in zip(x,z)]
    vectorsum_res = [math.sqrt(i) for i in sum]
    return vectorsum_res
```

C8:Standard deviation magnitude on horizontal plane

```
def deviation0 (Acc_x, Acc_z):
    s = [0 for _ in range(100)]
    for i in range(100):
        s[i] = math.sqrt(np.var(Acc_z[i:i+25]) + np.var(Acc_x[i:i+25]))
    return s
```

C9:Standard deviation magnitude

```
def deviation1 (Acc_x, Acc_y,Acc_z):
    s = [0 for _ in range(100)]
    for i in range(100):
        s[i] = math.sqrt(np.var(Acc_x[i:i+25]) + np.var(Acc_y[i:i+25])+ np.var(Acc_y[i:i+25]))
    return s
```

The calculated data of features wil be saved as a pickle file.

### RNN model
A recurrent neural network (RNN) is a class of artificial neural networks where connections between nodes form a directed graph along a temporal sequence. Keras was used to build the model. Because it's a multi classification problem，I choose the loss functioon 'categorical_crossentropy' and used activation 'softmax'.
```
model_RNN = tf.keras.models.Sequential\
([
    tf.keras.layers.LSTM(10, input_shape=(X_train.shape[1], X_train.shape[2])), 
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
model_RNN.compile(optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'], loss='categorical_crossentropy')
```
The trained model wil be saved as a pickle file.

### Result
- C2:

![Accuracy](/C2/SisFall_Accuracy(BF).png)

![Loss](/C2/SisFall_Loss(BF).png)

- C8:

![Accuracy](/C8/SisFall_Accuracy(BF).png)

![Loss](/C8/SisFall_Loss(BF).png)

- C9:

![Accuracy](/C9/SisFall_Accuracy(BF).png)

![Loss](/C9/SisFall_Loss(BF).png)

As is shown in the picture, the accuracy of three features are all not very good.I thougt maybe there's something wrong with my model or the method I processed the data. Now I still didn't figure it out after choosing different functions and dividing the dataset in different ways. And I will keep working on it.

