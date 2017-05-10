import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
# load tweets and convert to lower case # for final we will not want to do this we can get a lot of character in the capitalization.
filename = "tweettext.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# creating mapping from character to number for all unique charatcers in data set

chars = sorted(list(set(raw_text)))
char_to_int = dict((c,i) for i, c in enumerate(chars))

# prepare the dataset of inputs 
n_chars = len(raw_text)
n_vocab = len(chars )
seq_length = 10
dataX =[]
dataY= []

for i in range(0,n_chars-seq_length,1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i+seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patters: ", n_patterns)


nX = dataX
nY = dataY

# Reshape X to be [samples, time steps, features] expected by LSTM
X = np.reshape(nX,(len(nX),seq_length,1))
# Normalize
X = X/float(n_vocab)
#one hot encoding
y = np_utils.to_categorical(nY)


# build network

TrumpNet = Sequential()
TrumpNet.add(LSTM(256,input_shape=(X.shape[1],X.shape[2])))
TrumpNet.add(Dropout(0.2))
TrumpNet.add(Dense(y.shape[1],activation='softmax'))
TrumpNet.compile(loss='categorical_crossentropy', optimizer='adam')
TrumpNet.summary()

# define checkpoint
filepath='weights-improvment-{epoch:02d}--{loss:4f}.hdf5'
checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbacks_list = [checkpoint]


# TRAIN!

TrumpNet.fit(X,y,epochs=20,batch_size=100,callbacks=callbacks_list)



filename = 'weights-improvment-19--2.883339.hdf5'
TrumpNet.load_weights(filename)
TrumpNet.compile(loss ='categorical_crossentropy', optimizer='adam')

int_to_char = dict((i,c) for i, c in enumerate(chars))

# Pick a random seed
start = np.random.randint(0,len(nX)-1)
pattern = nX[start]
print("Seed:")
print('\'' , ''.join([int_to_char[value] for value in pattern]),'\'')

# Generate Characters
for i in range(10):
    x = np.reshape(pattern,(1,len(pattern),1))
    x = x/float(n_vocab)
    prediction = TrumpNet.predict(x,verbose=0)
    index = np.argmax(prediction[0][:])
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print("\nDone.")
