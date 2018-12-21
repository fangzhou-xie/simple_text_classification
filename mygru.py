import numpy as np
import pandas as pd
import keras
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.preprocessing.text import Tokenizer
from keras.losses import hinge 
from keras.models import load_model
import pdb

max_words = 500    #only look at the most frequent 1000 words
batch_size = 32
epochs = 5

def dataRead(data_name):
    df = pd.read_csv(data_name,encoding='utf-8',delimiter='\t',header=None)
    df = df.values #return array
    print(df.shape)
    text, value = df[:,0], df[:,1]
    text, value = shuffle(text, value,random_state=43)
    # pdb.set_trace()
    return text, value

def textTokenize(text,value):
    """
    using tokenizer to make text into index numbers, although limited by max_words,
    tk will preserve all word count, even though the rest will not be used
    """
    tk = Tokenizer(num_words=max_words, filters='“”!"#$%&()*+,-./:;<=>?@[]^_`{|}~',
                 split=' ',lower=True) 
    tk.fit_on_texts(text)
    # print(tk.document_count)
    # print(tk.word_index)
    # print(tk.word_counts)
    # print(tk.word_docs)     #A dictionary of words and their counts.
    print("total {} sentences".format(tk.document_count))
    # pdb.set_trace()
    X = tk.texts_to_matrix(text, mode='binary')
    y = keras.utils.to_categorical(value, 2)
    # encoded_text = np.array(encoded_text)
    return X, y

def model_gru(x_train, x_test, y_train, y_test):
    # x_train, x_test, y_train, y_test=trainsetGenerate(dem_file_names,rep_file_names)
    print('Building model...')
    model = Sequential()
    model.add(Embedding(max_words,16,input_length=max_words))
    # model.add(Dense(200,activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(200, return_sequences=True)))
    model.add(Bidirectional(GRU(200)))
    # model.add(Dropout(0.2))
    model.add(Dense(250,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    print(model.summary())
    # pdb.set_trace()
    model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
    score = model.evaluate(x_test, y_test,
                       batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    return model


def main():
    inset, outset = dataRead('train.tsv')
    X, y = textTokenize(inset, outset)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=43)
    # pdb.set_trace()
    # fileWrite()
    mymodel = model_gru(x_train, x_test, y_train, y_test)
    mymodel.save('/Users/xiefangzhou/workspace/papers/pctpaper/BERT/rwcp/lstm/model_gru.h5') 
    

if __name__ == "__main__":
    main()