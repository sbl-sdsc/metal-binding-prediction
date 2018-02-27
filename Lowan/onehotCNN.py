import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt

# Importing all the necessary libraries to build CNN
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense

# Dictionary of Amino Acids
AADICT = {'A' : 0, 'C' : 1, 'D' : 2, 'E' : 3, 'F' : 4, 'G' : 5, 'H' : 6, 'I' : 7, 'K' : 8, \
          'L' : 9, 'M' : 10, 'N' : 11, 'P' : 12, 'Q' : 13, 'R' : 14, 'S' : 15, 'T' : 16, 'V' : 17, \
          'W' : 18, 'Y' : 19}

def one_hot_encode( seq ) :
    mat = list()
    for aa in seq :
        vec = [0] * 20
        if not aa == '0' :
            vec[ AADICT[aa] ] = 1
        mat.append(vec)
    return np.array(mat).transpose()

def main() :
    # Grab all data from parquet file
    dataset = pd.read_parquet( 'Metal_all_20180116.snappy.parquet' )

    # fix random seed for reproducibility
    np.random.seed( 7 )

    ''' PART 1. filter 'X', 'U', 'B' '''
    # Boolean Column to indicate which row contains 'X' and 'U'.
    xub_flag = list()
    seq_len = list()   # This will be used later to get rid of outliers
    for row in dataset.itertuples() :
        if ('X' in row[5]) or ('U' in row[5]) or ('B' in row[5]) : xub_flag.append(1)
        else : xub_flag.append(0)
        seq_len.append( len(row[5]) )

    # Append the columns
    dataset['XUBflag'] = xub_flag
    dataset['seqLength'] = seq_len

    # Exclude sequences containing 'X' and 'U' and 'B'
    dataset = dataset[dataset.XUBflag != 1]
    dataset = dataset.drop('XUBflag', axis=1)

    ''' PART 2. Get rid of outliers based on Sequence Length '''
    # Sort by 'seqLength'
    dataset = dataset.sort_values( by=['seqLength'], ascending=False )

    # Get standard deviation and mean of sequence length.
    seq_len = np.array(seq_len)
    stdev = np.std(seq_len, dtype=np.float32)
    mean = np.mean(seq_len, dtype=np.float32)

    # 3 standard deviations from left and right side (0.2% from each side)
    std3 = int(len(dataset) * 0.002)
    dataset = dataset[:-std3]
    dataset = dataset[std3:]

    ''' PART 3. Zero Padd to Maximum Length '''
    maxlength = dataset.iloc[0, len(dataset.columns)-1]

    # Zero padd to the max length
    for row in dataset.itertuples() :
        if not row[len(dataset.columns)] == maxlength :
            dataset.loc[row.Index, 'sequence'] = row[5] + ('0'*(maxlength-row[len(dataset.columns)]))

    ''' PART 3.1 Split into Training Set and Test Set (67% : 33%)'''
    train_num = int(len(dataset) * 0.67)
    test_num = len(dataset) - train_num

    # Splitting Input Data from the Dataframe
    x_train = dataset[:train_num]
    x_test = dataset[train_num:]

    ''' PART 4. Get Input Data as One-Hot '''
    indata = list()
    indata_test = list()

    for row in x_train.itertuples() :
        indata.append( one_hot_encode(row[5]) )

    for row in x_test.itertuples() :
        indata_test.append( one_hot_encode(row[5]) )

    ''' Save Output Data as 1 or 0 (Binary Classification) '''
    outdata = list()
    outdata_test = list()

    for row in x_train.itertuples() :
        if row[2] == 'ZN' : outdata.append(1)
        else : outdata.append(0)

    for row in x_test.itertuples() :
        if row[2] == 'ZN' : outdata_test.append(1)
        else : outdata_test.append(0)

    ''' BUILD CONVOLUTIONAL NEURAL NETOWRK '''
    classifier = Sequential()
    # This will create 32 Feature Maps.
    classifier.add(Conv2D(32, (3,3), input_shape=(20, maxlength, 1), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    # Don't need 'input_shape' because it's already given from previous step.
    classifier.add(Conv2D(32, (3,3), activation="relu"))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Dropout(0.25))
    # Flattening
    classifier.add(Flatten())
    # Choose units to be not too big but not too small. Common practice is to pick power of 2.
    classifier.add(Dense(units=128, activation="relu"))
    # Hidden layer to Output layer
    classifier.add(Dense(units=1, activation="sigmoid"))
    # Binary_crossentropy because we are classifying 2 outcomes.
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # Fit the model (TRAINING)
    indata = np.array(indata).reshape(len(x_train), 20, maxlength, 1)
    start = time.time()
    classifier.fit(indata, outdata, validation_split=0.33, epochs=20, batch_size=10)
    end = time.time()
    print("Model took %0.2f seconds" %(end - start))
    score = classifier.evaluate(indata_test, outdata_test, verbose=0)
    print(score)



if __name__ == '__main__' :
    main()
