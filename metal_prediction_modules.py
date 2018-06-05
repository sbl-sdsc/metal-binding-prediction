
# coding: utf-8

# # Data Generator

# In[ ]:


import numpy as np
import keras


# In[ ]:


class OneHotGenerator(keras.utils.Sequence):
    
    'Gererate data for Keras'
    def __init__(self, 
                 sequences, 
                 labels, 
                 cluster_numbers,
                 translator,
                 batch_size = 100, 
                 input_shape=(706,20),
                 label_shape=(706,)):
        'Initialization'
        self.sequences = sequences
        self.labels = labels
        
        #Clusters
        self.clusters = {}
        for i, c in enumerate(cluster_numbers):
            if c not in self.clusters.keys():
                self.clusters[c] = []
            self.clusters[c].append(i)
        
#         print ("Number of clusters", len(self.clusters.keys()))
        
        self.translator = translator
        
        # Referenced by Trainer
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.label_shape = label_shape
        
        self.on_epoch_end()
    
    def __len__(self):
        'Get the number of batches per epoch'
        return int(np.floor(len(self.sequences))/self.batch_size)
        
    def __getitem__(self, index):
        'Generate one batch of data'
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        
        sequences_temp = [self.sequences[k] for k in indices]
        labels_temp = [self.labels[k] for k in indices]
        
        X, y = self.generate_data(sequences_temp, labels_temp)
        return X, y
        
    def on_epoch_end(self):
        'Update indices at the end of each epoch'
        self.indices = []
        for i in range(len(self.sequences)):
            cluster_index = np.random.choice(list(self.clusters.keys()))
            self.indices.append(np.random.choice(self.clusters[cluster_index]))
                    
    def generate_data(self, sequences_temp, labels_temp):
        'Populate input and label tensors'
        X = np.zeros((self.batch_size, *self.input_shape))
        y = np.zeros((self.batch_size, *self.label_shape), dtype=int)
        
        cutoff = self.input_shape[0]
        dim = self.input_shape[1]
        # len(seq) < cutoff - zero padding
        # len(seq) >= cutoff - truncate to cutoff
        
        for i, seq in enumerate(sequences_temp):
            
            length = len(seq)
            for j, c in enumerate(seq):
                if j < cutoff:
                    X[i,j,] = np.array(self.translator[c])

            for j, f in enumerate(labels_temp[i]):
                if f < cutoff:
                    y[i,f,] = 1
            
        return X, y

from Zihan.fofe import FofeVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class FOFEGenerator(keras.utils.Sequence):

    'Gererate data for Keras'
    def __init__(self,
                 sequences,
                 labels,
                 translator,
                 batch_size = 100,
                 input_shape=(800,),
                 label_shape=(7,),
                 shuffle=True):
        'Initialization'
        self.sequences = sequences
        self.labels = labels
        
       
        #Clusters
        self.clusters = {}
        for i, c in enumerate(labels):
            if c[0] not in self.clusters.keys():
                self.clusters[c[0]] = []
            self.clusters[c[0]].append(i)
    
        self.translator = translator
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.shuffle = shuffle

        self.on_epoch_end()
        

    def __len__(self):
        'Get the number of batches per epoch'
        return int(np.floor(len(self.sequences))/self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        sequences_temp = [self.sequences[k] for k in indices]
        labels_temp = [self.labels[k] for k in indices]

        X, y = self.generate_data(sequences_temp, labels_temp)
        return X, y

    def on_epoch_end(self):
        'Update indices at the end of each epoch'
        self.indices = []
        if self.shuffle == True:
            for i in range(len(self.sequences)):
                cluster_index = np.random.choice(list(self.clusters.keys()))
                self.indices.append(np.random.choice(self.clusters[cluster_index])) 
        else: 
            self.indices = np.arange(len(self.sequences))
                                     
    def generate_data(self, sequences_temp, labels_temp):
        'Populate input and label tensors'
        X = np.zeros((self.batch_size, *self.input_shape))
        y = np.zeros((self.batch_size, *self.label_shape), dtype=int)

        # len(seq) < cutoff - zero padding
        # len(seq) >= cutoff - truncate to cutoff

        # convert string to list of amino acid 
        seq_ls = []
        for i in sequences_temp:
            seq_ls.append(list(i))
            
        two_gram_ls = []
        for i in seq_ls:
            cur_two_gram = []
            for j in range(len(i)-1):
                cur_two_gram.append(i[j]+i[j+1])
            two_gram_ls.append(cur_two_gram)
        #print (two_gram_ls[0])
    
        alpha = 0.95
        cutoff = 706

        fofe = FofeVectorizer(alpha)
        features = fofe.naive_transform(two_gram_ls, self.translator)

        #print (features[0])
        seq_ls_cut_rev = []
        for i in two_gram_ls:
            seq_ls_cut_rev.append(list(reversed(i)))
        features_rev = fofe.naive_transform(seq_ls_cut_rev, self.translator)

        # normalize feature matrix by subtracting each value by mean/std
        for i in range(len(features)):
            tmp_ls = np.array(features[i])
            tmp_std = np.std(tmp_ls, dtype=np.float32)
            tmp_mean = np.mean(tmp_ls, dtype=np.float32)
            for j in range(len(tmp_ls)):
                features[i][j]=tmp_ls[j]-tmp_mean/tmp_std

        # normalize feature matrix by subtracting each value by mean/std
        for i in range(len(features_rev)):
            tmp_ls = np.array(features_rev[i])
            tmp_std = np.std(tmp_ls, dtype=np.float32)
            tmp_mean = np.mean(tmp_ls, dtype=np.float32)
            for j in range(len(tmp_ls)):
                features_rev[i][j]=tmp_ls[j]-tmp_mean/tmp_std

        # Combine features and features_rev
        features_new = []
        for i in range(len(features)):
            features_new.append(np.concatenate((features[i],features_rev[i])))

        X = np.array(features_new)

        for i, seq in enumerate(sequences_temp):
            for j, f in enumerate(labels_temp[i]):
                if f < cutoff:
                    y[i,f,] = 1
                    
        #print (y[0])

        return X, y


# # Model Trainer

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


class Trainer():
    def __init__(self, 
                 model,
                 generators,
                 callbacks,
                 post_train_args):
        
        self.model = model
        
        assert len(generators) == 2
        self.train_gen = generators[0]
        self.val_gen = generators[1]
        self.callbacks = callbacks
        
        print ("Assigning validation generator...",end=' ')
        for cb in callbacks:
            if cb.validation_generator is None:
                cb.validation_generator = self.val_gen
        print ("Done")
        
        print ("Matching input shape...", end=' ')            
        assert self.model.layers[0].input_shape[1:] == self.train_gen.input_shape
        print ("Done")
        
        print ("Matching output shape...", end=' ')
        assert self.model.layers[-1].output_shape[1:] == self.train_gen.label_shape
        print ("Done")            
            
        assert self.train_gen.batch_size == self.val_gen.batch_size        
        self.batch_size = self.train_gen.batch_size
        print ("Trainer initialized.")
        
        self.post_train_args = post_train_args
        
    def start(self, epoch=1):
        assert epoch > 0 and isinstance(epoch, int)
        # Assume the model is compiled for now
        self.model.fit_generator(epochs=epoch,
                                 generator=self.train_gen,
#                                  validation_data=self.val_gen,
                                 callbacks=self.callbacks,
                                 use_multiprocessing=False, 
                                 workers=4,
                                 verbose=1)
        self.post_train(epoch)
        
#     def stop(self, error_string="Unspecified"):
#         self.model.stop_training = True
#         print ("Training stopped. Reason:", error_string)
    
    def post_train(self, epoch):
        print ("[End of Training]")
        for c in self.callbacks:
            c.post_train(epoch=epoch, batch=len(self.train_gen), post_train_args=self.post_train_args)

        
        