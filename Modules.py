
# coding: utf-8

# # Data Generator

# In[5]:


import numpy as np
import keras


# ### Note that OneHotGenerator supports single char to vector translation

# In[3]:


class OneHotGenerator(keras.utils.Sequence):
    
    'Gererate data for Keras'
    def __init__(self, 
                 sequences, 
                 labels, 
                 translator,
                 batch_size = 100, 
                 input_shape=(706,20),
                 label_shape=(706,),
                 shuffle=True):
        'Initialization'
        self.sequences = sequences
        self.labels = labels
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
        self.indices = np.arange(len(self.sequences))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
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
        


# In[4]:


class ProtVecGenerator(keras.utils.Sequence):
    
    'Gererate data for Keras'
    def __init__(self, 
                 sequences, 
                 labels, 
                 translator,
                 batch_size = 100, 
                 input_shape=(706,20),
                 label_shape=(706,),
                 shuffle=True, 
                 n_gram=3):
        'Initialization'
        self.sequences = sequences
        self.labels = labels
        self.translator = translator
        
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.label_shape = label_shape
        self.shuffle = shuffle
        self.n_gram=n_gram
        
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
        self.indices = np.arange(len(self.sequences))
        if self.shuffle == True:
            np.random.shuffle(self.indices)
            
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
            offset = self.n_gram - 1
            for j in range(length-offset):
                if j+offset < cutoff:
                    word = seq[j:j+self.n_gram]
                    if word in self.translator.keys():
                        X[i,j,] = np.array(self.translator[word])

            for j, f in enumerate(labels_temp[i]):
                if f < cutoff:
                    y[i,f,] = 1
            
        return X, y     


# In[5]:


class GeneratorArray(keras.utils.Sequence):
    
    'Gererate data for Keras'
    def __init__(self, generators):
        'Initialization'
        for gen in generators:
            assert gen is not None
        self.generators = generators
        
        self.on_epoch_end()
    
    def __len__(self):
        'Get the number of batches per epoch'
        return min([len(gen) for gen in self.generators])
        
    def __getitem__(self, index):
        'Generate one batch of data'
        X = []
        Y = None
        for gen in self.generators:
            x, y = gen[index]
            X.append(x)
            if Y is None:
                Y = y
            assert Y == y
            
        return X, Y
        
    def on_epoch_end(self):
        'Update indices at the end of each epoch'
        for gen in self.generators:
            gen.on_epoch_end()


# In[6]:


class GenericGenerator(keras.utils.Sequence):
    
    'Gererate data for Keras'
    def __init__(self, 
                 sequences, 
                 labels, 
                 translator,
                 generate_func,
                 batch_size = 100, 
                 input_shape=(706,20),
                 label_shape=(706,),
                 shuffle=True):
        'Initialization'
        self.sequences = sequences
        self.labels = labels
        self.translator = translator
        self.generate_func = generate_func
        
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
        
        X, y = self.generate_func(sequences_temp, labels_temp)
        return X, y
        
    def on_epoch_end(self):
        'Update indices at the end of each epoch'
        self.indices = np.arange(len(self.sequences))
        if self.shuffle == True:
            np.random.shuffle(self.indices)


# # Model Trainer

# In[7]:


import matplotlib.pyplot as plt


# In[8]:


class Trainer():
    def __init__(self, 
                 model,
                 generators,
                 callbacks):
        
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
        self.post_train()
        
#     def stop(self, error_string="Unspecified"):
#         self.model.stop_training = True
#         print ("Training stopped. Reason:", error_string)
    
    def post_train(self):
        print ("[End of Training]")
        for c in self.callbacks:
            c.post_train(epoch=self.epoch, batch=len(self.train_gen))
        


# # Evaluator

# In[9]:


from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np


# In[10]:


class F1_history(keras.callbacks.Callback):
    def __init__(self, 
                 threshold_func):
        print ("Callback initialized.")
        self.threshold_func = threshold_func
        
        self.validation_generator = None
        self.validation_steps = 0
        
        self.f1_scores = []
        self.precisions = []
        self.recalls = []        
        
    def on_train_begin(self, logs={}):
#         self.epoch_count = 0
        self.validation_steps = len(self.validation_generator)
        self.f1_scores = []
        self.precisions = []
        self.recalls = []
        return
    
    def on_train_end(self, logs={}):
        return

    
    def on_epoch_begin(self, epoch, logs={}):
#         self.batch_count = 0
#         self.epoch_count += 1
        return
    
    def on_epoch_end(self, epoch, logs={}):
        return

    def on_batch_begin(self, batch, logs={}):
#         self.batch_count += 1
        return

    def on_batch_end(self, batch, logs={}):
        rand_index = np.random.randint(self.validation_steps)
        X_val, y_val = self.validation_generator[rand_index]
        
        pred_val = self.model.predict_on_batch(X_val) 
        pred_val = self.threshold_func(pred_val).astype(int)
        label_val = y_val
        
        f1_val = f1_score(label_val.ravel(), pred_val.ravel())
        precision_val = precision_score(label_val.ravel(), pred_val.ravel())
        recall_val = recall_score(label_val.ravel(), pred_val.ravel())
        
        self.f1_scores.append(round(f1_val,4))
        self.precisions.append(round(precision_val,4))
        self.recalls.append(round(recall_val,4))
#         print (' F1', round(f1_val, 3))
        return

    def post_train(self, epoch, batch):
        F1_over_epoch = []
        for i in range(self.epoch):
            F1_over_epoch.append(np.average(cb.f1_scores[i*batch:(i+1)*batch]))

        fig = plt.figure(0)
        ax = fig.add_subplot(111)
        ax.set_title('F1 score over batch')
        ax.plot(F1_over_epoch)
        fig.canvas.draw()
        plt.show()
        # plt.savefig(fig_path + "sgbrnn_oh_adam_31.png")

