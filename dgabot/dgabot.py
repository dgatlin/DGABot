#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation 
from keras.layers.embeddings import Embedding 
from keras.layers.recurrent import LSTM 
from keras.models import load_model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score,f1_score, roc_auc_score
#from io import StringIO
import tldextract
from utils import topLevel, validChars
#import logging 


class DGABot: 
#TODO implement the logging funcitonality accross all methods 
    
    def __init__(self,data_path=None,model_path=None,df=None): 
        """
        Constructor to set up path vaiables for model and sample data and 
        initialized object. 

        Parameters
        ----------
        data_path: str or None 
            This variable is the path to the location of where the data is stored
            on disk. 
            
        model_path: str or None 
            This variable is the path to the location of where the produciton 
            model is saved on disk. 
            
        df: pandas dataframe or None 
            This will allow the the object to be instantiated with a pre determined 
            dataframe 
        """
        #self.logger = logging.getLogger()
        
        self.clear_state()
        self.model_path = model_path 
        self.data_path = data_path 
        self.df = df 
        
        
    def clear_state(self): 
        """
        Resets object's state to clear our all model internals created after 
        loading state from dist
        
        """
        self.model = None 
        self.modeldata = None
        self.model_path = None 
        self.data_path = None  
        
        
    def set_paths(self, data_path=None, model_path=None): 
        """
        Helper function to set paths 
 
    
        Parameters
        ----------
        """
        #TODO implement the return and error handing funcitonality

         
    def build_model(self, max_features, maxlen):
        """
        Build LSTM model and  to configure the learning process, which is done
        via the compile method.
        
        Parameters
        ----------
        max_features: int 
             The number of rows of the embedding matrix
        
        maxlen: int 
            Maximum length of all sequences.
        
        Returns
        -------
        model: The Keras Sequential model is a linear stack of layers.
        
        References
        ----------
        .. [1] `Getting started with the Keras Sequential model
               <https://keras.io/getting-started/sequential-model-guide/>`_
        """
        model = Sequential()
        model.add(Embedding(max_features, 128, input_length=maxlen))
        model.add(LSTM(128))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop')
        return model
    
    
    def load_data(self,path): 
        """
        Collects and cleans the data
        """
        df = pd.read_csv(path, names=['Domain','Origin','Class'], sep='\x01')        
        df = df.replace('legit',0)
        df = df.replace('LEGIT',0)
        df = df.replace('lgit',0) 
        df = df.replace('lefit',0) 
        df = df.replace('legip',0)
        df = df.replace('leNit',0)
        df = df.replace('dga',1)
        df = df.replace('DGA',1)
        df = df.replace('dg',1)
        df = df.replace('da',1)
        df = df.replace('dgb',1)
        df = df.replace('dgf',1)
        df = df.replace('dgS',1) 
        df = df.replace('dha',1) 
        
        df = df.drop(['Origin'], axis = 1)

        # drop rows where the class is null 
        nan_rows = df[df['Class'].isnull()].index
        i = list(nan_rows)
        df = df.drop(df.index[i])
        df = df.reset_index()
        
        df = topLevel(df)
        
        self.df = df 
    
    
    def train_model(self,max_epoch, nfolds, batch_size): 
        """
        This function builds the models based on the classifier and labels.
        
        Parameters
        ----------
        max_epoch: int
            One Epoch is when an ENTIRE dataset is passed forward and backward 
            through the neural network only ONCE
        
        nfolds: int 
            The number of batches needed to complete one epoch.
        
        batch_size: int 
            Total number of training examples present in a single batch.
        
        Returns
        -------
            A trained Keras model 
        
        References
        ----------
        .. [1] Using Deep Learning To Detect DGAs
           <https://www.endgame.com/blog/technical-blog/using-deep-learning-detect-dgas>
           <https://github.com/endgameinc/dga_predict> 
        """
        
        path = "./dga-dataset.txt"
        self.load_data(path)
    
        # Extract data and labels
        X = list(self.df['Domain'])
        labels = list(self.df['Class'])
    
        # Generate a dictionary of valid characters
        valid_chars = validChars(X)
    
        max_features = len(valid_chars) + 1
        maxlen = np.max([len(x) for x in X])
    
        # Convert characters to int and pad
        X = [[valid_chars[y] for y in x] for x in X]
        X = sequence.pad_sequences(X, maxlen=maxlen)
    
        # Convert labels to 0-1
        y = list(self.df['Class'])
      
        for fold in range(nfolds):
            print("fold %u/%u" % (fold+1, nfolds))
            X_train, X_test, y_train, y_test, _, label_test = train_test_split(X, y, labels, 
                                                                               test_size=0.2)
    
            print('Build model...')
            model = self.build_model(max_features, maxlen)
    
            print("Train...")
            X_train, X_holdout, y_train, y_holdout = train_test_split(X_train, 
                                                                      y_train, test_size=0.2)
            best_iter = -1
            best_auc = 0.0
    
            for ep in range(max_epoch):
                model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1)
    
                t_probs = model.predict_proba(X_holdout)
                t_auc = roc_auc_score(y_holdout, t_probs)
                
                
                print('Epoch %d: auc = %f (best=%f)' % (ep, t_auc, best_auc))
    
                if t_auc > best_auc:
                    best_auc = t_auc
                    best_iter = ep
                                        
                    probs = model.predict_proba(X_test)
        
                    print(sklearn.metrics.confusion_matrix(y_test, probs > .5))
                    
                    print("Saving the Better Model")
                    print("t_auc = " + str(t_auc))
                    print("best_auc = " + str(best_auc))
                    self.model = model 
                    
                else:
                    # No longer improving...break and calc statistics
                    if (ep-best_iter) > 2:
                        break
    
 
    def save_model(self): 
        """
        Saves all necessary model state information for classification work to disk.
        :return: True if it succeeded and False otherwise.
        """
        self.model.model.save('saved_model.h5')
        #TODO implement the return and error handing funcitonality
        
        
    def load_model(self): 
        """
        This function attempts to load the model from h5 file
        :return: True for success, False for failure 
        """
        self.model = load_model('saved_model.h5')
        #TODO implement the return and error handing funcitonality 
        
    
    def delete_model(self, modelRebuild=False, exclude=None, labeled_df=None): 
        """
        Initiates the machine learning models used in order to begin making predictions 
        """
        self.model = None
        #TODO implement the return and error handing funcitonality
        
        
    def evaluate_model(self): 
        """
        Returns classifcation metrics, tested over most of the dataset   
        """
        if self.df is not None: 
            X = list(self.df['Domain'])
            y = list(self.df['Class'])
      
        else: 
            path = "./dga-dataset.txt"
            self.load_data(path)
            X = list(self.df['Domain'])
            y = list(self.df['Class'])

        valid_chars = validChars(X)
                
        maxlen = np.max([len(x) for x in X])
        
        # Convert characters to int and pad
        X = [[valid_chars[y] for y in x] for x in X]
        X = sequence.pad_sequences(X, maxlen=maxlen)
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 
                                                            0.90, random_state = 0)
        
        ypred = self.model.predict_classes(X_test)
        
        accuracy = accuracy_score(y_test,ypred)
        recall = recall_score(y_test,ypred)
        precision = precision_score(y_test,ypred)
        f1 = f1_score(y_test,ypred)
        roc = roc_auc_score(y_test,ypred)
                
        return {'accuracy':accuracy,'precision': precision, 
                'recall': recall, 'f1':f1, 'roc':roc} 
        
        
    def predict(self,y):
        """
        Given a url input determine if the url is legitimate or a DGA generated
        domain
        
        Parameters
        ----------
        y: str 
            The targert url on which the prediction will be made 
            
        Retrurns
        --------
        dict: The predicted class of input url 
        
        """
        if self.df is not None: 
            X = list(self.df['Domain'])
        else: 
            path = "./dga-dataset.txt"
            self.load_data(path)
            X = list(self.df['Domain'])
       
        valid_chars = validChars(X)
        maxlen = np.max([len(x) for x in X])
        
        # Set up Input 
        y = [tldextract.extract(y).domain]
        
        # Convert characters to int and pad
        y = [[valid_chars[i] for i in x] for x in y]
        y = sequence.pad_sequences(y, maxlen=maxlen)
        
        ypred = self.model.predict_classes(y)
        ypred = ypred.item(0)
        
        return {'class': ypred}
    
    
    def dgab_prediction_to_json(self,prediction): 
        """
        Given a prediction DataFrame obtained from calling mmb_predict() convert 
        primary fields into a dict that can be easily converted to a 
        search-friendly json representation for a technology like a No-SQL 
        database or technology like Elasticsearch.
        :param prediction: 
        :return: a dictionary of statistics and classification results for the sample
        """
        #TODO implement the return and error handing funcitonality
    
    
    
    
    
    