#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:31:30 2020

@author: sshekhar
"""
from keras.models import Sequential
from keras.models import load_model

from collections import defaultdict

import pickle


def _merge_dict(dict_list):
    dd = defaultdict(list)    
    for d in dict_list:
        for key, value in d.items():
            if not hasattr(value, '__iter__'):
                value = (value,)
            [dd[key].append(v) for v in value]
    return dict(dd)

def save(obj, name):
    try:
        filename = open(name + ".pickle","wb")
        pickle.dump(obj, filename)
        filename.close()
        return(True)
    except:
        return(False)

def load(name):
    filename = open(name + ".pickle","rb")
    obj = pickle.load(filename)
    filename.close()    
    return(obj)

def load_model_w(name):
    model_k = load_model(name)
    history = load(name)
    model = Sequential_wrapper(model_k)
    model.history = history
    return(model)

class Sequential_wrapper():
    """
    %s
    """%Sequential.__doc__
    
    def __init__(self, model=Sequential()):
        self.history = {}
        self.model = model
        
        # method shortcuts
        methods = dir(self.model)
        for method in methods:
            if method.startswith('_'): continue
            if method in ['model','fit','save']: continue
            try:
                exec('self.%s = self.model.%s' % (method,method))
            except:
                pass
    
    def _update_history(self,history):
        if len(self.history)==0:
            self.history = history
        else:
            self.history = _merge_dict([self.history,history])
    
    def fit(self, x, y, batch_size=32, epochs=10, verbose=1, callbacks=None,
            validation_split=0.0, validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None,
            initial_epoch=0, **kwargs):
        """
        %s
        """%self.model.fit.__doc__
        h = self.model.fit(x, y, batch_size, epochs, verbose, callbacks,
                     validation_split, validation_data, shuffle,
                     class_weight, sample_weight,
                     initial_epoch, **kwargs)
        self._update_history(h.history)
        return h
    
    def save(self, filepath, overwrite=True):
        """
        %s
        """%self.model.save.__doc__
        save(self.history,filepath)
        self.model.save(filepath, overwrite)