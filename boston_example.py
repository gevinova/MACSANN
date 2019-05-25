# -*- coding: utf-8 -*-

import macsann
from sklearn.datasets import load_boston
import numpy as np
import pickle


if __name__ == '__main__':
    boston = load_boston()
    
    #Normalize data
    boston['target'] /= np.max(boston['target'])
    boston['data'] = np.divide(boston['data'], np.max(boston['data'], axis=0))
    
    #Create macsann population
    population = macsann.population('config-example.ini')
    
    #Train data
    population.train_input = boston['data'][:300]
    population.train_output = boston['target'][:300]
    
    #Validation data
    population.eval_input = boston['data'][300:]
    population.eval_output = boston['target'][300:]
    
    #Run and get the best population
    best_pop = population.run()
    with open('best_pop.macsann','wb') as f:
        pickle.dump(best_pop, f)