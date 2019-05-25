# -*- coding: utf-8 -*-
import random
import numpy as np
from .ind import ind
from multiprocessing import Pool
from time import time
from pickle import load, dump
import configparser
import os
import tensorflow as tf
#Disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

'''TODOs
Asserts for
train and eval data
config data
min or max metric
Show topology of all layers'''

class population():

    def __init__(self, config_path):
        self.config = configparser.ConfigParser()
        self.config.read(config_path)

        self.LAYERS = int(self.config['MACSANN']['LAYERS'])
        self.ACTIVATION_FUNC = self.config['MACSANN']['ACTIVATION_FUNC']
        self.OPTIMIZER = self.config['MACSANN']['OPTIMIZER']
        self.NUM_IND = int(self.config['MACSANN']['NUM_IND'])
        self.EPOCHS = int(self.config['MACSANN']['EPOCHS'])
        self.GENERATIONS = int(self.config['MACSANN']['GENERATIONS'])
        self.LAST_ACTIVATION = self.config['MACSANN']['LAST_ACTIVATION']
        self.METRIC_FUNC = self.config['MACSANN']['METRIC_FUNC']
        self.LOSS_FUNC = self.config['MACSANN']['LOSS_FUNC']
        self.OUTPUT_NEURONS = int(self.config['MACSANN']['OUTPUT_NEURONS'])
        self.MAX_HIDDEN_NEURONS = int(self.config['MACSANN']['MAX_HIDDEN_NEURONS'])
        self.DROPOUT_RATE = float(self.config['MACSANN']['DROPOUT_RATE'])

        self.train_input = None
        self.train_output = None
        self.eval_input = None
        self.eval_output = None

    def print_stats(self, mtr, nc, bf):
        print("Metric")
        mtr = np.sort(mtr)
        print("Average :", str(np.mean(mtr)), " std: ", str(np.std(mtr)))
        print("Best: ", str(mtr[0]), ' Worst: ', str(mtr[-1]))
        print("Loss")
        bf = np.sort(bf)
        print("Average :", str(np.mean(bf)), " std: ", str(np.std(bf)))
        print("Best: ", str(bf[0]), ' Worst: ', str(bf[-1]))
        print("Topology first layer")
        nc = np.sort(nc)
        print("Average :", str(np.mean(nc)), " std: ", str(np.std(nc)))
        print("lowest: ", str(nc[0]), " highest: ", str(nc[-1]))

    def eval_parallel(self, indi):
        if indi.fitness == None:
            if indi.weights == None:
                modelo = indi.build_model()
            else:
                modelo = indi.build_weights()
            if self.EPOCHS > 0:
                modelo.fit(self.train_input, self.train_output, epochs=self.EPOCHS,
                               verbose=0)
            indi.weights = modelo.get_weights()
            indi.fitness, indi.metric = modelo.evaluate(
                self.eval_input, self.eval_output, verbose=0)
        return indi

    def run(self):
        RANDOM_FILE = 'saved_'+str(random.random())[2:6]+'.pck'
        ind_id = 0

        '''Create individuals'''
        pop = []
        print('Creating population')
        for i in range(self.NUM_IND):
            ind_id += 1
            pop.append(ind(self.LAYERS, self.ACTIVATION_FUNC, self.OPTIMIZER,
                           self.train_input.shape[1:], self.LAST_ACTIVATION, self.METRIC_FUNC, self.LOSS_FUNC,
                           self.OUTPUT_NEURONS, self.MAX_HIDDEN_NEURONS, self.DROPOUT_RATE, ind_id))

        print("")
        for g in range(self.GENERATIONS):
            print(" ")
            print("Evaluating population")
            start = time()
            with Pool(processes=4) as mp:
                pop = mp.map(self.eval_parallel, pop)
                mp.terminate()

            duration = time() - start
            print('Time: ', duration)

            fitnesses = [i.fitness for i in pop]
            # quita a los malos y deja los buenos despues de gen0
            if g > 0:
                index_fitness = np.argsort(fitnesses)[:self.NUM_IND]
                #pop = [pop[i] for i in index_fitness]
                pop[random.randint(self.NUM_IND, self.NUM_IND*2 -1)] = pop[index_fitness[0]]
                pop = pop[self.NUM_IND:]
            #Save stats (beta)
            '''
                with open(RANDOM_FILE, 'rb') as fr:
                    _, sfitnesses, smetrics, sduration = load(
                        fr)
                with open(RANDOM_FILE, 'wb') as fp:
                    sfitnesses.append(fitnesses)
                    smetrics.append(metrics)
                    sduration.append(duration)
                    dump([pop, sfitnesses, smetrics, sduration], fp)
            elif g == 0:
                fitnesses = [i.fitness for i in pop]
                metrics = [i.metric for i in pop]
                with open(RANDOM_FILE, 'wb') as fp:
                    dump([pop, [fitnesses], [metrics], [duration]], fp)'''
                    
            #STOP CONDITIONS
            if (g == (self.GENERATIONS - 1)):
                return pop
            
            #Metrics
            neuron_count = []
            for indi in pop:
                neuron_count.append(indi.weights[0].shape[1])
            fitnesses = [i.fitness for i in pop]
            metrics = [i.metric for i in pop]
            self.print_stats(metrics, neuron_count, fitnesses)

            # 1 Choose parents
            print("Reproduction generation: "+str(g+1))
            new_gen = []
            for _ in range(int(self.NUM_IND/2)):
                wpadres = []
                for _ in range(2):

                    a = random.choice(pop)
                    b = random.choice(pop)
                    r = random.random()
                    while a.ind_id == b.ind_id:
                        b = random.choice(pop)
                    if a.fitness < b.fitness and r < 0.9:
                        wpadres.append(a.weights.copy())
                    else:
                        wpadres.append(b.weights.copy())

                # 2 extend neurons and weights

                for i in range(0, len(wpadres[0]), 2):
                    # neuronas de capa i padre 0
                    neuronas0 = wpadres[0][i].shape[1]
                    # neuronas de capa i padre 1
                    neuronas1 = wpadres[1][i].shape[1]

                    if neuronas0 > neuronas1:
                        neuronas_faltantes = neuronas0-neuronas1
                        wpadres[1][i] = np.hstack((wpadres[1][i], np.zeros(
                            (wpadres[1][i].shape[0], neuronas_faltantes))))
                    elif neuronas1 > neuronas0:
                        neuronas_faltantes = neuronas1-neuronas0
                        wpadres[0][i] = np.hstack((wpadres[0][i], np.zeros(
                            (wpadres[0][i].shape[0], neuronas_faltantes))))

                    if i > 1:
                        # neuronas de capa i padre 0
                        pesos0 = wpadres[0][i].shape[0]
                        # neuronas de capa i padre 1
                        pesos1 = wpadres[1][i].shape[0]
                        if pesos0 > pesos1:
                            pesos_faltantes = pesos0-pesos1
                            wpadres[1][i] = np.vstack((wpadres[1][i], np.zeros(
                                (pesos_faltantes, wpadres[1][i].shape[1]))))
                        elif pesos1 > pesos0:
                            pesos_faltantes = pesos1-pesos0
                            wpadres[0][i] = np.vstack((wpadres[0][i], np.zeros(
                                (pesos_faltantes, wpadres[0][i].shape[1]))))
                            
                # extend bias
                for i in range(1, len(wpadres[0]), 2):
                    # neuronas de capa i padre 0
                    neuronas0 = wpadres[0][i].shape[0]
                    # neuronas de capa i padre 1
                    neuronas1 = wpadres[1][i].shape[0]

                    if neuronas0 > neuronas1:
                        new_layer = np.zeros((neuronas0, ))
                        new_layer[:neuronas1] = wpadres[1][i]
                        wpadres[1][i] = new_layer
                    elif neuronas1 > neuronas0:
                        new_layer = np.zeros((neuronas1, ))
                        new_layer[:neuronas0] = wpadres[0][i]
                        wpadres[0][i] = new_layer

                # 3 choose single point cross
                wson = wpadres[0].copy()
                wdau = wpadres[1].copy()
                for i in range(0, len(wpadres[0])-1, 2):
                    # punto de cruce
                    r = random.randint(1, max(1, wpadres[0][i].shape[1]-1))
                    '''hacer el cruce'''
                    wson[i][:, r:] = wpadres[1][i][:, r:]
                    wdau[i][:, r:] = wpadres[0][i][:, r:]

                    wson[i+1][r:] = wpadres[1][i+1][r:]
                    wdau[i+1][r:] = wpadres[0][i+1][r:]

                for i in range(0, len(wpadres[0]), 2):
                    # Revisamos columnas de ceros
                    a = wson[i].sum(axis=0)
                    # Quitamos las columnas con cero
                    wson[i] = wson[i][:, a != 0]
                    if i < len(wpadres[0])-2:
                        wson[i+1] = wson[i+1][a != 0]
                        # Quitamos filas de la siguiente capa en cero
                        wson[i+2] = wson[i+2][a != 0]

                    # Revisamos columnas de ceros
                    a = wdau[i].sum(axis=0)
                    # Quitamos las columnas con cero
                    wdau[i] = wdau[i][:, a != 0]
                    if i < len(wpadres[0])-2:
                        wdau[i+1] = wdau[i+1][a != 0]
                        # Quitamos filas de la siguiente capa en cero
                        wdau[i+2] = wdau[i+2][a != 0]

                r_mutate = random.random()
                if r_mutate < 0.05:
                    w_mutate = random.randint(0, 2)
                    if w_mutate == 0:
                        #Delete neuron
                        index_layer = random.randint(0, (self.LAYERS-1))*2
                        index_neuro = random.randint(
                            0, wson[index_layer].shape[1]-1)
                        wson[index_layer] = np.delete(
                            wson[index_layer], index_neuro, 1)
                        wson[index_layer+2] = np.delete(wson[index_layer+2],
                                                        index_neuro, 0)
                        wson[index_layer +
                             1] = np.delete(wson[index_layer+1], index_neuro)
                    elif w_mutate == 1:
                        # Add neuron
                        index_layer = random.randint(0, (self.LAYERS-1))*2
                        b = np.random.uniform(-1, 1,
                                              wson[index_layer].shape[0])
                        wson[index_layer] = np.insert(
                            wson[index_layer], wson[index_layer].shape[1], b, axis=1)
                        b = np.random.uniform(-1, 1,
                                              wson[index_layer + 2].shape[1])
                        wson[index_layer + 2] = np.insert(
                            wson[index_layer + 2], wson[index_layer+2].shape[0], b, axis=0)
                        wson[index_layer + 1] = np.insert(
                            wson[index_layer + 1], wson[index_layer+1].shape[0], np.random.uniform(-1, 1, 1))

                ind_id += 1
                new_gen.append(ind(self.LAYERS, self.ACTIVATION_FUNC, self.OPTIMIZER,
                           self.train_input.shape[1:], self.LAST_ACTIVATION, self.METRIC_FUNC, self.LOSS_FUNC,
                           self.OUTPUT_NEURONS, self.MAX_HIDDEN_NEURONS, self.DROPOUT_RATE, ind_id, weights=wson))

                ind_id += 1
                new_gen.append(ind(self.LAYERS, self.ACTIVATION_FUNC, self.OPTIMIZER,
                           self.train_input.shape[1:], self.LAST_ACTIVATION, self.METRIC_FUNC, self.LOSS_FUNC,
                           self.OUTPUT_NEURONS, self.MAX_HIDDEN_NEURONS, self.DROPOUT_RATE, ind_id, weights=wdau))

            pop.extend(new_gen)
            new_gen = None
