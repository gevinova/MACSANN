# MACSANN

This is a python implementation of Memetic algorithm with crossover to search architecture of neural network (MACSANN). The objective of the algorithm is to find an architecture of an ANN that could solve a given problem.

# How to use

- Configure your config file, you can see an example in ("config-example.ini")
    - Here you set the hyperparameters of your experiment:
- Create the population
    ```sh
    import macsann
    population = macsann.population('config-example.ini')
    ```
- Set train and validation data
    ```
            #Train data
            population.train_input = boston['data'][:300]
            population.train_output = boston['target'][:300]
            
            #Validation data
            population.eval_input = boston['data'][300:]
            population.eval_output = boston['target'][300:]
    ```
- Run your experiment and save the best population
    ```
    best_pop = population.run()
    with open('best_pop.macsann','wb') as f:
        pickle.dump(best_pop, f)
    ```
 
- See the complete example in 'boston-example.py'

