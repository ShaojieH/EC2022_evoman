File using deap is `my_specialist_demo.py`.

Usage:
1. run the file with `run_mode = 'train'` in the code. Weight with the best fitness is output to my_specialist_demo/test.txt.
2. run again with `run_mode = 'test`.

Note: 
1. The file basic simplifies `optimization_specialist_demo.py` with deap, and uses the neural network defined in `demo_controller.py` with deap. 
2. Training time is pretty long. Using the original 100 population size and 30 generations, it runs for about 30min. Currently, it runs with 40 population size and 20 generations.