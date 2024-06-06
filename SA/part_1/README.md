# Informaton about the weights
Since the classes to identify where 3, I didn't used the class Lang. Instead the slot2id thing is like:
- 0 = PAD
- 1 = O
- 2 = T

# Informaton about the dataset path
In order to set correctly the dataset path you have to change the variable `DATASET_PATH` in `main.py` with the appropriate path

# Information for testing
In order to test the model with the pretrained weights:
- set at True the `TEST` flag
- select the weights with the `WEIGHTS` variable
- run `main.py`