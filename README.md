![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
<p align='center'>
    <h1 align="center">NLU tasks</h1>
    <p align="center">
    Project for NLU at the University of Trento A.Y.2023/2024
    </p>
    <p align='center'>
    Developed by:<br>
    De Martini Davide <br>
    </p>   
</p>

----------

- [Project Description](#project-description)
- [Project structure](#project-structure)
- [Installation](#installation)
- [Running the project](#running-the-project)


## Project Description
This project aimed to introduce the main tasks of Natural Language Understanding field. 

1) Build a Neural Language Model from an LSTM and use some regularization techniques in order to get better results.
2) Intent classification and Slot Fillling
   1) At first with LSTM
   2) Then done with BERT
3) Use BERT for 'Aspect Based Sentiment Analysis', only the part of term extraction is done

In each folder is present a report that outlines better the tasks.
## Project structure
```
basic_nlu_tasks
├── LM
│   ├── part_1
│   │   ├── dataset
│   │   ├── functions.py
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── README.md
│   │   └── utils.py
│   └── part_2
├── NLU
│   ├── part_1
│   └── part_2
├── nlu_env.yaml
├── README.md
├── requirements.txt
└── SA
    └── part_1
```
- LM: Language models
- NLU: Slot filling and intent recognition
- SA: Aspect extraction for Sentiment Analyisis

## Installation

In order to run the project you'll need to clone it and install the requirements. We suggest you to create a virtual environment 
- Clone it

    ```BASH
    git clone https://github.com/davidedema/basic_nlu_tasks

    ```
- Create the env, in this case with conda but venv could be also used:

    ```bash
    conda env create -f nlu_env.yaml -n nlu24
    conda activate nlu24
    ``` 

## Running the project
In order to run the examples enter in the folder of the task and run the `main.py` file
