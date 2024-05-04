from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

from pprint import pprint
from utils import *

DATASET_PATH = '/home/davide/Desktop/nlu_exam/248445_davide_de_martini/NLU/part_2'

tmp_train_raw = load_data(os.path.join(DATASET_PATH,'dataset','train.json'))
test_raw = load_data(os.path.join(DATASET_PATH,'dataset','test.json'))

portion = 0.10

intents = [x['intent'] for x in tmp_train_raw] # We stratify on intents
count_y = Counter(intents)

labels = []
inputs = []
mini_train = []

for id_y, y in enumerate(intents):
    if count_y[y] > 1: # If some intents occurs only once, we put them in training
        inputs.append(tmp_train_raw[id_y])
        labels.append(y)
    else:
        mini_train.append(tmp_train_raw[id_y])
# Random Stratify
X_train, X_dev, y_train, y_dev = train_test_split(inputs, labels, test_size=portion, 
                                                    random_state=42, 
                                                    shuffle=True,
                                                    stratify=labels)
X_train.extend(mini_train)
train_raw = X_train
dev_raw = X_dev

y_test = [x['intent'] for x in test_raw]

words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute the cutoff

corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, 
                                        # however this depends on the research purpose
slots = set(sum([line['slots'].split() for line in corpus],[]))
intents = set([line['intent'] for line in corpus])

lang = Lang(words, intents, slots, cutoff=0)

train_dataset = IntentsAndSlots(train_raw, lang)


