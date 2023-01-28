import random       # for shuffling responses
import json         # for reading the training file
import pickle       # for sterilisation
import numpy as np  # array
import nltk         # natural language processign
from nltk.stem import WordNetLemmatizer # reducing words to its stem
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout   # tweaked ones are working


nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()        # initating lemmatizer

intents = json.loads(open('intents.json').read())  # loading the file, making  the training file a dictionary in python


all_words = []
classes = []
docs = []
ignore_keys = [',', ',', '?', '.','']   # ignoring all these keys

for intent in intents['intents']:            # dictonaries in python, looping trough all sub values within the file
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        all_words.extend(word_list)           # tokenising, splitting inputs into words and adding it all to the words list
        docs.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# print(docs)

all_words = [lemmatizer.lemmatize(word) for word in all_words if word not in ignore_keys]
all_words = sorted(set(all_words))                  # later added! eliminating the duplicates and turning it back into a list
# print(words)

classes = sorted(set(classes))

# so far we have words, but need to turn it into numerical values that we can feed the neural network with
pickle.dump(all_words, open('words.pkl', 'wb'))  # wb means writing into binariries
pickle.dump(classes, open('classes.pkl', 'wb'))
######################################################### Machine learning part!!! ######################################################

                            # giving 1 values for words depending on occourance and zeros for the ones that doesnt
train = []
empty_output = [0] * len(classes)

for d in docs:
    bag = []
    word_patterns = d[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]   #capital letter error handling
    for word in all_words:                                                # checking each words if they occour in the pattern
        bag.append(1) if word in word_patterns else bag.append(0)     # if it does giving it a value of one othervise zero

    output = list(empty_output)
    output[classes.index(d[1])] = 1
    train.append([bag, output])       # adding evrything in the docu ment into the train list that will be used to train the neural network

random.shuffle(train)      #shuffle to get a random output from the resoponses

max_len = max([len(x) for x in train])      # later added
train = [x + [0] * (max_len - len(x)) for x in train]

train = np.array(train, dtype=object)

training_x = list(train[:, 0])
training_y = list(train[:, 1])

##################################### neural network model'############################# 4 linear stack layers.
model = Sequential()
model.add(Dense(128, input_shape=(len(training_x[0]),), activation='relu')) # input layer
model.add(Dropout(0.5))                                     # avoid overfitting:  The second layer is a dropout layer with a rate of 0.5, which will randomly drop out 50% of the units in the previous layer during training to avoid overfitting.
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(training_y[0]), activation='softmax'))    # matching the neurons with the number of classes!

sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#sgd = SGD(lr=0.01, decay=1e-6, momemtum=0.9, nesterov=True) # not working
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])     #

model.fit(np.array(training_x), np.array(training_y), epochs=200, batch_size=5, verbose=1)
model.save('Chatbot_model.model')
print("Training Finished... run chatbot now ->")