#import random
import json
import pickle #for sterilisation
import numpy as np
import random
import nltk
#lemmatizer is used in stemming words
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
docs = []
ignore = [',',',','?','.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize is used in splitting words
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        docs.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore]
words = sorted(set(words))


classes = sorted(set(classes))

#save to file using pickle to dump
pickle.dump(words, open ('words.pkl', 'wb'))#wb means writing into binariries
pickle.dump(classes, open ('classes.pkl', 'wb'))

#represents the words as numerical values using bag_of_words
train = []
empty_output = [0] * len(classes)

for doc in docs:
    #create an empty list of bag of words
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns  else bag.append(0)

    output = list(empty_output)
    output[classes.index(doc[1])] = 1
    train.append([bag, output])

#shuffling train data
random.shuffle(train)

train = np.array(train)

training_x = list(train[:, 0])
training_y = list(train[:, 1])

#building the neural network model
model = Sequential()
#adding layers to the model
model.add(Dense(128, input_shape=(len(training_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(training_y[0]), activation = 'softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momemtum = 0.9,nesterov = True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

model.fit(np.array(training_x), np.array(training_y), epochs=200, batch_size = 5, verbose = 1)
model.save('Chatbot_model.model')
print("Finished...")