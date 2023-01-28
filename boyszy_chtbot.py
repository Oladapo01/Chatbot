import random
import pickle
import numpy as np
import json
import speech_recognition
import pyttsx3
import nltk
from nltk.stem import WordNetLemmatizer
engine = pyttsx3.init() # initiatin the speech method

#from tensorflow.keras.models import load_model
from keras.models import load_model

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('Chatbot_model.model')


def clean_sentence(sentence):
    words_sentence = nltk.word_tokenize(sentence)
    words_sentence = [lemmatizer.lemmatize(word) for word in words_sentence]
    return words_sentence


def words_box(sentence, show_details=True):
    words_sentence = clean_sentence(sentence)
    box = [0] * len(words)
    for j in words_sentence:
        for i, word in enumerate(words):
            if word == j:
                box[i] = 1
                if show_details:
                    print("found in bag: %s" % word)
    return np.array(box)


def prediction(sentence):
    bd = words_box(sentence, show_details=False)
    res = model.predict(np.array([bd]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    get_back_list = []
    for r in results:
        get_back_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return get_back_list


def get_reply(intents_list, intents_json):
    tag = intents_list[0]['intent']
    all_intents = intents_json['intents']
    for i in all_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result


print("Chatbot started.......")
recognizer = speech_recognition.Recognizer()
while True:
    try:
        # Listen to audio input using speech recognition
        with speech_recognition.Microphone() as mic:
            #recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)
            message = recognizer.recognize_google(audio)
            message = message.lower()
            #print(f"Recognized {message}")
    except:
        recognizer = speech_recognition.Recognizer()
        continue

    # Predict the intent of the message using the model
    pred = prediction(message)

    # Get the response for the predicted intent
    res = get_reply(pred, intents)
    engine.say(res)
    engine.runAndWait()
    print(res)

    if message == 'close':
        break