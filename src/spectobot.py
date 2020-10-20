import time
from datetime import date
import os
import yaml

import tensorflow as tf
import numpy as np
import pandas as pd

from data_handling import ImageGenerator

from telepot.loop import MessageLoop
from telepot.delegate import (per_chat_id,
                              create_open, 
                              pave_event_space, 
                              per_callback_query_chat_id,
                              include_callback_query_chat_id)
from telepot.namedtuple import ReplyKeyboardMarkup
import telepot

# reading in parameters and hyperparameters
with open('spectobot.yaml') as file:
    yaml_data = yaml.safe_load(file)

TOKEN = yaml_data['TOKEN']
MODEL_LOCATION = yaml_data['MODEL_LOCATION']
STOP_PROGRAM = False
CLASS_NAMES = ['afr', 'eng', 'nbl', 'nso', 'ven', 'sot', 'ssw', 'tso', 'tsn', 'xho', 'zul']
CONVERTER_DICT = {"afr":'Afrikaans',
                  "eng":'English',
                  "tso":'Tsonga',
                  "nbl":'Ndebele',
                  "ven":'Tshivenda',
                  "tsn":'Setswana',
                  "sot":'Sesotho',
                  "nso":'Sepedi',
                  "xho":'Xhosa',
                  "zul":'Zulu',
                  "ssw":'Siswati'}
INVERTER_DICT = {'Afrikaans':'afr',
                 'English':'eng',
                 'Tsonga':'tso',
                 'Ndebele':'nbl',
                 'Tshivenda':'ven',
                 'Setswana':'tsn',
                 'Sesotho':'sot',
                 'Sepedi':'nso',
                 'Xhosa':'xho',
                 'Zulu':'zul',
                 'Siswati':'ssw'}

class SpectoNet(object):

    def __init__(self, model_location):
        self.model_name = model_location.split('/')[-1]
        self.model = tf.keras.models.load_model(model_location) 

    def predict_from_file(self, file_location):
        img = tf.keras.preprocessing.image.load_img(file_location, 
                                                 target_size=(64, 64))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        return CLASS_NAMES[np.argmax(score)], 100 * np.max(score)

    def update_performance_file(self, predicted_language, true_language, prob, performance_location):
        
        try:
            performance = pd.read_csv(performance_location)
        except:
            performance = pd.DataFrame()
        
        _ = pd.DataFrame({"PRED_LANG":[predicted_language],
                         "TRUE_LANG":[true_language],
                         "CONFIDENCE":[prob],
                         "MODEL_NAME":[self.model_name],
                         "DATE":[date.today()]})

        performance = pd.concat([performance, _])
        performance.to_csv(performance_location, index=False)

class SpectoBot(telepot.helper.ChatHandler):
    def __init__(self, *args, **kwargs):
        super(SpectoBot, self).__init__(*args, **kwargs)
        self.save_location = './spectonet_telegram_peformance.csv'
        self.predicted_language =  ''
        self.predicted_language = ''
        self.prob = 0
        self.ANSWERED_QUESTION = False

    def on_chat_message(self, message):
    
        global STOP_PROGRAM
        content_type, _, chat_id = telepot.glance(message)

        #TODO: ADD WAY OF GATHERTING THE META DATA BELOW

        if content_type == 'text':
            received_text = message['text']  

            if received_text == 'Stop, from Ruan': # checking if it program should shut down
                STOP_PROGRAM = True
            elif received_text == '/start':
                bot.sendMessage(chat_id, 'Hallo! I am SpectoBot, an AI thats sole purpose is to predict what language you are speaking in a voice note.')
                bot.sendMessage(chat_id, 'I am still being tested and would love your help! But is important to know that if you send me a voice note it will be saved anonymously in order for me to predict as well as used for further training.')
                bot.sendMessage(chat_id, 'If you are happy with that, just send me a voice note saying anything you want in one of the South African languages. I am much more accuracte when the voice note is longer or equal to three seconds :) ')
            elif 'Yes I spoke ' in received_text:
                if self.ANSWERED_QUESTION:
                    markup = ReplyKeyboardMarkup(keyboard=[])
                    bot.sendMessage(chat_id, 'Wohooo!! If you want me to make another prediction just send another voice note :)', reply_markup=markup)
                    spectonet.update_performance_file(self.predicted_language, self.predicted_language, round(self.prob,2),self.save_location )
                    self.ANSWERED_QUESTION = False
                else:
                    bot.sendMessage(chat_id, 'If you want me to predict your language, send a voice note saying anything in a South African language')
            elif received_text=='No you are not':
                if self.ANSWERED_QUESTION:
                    markup = ReplyKeyboardMarkup(keyboard=[[CONVERTER_DICT[lang]] for lang in CLASS_NAMES])
                    bot.sendMessage(chat_id, 'Could you please indicate what language you spoke', reply_markup=markup)
                else:
                    bot.sendMessage(chat_id, 'If you want me to predict your language, send a voice note saying anything in a South African language')
            elif received_text in [CONVERTER_DICT[lang] for lang in CLASS_NAMES]:
                if self.ANSWERED_QUESTION:
                    spectonet.update_performance_file(self.predicted_language, INVERTER_DICT[received_text],
                                                    round(self.prob,2),
                                                    self.save_location)
                    markup = ReplyKeyboardMarkup(keyboard=[])
                    bot.sendMessage(chat_id, 'If you want me to predict your language again, send a voice note saying anything in a South African language', reply_markup=markup)
                    self.ANSWERED_QUESTION = False
                else:
                    bot.sendMessage(chat_id, 'If you want me to predict your language, send a voice note saying anything in a South African language')
            else:
                bot.sendMessage(chat_id, 'If you want me to predict your language, send a voice note saying anything in a South African language')
        
        if content_type == 'voice':
            file_name = message['voice']['file_id']
            file_save_location = os.path.join('./', f'{file_name}.wav')
            image_save_location = os.path.join('./', f'{file_name}.png')

            bot.download_file(file_name, file_save_location)

            image_generator._generate_spectogram_from_audio_file(file_location=file_save_location,
                                                                save_location=image_save_location,
                                                                image_h = 64,
                                                                image_w = 64,
                                                                n_mels=128,
                                                                fmax=8000, 
                                                                dpi=250)

            self.predicted_language, self.prob = spectonet.predict_from_file(image_save_location)

            bot.sendMessage(chat_id,  f'I am {round(self.prob,2)}% sure you spoke in {CONVERTER_DICT[self.predicted_language]}') 
            markup = ReplyKeyboardMarkup(keyboard=[[f'Yes I spoke {CONVERTER_DICT[self.predicted_language]}'],
                                                ['No you are not']])
            bot.sendMessage(chat_id, 'Am I correct?', reply_markup=markup)
            self.ANSWERED_QUESTION = True
        

if __name__ == '__main__':

    spectonet = SpectoNet(MODEL_LOCATION)
    image_generator = ImageGenerator()  
    bot = telepot.DelegatorBot(TOKEN, 
                              [include_callback_query_chat_id(pave_event_space())(per_chat_id(), create_open, SpectoBot, timeout=500)])
    MessageLoop(bot).run_as_thread()

    answerer = telepot.helper.Answerer(bot)
    print('Listening ...')

    # This might be needed 🤷‍
    while True:
        if STOP_PROGRAM:
            break
        time.sleep(2)