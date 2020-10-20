'''
Contains all of the code required to convert the data 
downloaded from NCHLT corpus into images that are 
found in relevant subfolders.
'''

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from more_itertools import chunked
import io

import tensorflow as tf
from sklearn.utils import shuffle

import librosa
import librosa.display
from librosa.util import pad_center as center

import xmltodict
from xml.parsers.expat import ExpatError
import json
from tqdm import tqdm

# detect language in text
from langdetect import DetectorFactory, detect, detect_langs
DetectorFactory.seed = 42

LANG_ABBREVIATIONS = ["afr", "eng", "tso", "nbl",
                      "ven", "tsn", "sot", "nso",
                      "xho", "zul", "ssw"]
LANG_FULL_TEXT = ["afrikaans", "english", "tsonga", "ndebele",
                  "tshivenda", "setswana", "sesotho", "sepedi",
                  "xhosa", "zulu", "siswati"]
SR = 16000
SAMPLES_PER_3S = SR*3
MINIMUM_SAMPLES = SR*1.02

def read_xml(self, file):
    with open(file) as f:
        data = f.read()
        try:
            doc = xmltodict.parse(data)
        except ExpatError:
            doc = xmltodict.parse(data[3:])

    return doc


class LwaziMetaDataCreator(object):

    def __init__(
            self,
            data_path=None):

        if data_path != None:
            self.data_path = data_path
        else:
            self.data_path = './'


    def _load_metadata_files(self, save_location):
        
        metadata_ = []
        
        for lang in LANG_ABBREVIATIONS:
            _ = pd.read_csv(f'{self.data_path}/{lang}/Lwazi_metadata_{lang}.csv')
            _['LABEL'] = lang
            metadata_.append(_)
            
        
        metadata = pd.concat(metadata_)
        
        metadata.to_csv(f'{save_location}/lwazi_metadata.csv', index=False)
        
    def get_audio_info(self, save_location='../../../data/lwazi'):

        def get_audio_files(row):
            if row['LABEL'] == 'afr':
                speaker = '%03d'%int(row['Speaker'])
                folder = f'{self.data_path}/{row["LABEL"]}/audio/afrikaans_{speaker}/'

                files = os.listdir(folder)

            elif row['LABEL'] == 'eng':
                speaker = '%03d'%int(row['Speaker'])
                folder = f'{self.data_path}/{row["LABEL"]}/audio/english_{speaker}/'

                files = os.listdir(folder)

            elif row['LABEL'] == 'tsn':
                speaker = '%03d'%int(row['Speaker'])
                folder = f'{self.data_path}/{row["LABEL"]}/audio/setswana_{speaker}/'

                files = os.listdir(folder)

            elif row['LABEL'] == 'nso':
                speaker = '%03d'%int(row['Speaker'])
                folder = f'{self.data_path}/{row["LABEL"]}/audio/sepedi_{speaker}/'

                files = os.listdir(folder)

            elif row['LABEL'] == 'xho':
                speaker = '%03d'%int(row['Speaker'])
                folder = f'{self.data_path}/{row["LABEL"]}/audio/isixhosa_{speaker}/'

                files = os.listdir(folder)

            elif row['LABEL'] == 'zul':
                speaker = '%03d'%int(row['Speaker'])
                folder = f'{self.data_path}/{row["LABEL"]}/audio/isizulu_{speaker}/'

                files = os.listdir(folder)

            durations = [librosa.get_duration(filename=f'{folder}{file}') for file in files]

            return pd.DataFrame({'Speaker':[row['Speaker']]*len(files),
                                 'Gender':[row['Gender']]*len(files),
                                 'Line':[row['Line']]*len(files),
                                 'Age':[row['Age']]*len(files),
                                 'LABEL':[row['LABEL']]*len(files),
                                 'FILE':[f'{folder}{file}' for file in files],
                                 'DURATION':durations})

        df = pd.read_csv(f'{save_location}/lwazi_metadata.csv')

        dfs = []

        for i, r in tqdm(df.iterrows()):
            dfs.append(get_audio_files(r))

        df = pd.concat(dfs)

        def read_transcript(file):
            with open(file.replace('audio', 'transcriptions').replace('.wav', '.txt')) as f:
                text = f.read().replace('[n] ', '').strip()        
            return text

        df['TRANSCRIPT'] = df.FILE.progress_apply(lambda x: read_transcript(x))

        df.to_csv(f'{save_location}/lwazi_metadata_with_audio.csv', index=False)

    def _clean_data(self,
                    file_location,
                    save_location,
                    min_seconds=3,
                    max_seconds=5,
                    lang_prob=0.9,):


        df = pd.read_csv(file_location)
        
        df = df[(df.DURATION>=min_seconds) & (df.DURATION<=max_seconds)]

        df.to_csv(os.path.join(save_location, 'lwazi_filtered_metadata.csv'), index=False)


    def create_final_metadata_file(self,
                                   save_location='./',
                                   min_seconds=3,
                                   max_seconds=7,
                                   lang_prob=0.9,):
        """
        Create final metadata CSV combining and cleaning all of the 
        data as well assigning roles and adding the image path where
        the spectrograms can be found.
        """
        
        self._load_metadata_files(save_location)
        self.get_audio_info(save_location)
        self._clean_data(file_location=os.path.join(save_location, 'lwazi_metadata_with_audio.csv'),
                         save_location=save_location)

        df = pd.read_csv(os.path.join(save_location, 'lwazi_filtered_metadata.csv'))

        def generate_image_path(row):
            audio_name = row['FILE'].split('/')[-1].split('.wav')[0] # extract audio name and remove wav
            line = row['Line']
            gender = row['Gender']
            file_name = f'{audio_name}_{line}_{gender}.png'
            language = row['LABEL']
            image_path = os.path.join(language, file_name).replace('\\', '/')#ensure window runs don't mess it up
            return  image_path
    
        df['IMAGE_PATH'] = df.apply(lambda row: generate_image_path(row), axis=1)

        df.to_csv(os.path.join(save_location, 'lwazi_final_metadata.csv'), index=False)


class NchltMetaDataCreator(object):

    def __init__(
            self,
            data_path=None):

        if data_path != None:
            self.data_path = data_path
        else:
            self.data_path = './'


                # convert all of the zip files to xml files

        print('Converting XML data to JSON')

        # reading in all of the aux data
        afr_xml = read_xml(f'{self.data_path}/afr-aux1/afr/info/nchltAux1_afr.xml')
        self.afr_json_aux = json.loads(json.dumps(afr_xml))

        eng_xml = read_xml(f'{self.data_path}/eng-aux1/eng/info/nchltAux1_eng.xml')
        self.eng_json_aux = json.loads(json.dumps(eng_xml))

        tso_xml = read_xml(f'{self.data_path}/tso-aux1/tso/info/nchltAux1_tso.xml')
        self.tso_json_aux = json.loads(json.dumps(tso_xml))

        nbl_xml = read_xml(f'{self.data_path}/nbl-aux1/nbl/info/nchltAux1_nbl.xml')
        self.nbl_json_aux = json.loads(json.dumps(nbl_xml))

        ven_xml = read_xml(f'{self.data_path}/ven-aux1/ven/info/nchltAux1_ven.xml')
        self.ven_json_aux = json.loads(json.dumps(ven_xml))

        tsn_xml = read_xml(f'{self.data_path}/tsn-aux1/tsn/info/nchltAux1_tsn.xml')
        self.tsn_json_aux = json.loads(json.dumps(tsn_xml))

        sot_xml = read_xml(f'{self.data_path}/sot-aux1/sot/info/nchltAux1_sot.xml')
        self.sot_json_aux = json.loads(json.dumps(sot_xml))

        nso_xml = read_xml(f'{self.data_path}/nso-aux1/nso/info/nchltAux1_nso.xml')
        self.nso_json_aux = json.loads(json.dumps(nso_xml))

        xho_xml = read_xml(f'{self.data_path}/xho-aux1/xho/info/nchltAux1_xho.xml')
        self.xho_json_aux = json.loads(json.dumps(xho_xml))

        zul_xml = read_xml(f'{self.data_path}/zul-aux1/zul/info/nchltAux1_zul.xml')
        self.zul_json_aux = json.loads(json.dumps(zul_xml))

        ssw_xml = read_xml(f'{self.data_path}/ssw-aux1/ssw/info/nchltAux1_ssw.xml')
        self.ssw_json_aux = json.loads(json.dumps(ssw_xml))

        # reading in all of the first release data
        afr_xml = read_xml(f'{self.data_path}/nchlt_afr/transcriptions/nchlt_afr.trn.xml')
        self.afr_json = json.loads(json.dumps(afr_xml))

        eng_xml = read_xml(f'{self.data_path}/nchlt_eng/transcriptions/nchlt_eng.trn.xml')
        self.eng_json = json.loads(json.dumps(eng_xml))

        tso_xml = read_xml(f'{self.data_path}/nchlt_tso/transcriptions/nchlt_tso.trn.xml')
        self.tso_json = json.loads(json.dumps(tso_xml))

        nbl_xml = read_xml(f'{self.data_path}/nchlt_nbl/transcriptions/nchlt_nbl.trn.xml')
        self.nbl_json = json.loads(json.dumps(nbl_xml))

        ven_xml = read_xml(f'{self.data_path}/nchlt_ven/transcriptions/nchlt_ven.trn.xml')
        self.ven_json = json.loads(json.dumps(ven_xml))

        tsn_xml = read_xml(f'{self.data_path}/nchlt_tsn/transcriptions/nchlt_tsn.trn.xml')
        self.tsn_json = json.loads(json.dumps(tsn_xml))

        sot_xml = read_xml(f'{self.data_path}/nchlt_sot/transcriptions/nchlt_sot.trn.xml')
        self.sot_json = json.loads(json.dumps(sot_xml))

        nso_xml = read_xml(f'{self.data_path}/nchlt_nso/transcriptions/nchlt_nso.trn.xml')
        self.nso_json = json.loads(json.dumps(nso_xml))

        xho_xml = read_xml(f'{self.data_path}/nchlt_xho/transcriptions/nchlt_xho.trn.xml')
        self.xho_json = json.loads(json.dumps(xho_xml))

        zul_xml = read_xml(f'{self.data_path}/nchlt_zul/transcriptions/nchlt_zul.trn.xml')
        self.zul_json = json.loads(json.dumps(zul_xml))

        ssw_xml = read_xml(f'{self.data_path}/nchlt_ssw/transcriptions/nchlt_ssw.trn.xml')
        self.ssw_json = json.loads(json.dumps(ssw_xml))


    def _extract_data_from_json(json_, lang, save_location, aux):

            self.df = pd.DataFrame()
            try:
                speakers = json_['corpus']['speaker']
                print(f'Total speakers: {len(speakers)}')
            except:
                print('Could not load speakers')
                speakers = []

            for j, speaker in tqdm(enumerate(speakers)):
                id_ = speaker["@id"]
                age = speaker['@age']
                gender = speaker['@gender']
                total_recordings = len(speaker['recording'])
                for i, recording in enumerate(speaker['recording']):
                    if aux:
                        file = recording['@audio'].split('nchltAux1/')[1]
                    else:
                        file = recording['@audio'].split(f'nchlt_{lang}/')[1]
                    duration = recording['@duration']
                    pdp_score = recording['@pdp_score']
                    transcript = recording['orth']
                    try:
                        lang_detect = detect_langs(transcript)[0]
                        lang_predicted = lang_detect.lang
                        lang_predict_prob = lang_detect.prob
                    except:
                        print(f'Lang detect failed')
                        lang_predicted = lang
                        lang_predict_prob = 0.5

                    self.df = self.df.append(pd.DataFrame({"ID": [id_],
                                                           "AGE": [age],
                                                           "GENDER": [gender],
                                                           "FILE": [file],
                                                           "DURATION": [duration],
                                                           "PDP_SCORE": [pdp_score],
                                                           "TRANSCRIPT": [transcript],
                                                           "LANGUAGE": [lang],
                                                           "DETECTED_LANGUAGE": lang_predicted,
                                                           "DETECTED_LANGUAGE_PROB": lang_predict_prob,
                                                           "AUXS": int(aux)}))

            print(f'Done creating meta data files for {lang}')
            print('*'*50)

            self.df.to_csv(os.path.join(save_location, f'{lang}_metadata.csv'), index=False)



    def _extract_meta_data(self, save_location='./', aux=False):
        """
        Extract all of the metadata contained in the 
        zip files for each langauge. The function will then
        save all of the metadata for each language in a 
        seperate csv file in the location specified by the user.

        Inputs:
            save_location (str): Location to save all of the CSV files.
                                 The default will be in the directory script is 
                                 run from.
        """

        

        lang_jsons = [self.afr_json,
                      self.eng_json,
                      self.tso_json,
                      self.nbl_json,
                      self.ven_json,
                      self.tsn_json,
                      self.sot_json,
                      self.nso_json,
                      self.xho_json,
                      self.zul_json,
                      self.ssw_json]

        lang_jsons_aux = [self.afr_json_aux,
                          self.eng_json_aux,
                          self.tso_json_aux,
                          self.nbl_json_aux,
                          self.ven_json_aux,
                          self.tsn_json_aux,
                          self.sot_json_aux,
                          self.nso_json_aux,
                          self.xho_json_aux,
                          self.zul_json_aux,
                          self.ssw_json_aux]

        for lang_json, lang in zip(lang_jsons, LANG_ABBREVIATIONS):
            _save_location = os.path.join(save_location, f'{lang}_metadata.csv')
            print(f'Busy getting and saving meta data for {lang} and saving at {_save_location}')
            print(f'This might take a while!')
            self._extract_data_from_json(lang_json, lang, os.path.join(save_location, 'csv_metadata_nchlt'), aux=True)

        for lang_json, lang in zip(lang_jsons, LANG_ABBREVIATIONS):
            _save_location = os.path.join(save_location, f'{lang}_metadata.csv')
            print(f'Busy getting and saving meta data for {lang} and saving at {_save_location}')
            print(f'This might take a while!')
            self._extract_data_from_json(lang_json, lang, os.path.join(save_location, 'csv_metadata_nchlt_aux'), aux=False)


    def _create_combined_metadata_csv(self, save_location):
        df = pd.DataFrame()
        for lang in LANG_ABBREVIATIONS:
            _ = pd.read_csv(os.path.join(save_location, 'csv_metadata_nchlt', f'{lang}_metadata.csv'))
            df = df.append(_)
            del _

        print(f"Size before droping duplicates: {len(df)}")
        df = df.drop_duplicates()
        print(f"Size after droping duplicates: {len(df)}")
        df.to_csv(os.path.join(save_location, 'csv_metadata_nchlt','all_data.csv'), index=False)

        # doing the excact same for aux data
        df = pd.DataFrame()
        for lang in LANG_ABBREVIATIONS:
            _ = pd.read_csv(os.path.join(save_location, 'csv_metadata_nchlt_aux', f'{lang}_metadata.csv'))
            df = df.append(_)
            del _

        print(f"Size before droping duplicates: {len(df)}")
        df = df.drop_duplicates()
        print(f"Size after droping duplicates: {len(df)}")
        df.to_csv(os.path.join(save_location, 'csv_metadata_nchlt_aux','all_data.csv'), index=False)


    def _clean_data(self,
                   save_location,
                   aux_location,
                   nchlt_location,
                   min_pdp = 0.2,
                   min_seconds=3,
                   max_seconds=5,
                   lang_prob=0.9,):

        nchlt = pd.read_csv(nchlt_location)
        nchlt_aux = pd.read_csv(aux_location)

        # ensure AUX data has good pronounciation
        nchlt_aux = nchlt_aux[nchlt_aux.PDP_SCORE>=min_pdp]

        df = pd.concat([nchlt,nchlt_aux])
        df = df[(df.DURATION>=min_seconds) & (df.DURATION<=max_seconds)]

        def detect_incorrect_lang(row):
            if row.LANGUAGE == 'eng':
                pass
            else:
                if row.DETECTED_LANGUAGE=='en' and row.DETECTED_LANGUAGE_PROB>lang_prob:
                    return False
            
            
            return True
        
        df["CORRECT LANGUAGE"] = df.apply(lambda x: detect_incorrect_lang(x), axis = 1)
        df = df[df['CORRECT LANGUAGE']]

        df.to_csv(os.path.join(save_location, 'filtered_metadata.csv'), index=False)
    
    def _load_data_and_divide_data_into_data_sets(self, save_location, seed=42):

        np.random.seed(seed)
        speaker_roles = pd.DataFrame(columns=["ID", "LANGUAGE", "ROLE"])
        df = pd.read_csv(os.path.join(save_location, 'filtered_metadata.csv'))
        
        for lang in LANG_ABBREVIATIONS:
            
            df_shuffled = shuffle(df[df.LANGUAGE == lang])
            speakers = df_shuffled.ID.unique()

            total_train_speakers = int(np.floor(len(speakers)*0.8))
            total_validation_speakers = (len(speakers)-total_train_speakers)//2

            train_speakers = speakers[:total_train_speakers]
            val_speakers = speakers[total_train_speakers:-total_validation_speakers]
            test_speakers = speakers[-total_validation_speakers:]

            train = pd.DataFrame({"ID": train_speakers,
                                "LANGUAGE": [lang]*total_train_speakers,
                                "ROLE": ["TRAIN"]*total_train_speakers})
            val = pd.DataFrame({"ID": val_speakers,
                                "LANGUAGE": [lang]*total_validation_speakers,
                                "ROLE": ["VALIDATION"]*total_validation_speakers})
            test = pd.DataFrame({"ID": test_speakers,
                                "LANGUAGE": [lang]*total_validation_speakers,
                                "ROLE": ["TEST"]*total_validation_speakers})

            _ = pd.concat([train, val, test])
            speaker_roles = pd.concat([speaker_roles, _])
        
        df = pd.merge(df, speaker_roles, on=['ID', 'LANGUAGE'])
        df.drop('Unnamed: 0', axis=1).to_csv(os.path.join(save_location, 'filtered_metadata_with_roles.csv'), index=False)


    def create_final_metadata_file(self,
                                   data_folder='./',
                                   min_pdp = 0.2,
                                   min_seconds=3,
                                   max_seconds=5,
                                   lang_prob=0.9,):
        """
        Create final metadata CSV combining and cleaning all of the 
        data as well assigning roles and adding the image path where
        the spectrograms can be found.
        """
        
        self._extract_meta_data(data_folder)
        self._create_combined_metadata_csv(data_folder)
        self._clean_data(save_location=os.path.join(data_folder, 'csv_metadata'),
                         aux_location=os.path.join(data_folder, 'csv_metadata_nchlt_aux', 'all_data.csv'),
                         nchlt_location=os.path.join(data_folder, 'csv_metadata_nchlt', 'all_data.csv'))
        self._load_data_and_divide_data_into_data_sets(save_location=os.path.join(data_folder, 'csv_metadata'), seed=42)

        df = pd.read_csv(os.path.join(os.path.join(data_folder, 'csv_metadata'), 'filtered_metadata_with_roles.csv'))

        def generate_image_path(row):
            audio_name = row['FILE'].split('/')[-1].split('.wav')[0] # extract audio name and remove wav
            duration = str(row['DURATION']).replace('.', '') # get duration
            file_name = f'{audio_name}_{duration}.png'
            language = row['LANGUAGE']
            role = row['ROLE']
            image_path = os.path.join(role, language, file_name).replace('\\', '/')#ensure window runs don't mess it up
            return  image_path
    
        df['IMAGE_PATH'] = df.apply(lambda row: generate_image_path(row), axis=1)

        df.to_csv(os.path.join(os.path.join(data_folder, 'csv_metadata'), 'metadata.csv'), index=False)

class ImageGenerator(object):

    def __init__(self,
                 meta_data_path=None):

        if meta_data_path is None:
            pass
        else:
            self.meta_data_path = meta_data_path
            self.meta_data = pd.read_csv(self.meta_data_path)

    def _divide_audio_sample_into_chunks(self,
                                         file_location):

        y, _ = librosa.load(file_location, sr=SR)
        chunk = np.array(list(chunked(y, SAMPLES_PER_3S))[0])
        return center(chunk, SAMPLES_PER_3S, mode='constant') # ensure 3 seconds long

    def _generate_spectogram_from_audio_file(self,
                                             file_location,
                                             save_location,
                                             image_h = 128,
                                             image_w = 128,
                                             n_mels=128,
                                             fmax=8000, 
                                             dpi=250):
        
        audio = self._divide_audio_sample_into_chunks(file_location)
        S = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=n_mels, fmax=fmax)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # first get image out of LIBROSA
        fig = plt.figure()
        _ = librosa.display.specshow(data=S_dB, y_axis=None, x_axis=None, sr=SR, fmax=fmax)
        fig.savefig(save_location, format="png", dpi=dpi, bbox_inches='tight', pad_inches=0, )
        plt.close(fig)
        
        # second load with TF and make correct format for models
        img = tf.keras.preprocessing.image.load_img(save_location, grayscale=False,
                                                    color_mode='rgb', target_size=(image_h, image_w),
                                                    interpolation='nearest')
        tf.keras.preprocessing.image.save_img(save_location, img, scale=True)
        

    def convert_and_save_all_audio(self, 
                                   recording_location='./',
                                   image_folders_location='./',
                                   image_h = 128,
                                   image_w = 128,
                                   n_mels=128,
                                   fmax=8000, 
                                   dpi=250):

        total_files_not_found = {lang:0 for lang in LANG_ABBREVIATIONS}
        total_training_examples = {lang:0 for lang in LANG_ABBREVIATIONS}
        for _, row in tqdm(self.meta_data.iterrows()):

            file_location = row['FILE']
            save_location = os.path.join(image_folders_location, row['IMAGE_PATH'])    

            if os.path.isfile(os.path.join(recording_location, file_location)):   
                self._generate_spectogram_from_audio_file(file_location = os.path.join(recording_location, file_location), 
                                                          save_location = save_location)
                total_training_examples[row['LANGUAGE']]+=1
            else:
                total_files_not_found[row['LANGUAGE']]+=1

        print('Below the summary of successful examples: ')
        print(total_training_examples)
        print('')
        print('Below the summary of unsuccessful examples: ')
        print(total_files_not_found)




