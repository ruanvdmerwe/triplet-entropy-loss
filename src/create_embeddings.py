import os
import pandas as pd
import tensorflow as tf

# Set up GPU config
print("Setting up GPU if found")
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        
print(f'Tensorflow version: {tf.__version__}')

DATA_DIR = "./"
CLASS_NAMES = ['afr', 'eng','nso','tsn', 'xho', 'zul']
AUTOTUNE = tf.data.experimental.AUTOTUNE

def spectogram_generator(file_locations, model_location):
    for file in file_locations:
        img = tf.io.read_file(file)
        img = tf.image.decode_png(img, channels=3)
        
        if 'custom' in model_location.decode('ascii'):
            img = tf.image.resize(img, [128, 128])
        elif 'crnn' in model_location.decode('ascii'):
            img = tf.image.resize(img, [128, 150])
        elif 'resnet50' in model_location.decode('ascii'):
            img = tf.image.resize(img, [224, 224])
        elif 'densenet' in model_location.decode('ascii'):
            img = tf.image.resize(img, [224, 224])
        elif 'inception' in model_location.decode('ascii'):
            img = tf.image.resize(img, [224, 224])

        yield img

def get_files():
    files = []
    labels = []
    for lang in CLASS_NAMES:
        files = [os.path.join(DATA_DIR, lang, file) for file in os.listdir(os.path.join(DATA_DIR, lang)) if '.DS' not in file]
        files.extend(files)
        labels.extend([lang]*len(files))
        
    return files, labels
    
        
def get_model_datasets(model_location):
    
    files, labels = get_files()

    model = tf.keras.models.load_model(model_location, compile=False)
    
    ds = tf.data.Dataset.from_generator(spectogram_generator,
                                              args=[files, model_location],
                                              output_types=tf.float16)
    ds = ds.batch(1).prefetch(AUTOTUNE)
    
    model.summary()
    
    return model, ds, labels, files

locations = ['./densenet_both_imagenet'] # location of saved .pb model 

for loc in locations:
    
    print(f'Busy with {loc.split("/")[-1]}')
    
    if '.ipynb' not in loc and '.py' not in loc and '.pkl' not in loc and '.ipynb_checkpoints' not in loc:
    
        model, ds, labels, files = get_model_datasets(loc)

        files = [f.split('/')[-1] for f in files]

        if 'triplet' in loc or 'softmax' in loc:
            embeddings = model.predict(ds, verbose=1)
            embeddings = pd.DataFrame(embeddings)
            embeddings['LABEL'] = labels
            embeddings['FILE'] = files
            embeddings.to_pickle(f'{loc.split("/")[-1]}_predictions.pkl')

        elif 'softmax' in loc:
            
            # create model to extract only the embeddings
            input_ = tf.keras.layers.Input(shape=(224, 224, 3))
            x = model.layers[1](input_)
            for layer in model.layers[2:-3]:
                layer.trainable = False
                x = layer(x)
            model_ = tf.keras.Model(inputs=input_, outputs=x)

            embeddings = model_.predict(ds, verbose=1)
            embeddings = pd.DataFrame(embeddings)
            embeddings['LABEL'] = labels
            embeddings['FILE'] = files

            logits_ = model.predict(ds, verbose=1)
            embeddings['AFR'] = logits_[:, 0]
            embeddings['ENG'] = logits_[:, 1]
            embeddings['NSO'] = logits_[:, 2]
            embeddings['TSN'] = logits_[:, 3]
            embeddings['XHO'] = logits_[:, 4]
            embeddings['ZUL'] = logits_[:, 5]

            embeddings.to_pickle(f'{loc.split("/")[-1]}_predictions.pkl')
            
        elif 'both' in loc:
            embeddings = model.predict(ds, verbose=1)
            embedddings = embeddings[0]
            logits =  embeddings[1]
            embeddings = pd.DataFrame(embedddings)
            embeddings['AFR'] = logits[:, 0]
            embeddings['ENG'] = logits[:, 1]
            embeddings['NSO'] = logits[:, 2]
            embeddings['TSN'] = logits[:, 3]
            embeddings['XHO'] = logits[:, 4]
            embeddings['ZUL'] = logits[:, 5]
            embeddings['LABEL'] = labels
            embeddings['FILE'] = files
            embeddings.to_pickle(f'{loc.split("/")[-1]}_predictions.pkl')