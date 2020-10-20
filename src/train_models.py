import os
import pickle
import numpy as np
import pandas as pd

import tensorflow as tf
# Set up GPU config
print("Setting up GPU if found")
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if physical_devices:
    print(f'Physical devices found: {physical_devices}')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

import tensorflow_addons as tfa
import nlpaug.augmenter.spectrogram as nas
import nlpaug.flow as naf

AUTOTUNE = tf.data.experimental.AUTOTUNE

DATA_DIR = './'
MODEL_LOCATION = './'
CLASS_NAMES = ['afr', 'eng', 'nso', 'tsn', 'xho', 'zul']
BATCH_SIZE = 32


MODELS = ['custom', 'crnn', 'resnet50', 'densenet', 
          'custom', 'crnn', 'resnet50', 'densenet',
          'custom', 'crnn', 'resnet50', 'densenet']
TESTS = ['both', 'both', 'both', 'both',
         'triplet', 'triplet', 'triplet', 'triplet',
         'softmax', 'softmax', 'softmax', 'softmax']

# setting seeds for reproducibility
np.random.seed(1432)
tf.random.set_seed(1432)
random.seed(1432)

def extract_language_from_file(file_path, type_='both'):
    lang = tf.strings.split(file_path, os.path.sep)
    lang = (lang[-2] == CLASS_NAMES) # create one hot encoded vector
    lang = tf.dtypes.cast(lang, tf.int8) 

    if type_=='both':
        label = {'triplet-output':tf.argmax(lang),'softmax-output': lang}
    elif type_=='softmax':
        label = {'softmax-output': lang}
    elif type_ == 'triplet':
        label = {'triplet-output':tf.argmax(lang)}
    
    return label

def perform_spec_flow(image):
    image = image.numpy()
    options = ['H', 'V', 'HV', 'NORMAL']
    choice = np.random.choice(options, 1, p=[0.2, 0.2, 0.35, 0.25])[0]
    if choice == 'HV':
        flow = naf.Sequential([nas.FrequencyMaskingAug(mask_factor=30), nas.TimeMaskingAug(mask_factor=30), nas.TimeMaskingAug(mask_factor=20)])
        image = flow.augment(image)
    elif choice == 'H':
        flow = naf.Sequential([nas.TimeMaskingAug(mask_factor=30)])
        image = flow.augment(image)
    elif choice == 'V':
        flow = naf.Sequential([nas.FrequencyMaskingAug(mask_factor=30)])
        image = flow.augment(image)
    else:
        image = image
    image = tf.convert_to_tensor(image)
    return image

def create_input_data_and_get_labels_train(path, width=128, height=128, type_='both'):
    label = extract_language_from_file(path, type_)
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.dtypes.cast(img, tf.float32)
    img = tf.image.resize(img, [height, width])
    img = tf.py_function(perform_spec_flow, [img], [tf.float32])[0]
    return img, label

def create_input_data_and_get_labels_val(file, width=128, height=128, type_='both'):
    label = extract_language_from_file(file, type_)
    img = tf.io.read_file(file)
    img = tf.image.decode_png(img, channels=3)
    img = tf.dtypes.cast(img, tf.float32)
    img = tf.image.resize(img, [height, width])
    return img, label

def set_shapes(img, label, img_shape):
    img.set_shape(img_shape)
    return img, label

def return_densenet_model(image_width = 128, image_height = 128, type_='both'):
    weight_decay = 0.001

    input_ = tf.keras.layers.Input(shape=(image_height, image_width, 3))
    x = tf.keras.applications.densenet.preprocess_input(input_)
    res50 = tf.keras.applications.DenseNet121(include_top=False,
                                              input_shape = (image_height,image_width,3),
                                              weights=None,
                                              classes = len(CLASS_NAMES), 
                                              pooling='avg')(x)
    x = tf.keras.layers.Flatten()(res50)

    if type_ == 'both':
        embedding = tf.keras.layers.Dense(units=512,
                                          activation=None,
                                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          name='embeddings')(x)
        triplet_output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='triplet-output')(embedding)

        softmax_output = tf.keras.layers.Dense(units=len(CLASS_NAMES), activation='softmax', name='softmax-output')(triplet_output)
        model = tf.keras.Model(inputs=input_, outputs=[triplet_output, softmax_output])

    elif type_ == 'softmax':
        x = tf.keras.layers.Dense(units=512,
                                  activation=None,
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                  name='embeddings')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        softmax_output = tf.keras.layers.Dense(units=len(CLASS_NAMES), activation='softmax', name='softmax-output')(x)
        model = tf.keras.Model(inputs=input_, outputs=[softmax_output])

    elif type_ == 'triplet':
        embedding = tf.keras.layers.Dense(units=512,
                                          activation=None,
                                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          name='embeddings')(x)
        triplet_output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='triplet-output')(embedding)
        model = tf.keras.Model(inputs=input_, outputs=[triplet_output])

    return model


def return_resnet_model(image_width = 128, image_height = 128, type_='both'):
    weight_decay = 0.001

    input_ = tf.keras.layers.Input(shape=(image_height, image_width, 3))
    x = tf.keras.applications.resnet.preprocess_input(input_)
    res50 = tf.keras.applications.ResNet50(include_top=False,
                                           input_shape = (image_height,image_width,3),
                                           weights=None,
                                           classes = len(CLASS_NAMES),
                                           pooling='avg')(x)
    x = tf.keras.layers.Flatten()(res50)

    if type_ == 'both':
        embedding = tf.keras.layers.Dense(units=512,
                                          activation=None,
                                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          name='embeddings')(x)
        triplet_output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='triplet-output')(embedding)

        softmax_output = tf.keras.layers.Dense(units=len(CLASS_NAMES), activation='softmax', name='softmax-output')(triplet_output)
        model = tf.keras.Model(inputs=input_, outputs=[triplet_output, softmax_output])

    elif type_ == 'softmax':
        x = tf.keras.layers.Dense(units=512,
                                  activation=None,
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                  name='embeddings')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        softmax_output = tf.keras.layers.Dense(units=len(CLASS_NAMES), activation='softmax', name='softmax-output')(x)
        model = tf.keras.Model(inputs=input_, outputs=[softmax_output])

    elif type_ == 'triplet':
        embedding = tf.keras.layers.Dense(units=512,
                                          activation=None,
                                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          name='embeddings')(x)
        triplet_output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='triplet-output')(embedding)
        model = tf.keras.Model(inputs=input_, outputs=[triplet_output])

    return model


def return_crnn_model(image_width = 150, image_height = 128, type_='both'):
    weight_decay = 0.001

    input_ = tf.keras.layers.Input(shape=(image_height, image_width, 3))
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(input_)
    x = tf.keras.layers.Conv2D(filters=64, 
                               kernel_size=(3,3), 
                               strides=1, activation=None,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=128, 
                               kernel_size=(3,3), 
                               strides=1,
                               activation=None, 
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=256, 
                               kernel_size=(3,3), 
                               strides=1,
                               activation=None, 
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = tf.keras.layers.Conv2D(filters=512, 
                            kernel_size=(3,3), 
                            strides=1,
                            activation=None, 
                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # (bs, y, x, c) --> (bs, x, y, c)
    x = tf.keras.layers.Permute((2, 1, 3))(x)

    middle = tf.keras.Model(inputs=input_, outputs=[x])
    # (bs, x, y, c) --> (bs, x, y * c)
    bs, x_, y, c = middle.layers[-1].output_shape


    x = tf.keras.layers.Reshape((x_, y*c))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False), merge_mode="concat")(x)
    
    if type_ == 'both':
        embedding = tf.keras.layers.Dense(units=512,
                                          activation=None,
                                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          name='embeddings')(x)
        triplet_output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='triplet-output')(embedding)

        softmax_output = tf.keras.layers.Dense(units=len(CLASS_NAMES), activation='softmax', name='softmax-output')(triplet_output)
        model = tf.keras.Model(inputs=input_, outputs=[triplet_output, softmax_output])

    elif type_ == 'softmax':
        x = tf.keras.layers.Dense(units=512,
                                  activation=None,
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                  name='embeddings')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        softmax_output = tf.keras.layers.Dense(units=len(CLASS_NAMES), activation='softmax', name='softmax-output')(x)
        model = tf.keras.Model(inputs=input_, outputs=[softmax_output])

    elif type_ == 'triplet':
        embedding = tf.keras.layers.Dense(units=512,
                                          activation=None,
                                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          name='embeddings')(x)
        triplet_output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='triplet-output')(embedding)
        model = tf.keras.Model(inputs=input_, outputs=[triplet_output])

    return model

class SpectoNetCnnBlockIdentity(tf.keras.Model):
    def __init__(self, kernel_size, filters, weight_decay, name=''):
        super(SpectoNetCnnBlockIdentity, self).__init__(name=name)
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters=filters1,
                                                kernel_size=(kernel_size, kernel_size),
                                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                activation='relu',
                                                strides=1,
                                                padding='same')
        self.bna = tf.keras.layers.BatchNormalization()
        
        self.conv2b = tf.keras.layers.Conv2D(filters=filters2,
                                                kernel_size=(kernel_size, kernel_size),
                                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                activation='relu',
                                                strides=1,
                                                padding='same')
        self.bnb = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters=filters3,
                                                kernel_size=(kernel_size, kernel_size),
                                                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                                activation='relu',
                                                strides=1,
                                                padding='same')
        self.bnc = tf.keras.layers.BatchNormalization()

        self.pool = tf.keras.layers.Conv2D(filters=filters3,
                                            kernel_size=(1, 1),
                                            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                            activation='relu',
                                            strides=2)

        self.correct_dimension = tf.keras.layers.Conv2D(filters=filters3,
                                                        kernel_size=(1, 1),
                                                        activation='relu',
                                                        strides=1)

    def call(self, input_tensor):
        x = self.conv2a(input_tensor)
        x = self.bna(x)

        x = self.conv2b(x)
        x = self.bnb(x)

        x = self.conv2c(x)
        x = self.bnc(x)

        identity = self.correct_dimension(input_tensor)
        x += identity

        x = self.pool(x)

        return tf.nn.relu(x)


def return_custom_model(image_width = 128, image_height = 128, type_='both'):
    weight_decay = 0.001

    input_ = tf.keras.layers.Input(shape=(image_height, image_width, 3))
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(input_)
    x = SpectoNetCnnBlockIdentity(3, [16, 16, 16], weight_decay)(x)
    x = SpectoNetCnnBlockIdentity(3, [32, 32, 32], weight_decay)(x)
    x = SpectoNetCnnBlockIdentity(3, [128, 128, 128], weight_decay)(x)
    x = SpectoNetCnnBlockIdentity(3, [256, 256, 256], weight_decay)(x)

    x = tf.keras.layers.Flatten()(x)

    if type_ == 'both':
        embedding = tf.keras.layers.Dense(units=512,
                                          activation=None,
                                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          name='embeddings')(x)
        triplet_output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='triplet-output')(embedding)

        softmax_output = tf.keras.layers.Dense(units=len(CLASS_NAMES), activation='softmax', name='softmax-output')(triplet_output)
        model = tf.keras.Model(inputs=input_, outputs=[triplet_output, softmax_output])

    elif type_ == 'softmax':
        x = tf.keras.layers.Dense(units=512,
                                  activation=None,
                                  kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                  name='embeddings')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        softmax_output = tf.keras.layers.Dense(units=len(CLASS_NAMES), activation='softmax', name='softmax-output')(x)
        model = tf.keras.Model(inputs=input_, outputs=[softmax_output])

    elif type_ == 'triplet':
        embedding = tf.keras.layers.Dense(units=512,
                                          activation=None,
                                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                          name='embeddings')(x)
        triplet_output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name='triplet-output')(embedding)
        model = tf.keras.Model(inputs=input_, outputs=[triplet_output])

    return model

def return_datasets(image_width = 128, image_height = 128, type_='both'):

    tf.random.set_seed(1432)

    train_ds = tf.data.Dataset.list_files(f'{DATA_DIR}TRAIN/*/*', shuffle=True)
    train_ds = train_ds.shuffle(20000, reshuffle_each_iteration=True, seed=1432)
    train_ds = train_ds.map(lambda x: create_input_data_and_get_labels_train(x, width=image_width, height=image_height, type_=type_), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(lambda img, label: set_shapes(img, label, [image_height,image_width,3]), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

    val_ds = tf.data.Dataset.list_files(f'{DATA_DIR}VALIDATION/*/*', shuffle=True)
    val_ds = val_ds.map(lambda x: create_input_data_and_get_labels_val(x, width=image_width, height=image_height, type_=type_), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds

def return_required_test_items(test, model):

    if model == 'custom':
        model = return_custom_model(image_width = 128, image_height = 128, type_=test)
        train_ds, val_ds = return_datasets(image_width = 128, image_height = 128, type_=test)
    elif model == 'crnn':
        model = return_crnn_model(image_width = 150, image_height = 128, type_=test)
        train_ds, val_ds = return_datasets(image_width = 150, image_height = 128, type_=test)   
    elif model == 'resnet50':
        model = return_resnet_model(image_width = 224, image_height = 224, type_=test)
        train_ds, val_ds = return_datasets(image_width = 224, image_height = 224, type_=test)       
    elif model == 'densenet':
        model = return_densenet_model(image_width = 224, image_height = 224, type_=test)
        train_ds, val_ds = return_datasets(image_width = 224, image_height = 224, type_=test)   

    return model, train_ds, val_ds

print(f'Tensorflow version: {tf.__version__}')

train_files = []
labels_train = []
for lang in CLASS_NAMES:
    files = [os.path.join(DATA_DIR, 'TRAIN', lang, file) for file in os.listdir(os.path.join(DATA_DIR, 'TRAIN', lang)) if '.DS' not in file]
    train_files.extend(files)
    labels_train.extend([lang]*len(files))
    
val_files = []
labels_val = []
for lang in CLASS_NAMES:
    files = [os.path.join(DATA_DIR, 'VALIDATION', lang, file) for file in os.listdir(os.path.join(DATA_DIR, 'VALIDATION', lang)) if '.DS' not in file]
    val_files.extend(files)
    labels_val.extend([lang]*len(files))

print(f'Total training images: {len(train_files)}')
print(f'Total validation images: {len(val_files)}')


# run all the tests
for m, t in zip(MODELS, TESTS):
    print(f'Busy with model {m} and test {t}')

    model, train_ds, val_ds = return_required_test_items(t, m)

    if t == 'both':
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss={"triplet-output": tfa.losses.TripletSemiHardLoss(margin=0.2),
                            "softmax-output": tf.keras.losses.CategoricalCrossentropy()},
                      metrics = {'softmax-output':['accuracy', tf.keras.metrics.AUC()]})
    elif t=='triplet':
         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                       loss={"triplet-output": tfa.losses.TripletSemiHardLoss(margin=0.2)})     
    elif t=='softmax':
         model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                       loss={"softmax-output": tf.keras.losses.CategoricalCrossentropy()},
                       metrics = {'softmax-output':['accuracy', tf.keras.metrics.AUC()]})   

    model.summary()

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(MODEL_LOCATION, 'models', f'{m}_{t}'),
                                                                   save_weights_only=False,
                                                                   monitor='val_loss',
                                                                   mode='min',
                                                                   save_best_only=True)

    history= model.fit(train_ds,
                       validation_data=val_ds,
                       epochs=500,
                       callbacks=[early_stopping_callback, model_checkpoint_callback],
                       verbose=1)

    with open(os.path.join(MODEL_LOCATION, 'history', f'{m}_{t}_history.pkl'), 'wb') as f:
        pickle.dump(history.history, f)