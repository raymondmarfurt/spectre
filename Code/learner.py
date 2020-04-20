import gzip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import time
import h5py
import urllib
import os
import sys
import math

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import itertools
import logging


# logging  

analyze_result = False

validate = False


LOG = "val_acc_max.log"                                                     

NUMBER_OF_SAMPLES = 8000
NUMBER_OF_VALIDATIONS = math.floor(NUMBER_OF_SAMPLES/8)

TEST_SUITE_NAME = "batch_" + str(NUMBER_OF_SAMPLES) + "_random_19-APR-2020"


logging.basicConfig(
        filename=LOG,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')

# console handler  
console = logging.StreamHandler()  
console.setLevel(logging.DEBUG)  
logging.getLogger("").addHandler(console)

my_batch_size = 128
nb_classes = 2
nb_epoch = 50
img_rows, img_cols, img_channels = 128, 128, 3
kernel_size = (3, 3)
my_input_shape = (img_rows, img_cols, img_channels)
pool_size = (2, 2)


def strip_zeros_from_params(param_str):
    params = param_str.strip().split()
    hist_entry = ''
    for i in range(int(len(params)/3)):
        if float(params[i*3]) > 0.0:
            hist_entry += ' ' + params[i*3] + ' ' + params[i*3+1] + ' ' + params[i*3+2]
    return hist_entry.strip()


layer_arch_history = {'empty'}
with open(LOG) as f:
    content = f.readlines()
    for line in content:
        if "INFO" in line or "DEBUG" in line:
            splits = line.split('For parameters')
            if len(splits) == 2:
                layer_arch_history.add(strip_zeros_from_params(splits[1].split(',')[0]))

for entry in layer_arch_history:
    print(entry)

logging.info("Found  " + str(len(layer_arch_history)) + " configurations in " + LOG + ". Will skip these.")


print("loading HDF5 file...")
h5f = h5py.File('sampled_spectrograms_3.hdf5', 'r')
X_train = h5f.get('X_train')[0:NUMBER_OF_SAMPLES]
Y_train = np.asarray(h5f.get('Y_train')[0:NUMBER_OF_SAMPLES]).astype(int)
Y_train = Y_train - 1 
X_valid = h5f.get('X_valid')[0:NUMBER_OF_VALIDATIONS]
Y_valid = np.asarray(h5f.get('Y_valid')[0:NUMBER_OF_VALIDATIONS]).astype(int)
Y_valid = Y_valid - 1 
print("loaded HDF5 file.")

#rmd=np.random.randint(0,len(X_train))
#img_data = X_train[rmd]
#plt.imshow(img_data,interpolation="bicubic")
#plt.show()

print("Calculating mean/std...")
X_mean = np.mean( X_train, axis = 0)
X_std = np.std( X_train, axis = 0)
X_train = (X_train - X_mean ) / (X_std + 0.0001)
X_valid = (X_valid - X_mean ) / (X_std + 0.0001)
print("Calculated mean/std.")

#img_data = X_train[rmd]
#plt.imshow(img_data,interpolation="bicubic")
#plt.show()

def convertToOneHot(vector, num_classes=None):
    result = np.zeros((len(vector), num_classes), dtype='int32')
    result[np.arange(len(vector)), vector] = 1
    return result

Y_train=convertToOneHot(Y_train,num_classes=2)
Y_valid=convertToOneHot(Y_valid,num_classes=2)


def add_c_layer(model, filter_param, dropout_param,max_pool_param):
    if filter_param > 0:
        model.add(Convolution2D(filter_param, kernel_size, input_shape=my_input_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        if max_pool_param > 0:
            model.add(MaxPooling2D(pool_size=pool_size))
        if dropout_param > 0:
            model.add(Dropout(dropout_param))



def run_model(run_name, parameters):
    if parameters[0] == None:
        return

    str_params = ''.join("{:10.2f}".format(p) for p in parameters)
    str_params = strip_zeros_from_params(str_params)
    if str_params in layer_arch_history:
        logging.debug("For parameters " + str_params + ", skipped because architecture already seen.")
        return
    else:
        layer_arch_history.add(str_params)


    c_layer_parameters = parameters[:-3]
    d_layer_parameters = parameters[-3:]
    number_of_c_layers = int(len(c_layer_parameters) / 3)
    model = Sequential()
    for i in range(number_of_c_layers):
        add_c_layer(model, c_layer_parameters[i * 3],
                c_layer_parameters[(i*3)+1], c_layer_parameters[(i*3)+2])
    model.add(Flatten())
    model.add(Dense(d_layer_parameters[0]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(d_layer_parameters[1]))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    count_param_max_threshold = 200000000
    count_param_min_threshold = 1000
    if model.count_params() > count_param_max_threshold:
        logging.debug("For parameters " + str_params + ", skipped. # params " + str(model.count_params()) + " too high.")
        K.clear_session()
        return
    if model.count_params() < count_param_min_threshold:
        logging.debug("For parameters " + str_params + ", skipped. # params " + str(model.count_params()) + " too low.")
        K.clear_session()
        return


    model.evaluate(X_train,Y_train)

    dir_name="Checkpoints/music/"+TEST_SUITE_NAME+"/" + run_name  

    my_callbacks = []
    if analyze_result:
        tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir='tensorboard/music/'+TEST_SUITE_NAME+"/"  + run_name + '/', 
                write_graph=True,
                histogram_freq=1)

        os.makedirs(dir_name,exist_ok=True)       
        checkpointer = tf.keras.callbacks.ModelCheckpoint( 
                filepath =  dir_name + "/cnn_model_"+"weights_epoch_{epoch:03d}-{val_loss:.2f}.hdf5",
                verbose = 1, 
                save_best_only = False,
                period = 10 )
        my_callbacks = [tensorboard, checkpointer]
    else:
        early_stopper = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.0,
                patience=3,
                mode='auto')
        my_callbacks = [early_stopper]


    history=model.fit(X_train, Y_train, 
                      batch_size=my_batch_size, 
                      epochs=nb_epoch,
                      verbose=2, 
                      validation_data=(X_valid, Y_valid),
                      callbacks=my_callbacks,)
    
    K.clear_session()



    logging.info("For parameters " + str_params + ", found MAX val_acc: " + "{:10.2f}".format(max(history.history['val_acc'])) + " with # params {:12}".format(model.count_params()))

layers = []
layers += [[16],[0],[0]]
layers += [[0,16],[0],[1]]
layers += [[0,16],[0],[0]]
layers += [[0,32],[0],[1]]
layers += [[0,32],[0],[0]]
layers += [[0,32],[0],[1]]
layers += [[0,64],[0],[0]]
layers += [[0,64],[0],[1]]
layers += [[0,64],[0],[0]]
layers += [[0,16],[0],[1]]
layers += [[400],[0.5],[0]]

if analyze_result:
    param_set = [16,0,1,16,0,1,16,0,1,16,0,1,400,0,0]
    run_name = "params_" + "_".join(str(p) for p in param_set)
    #print(run_name)
    run_model(run_name, param_set)

else:
    param_sets = itertools.product(*layers)

    run_count = 0
    for param_set in param_sets:
        run_count += 1
        run_name = "params_" + "_".join(str(p) for p in param_set)
        logging.debug("--> RUN " + str(run_count) + " --- " + run_name + " --------")
        #print(run_name)
        try:
            run_model(run_name, param_set)
        except:
            logging.error("Error during run " + run_name)



# summarize history for accuracy


#load a saved model

#from keras.models import load_model
#model = load_model("/notebooks/local/dl_course_2018/notebooks/Checkpoints/8_faces/model_1_fc/fc_model_weights_epoch_010-1.40.hdf5")
#preds=model.predict(X_valid)
#from sklearn.metrics import confusion_matrix
#print(confusion_matrix(np.argmax(Y_valid,axis=1),np.argmax(preds,axis=1)))
#print("Acc = " ,np.sum(np.argmax(Y_valid,axis=1)==np.argmax(preds,axis=1))/len(preds))

#load a saved model
if validate:
    from tf.keras.models import load_model
    #model = load_model("Checkpoints/8_faces/model_1_cnn/params_16_0_32_0_0_0_64_0_400_0/cnn_model_weights_epoch_010-1.07.hdf5")
    preds=model.predict(X_valid)
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(np.argmax(Y_valid,axis=1),np.argmax(preds,axis=1)))
    print("Acc = " ,np.sum(np.argmax(Y_valid,axis=1)==np.argmax(preds,axis=1))/len(preds))
    exit(0)

