import numpy as np
import h5py
import matplotlib.pyplot as plt
import math


IN_FILE ="/var/tmp/DATA_2/spectro_data_mel.hdf5"
OUT_FILE ="/var/tmp/DATA_2/sampled_spectro_data_mel.hdf5"
OUT_FILE_TEST ="/var/tmp/DATA_2/sampled_spectro_data_mel_test.hdf5"

h5f = h5py.File(IN_FILE, 'r')
X = h5f.get('X')[:]
Y = h5f.get('Y')[:]

#rmd=np.random.randint(0,len(X))
#img_data = X[rmd]
#plt.imshow(img_data,interpolation="bicubic")
#plt.show()
#exit(0)


number_of_songs = len(X)

samples_per_song = 1

width=480
height=640


random_song_idx = np.arange(number_of_songs)
np.random.shuffle(random_song_idx)


# randomize data
number_of_X_train_songs = math.floor(number_of_songs * 0.8)
number_of_X_valid_songs = math.floor(number_of_songs * 0.1)
number_of_X_test_songs = number_of_songs - number_of_X_train_songs - number_of_X_valid_songs

X_train = np.empty([number_of_X_train_songs,width,height,3])
Y_train = np.empty([number_of_X_train_songs])
X_valid = np.empty([number_of_X_valid_songs,width,height,3])
Y_valid = np.empty([number_of_X_valid_songs])
X_test = np.empty([number_of_X_test_songs,width,height,3])
Y_test = np.empty([number_of_X_test_songs])

random_song_idx_train = random_song_idx[0:number_of_X_train_songs]
random_song_idx_valid = random_song_idx[number_of_X_train_songs:number_of_X_train_songs + number_of_X_valid_songs]
random_song_idx_test = random_song_idx[number_of_X_train_songs + number_of_X_valid_songs:]

counter = 0
for song_idx in random_song_idx_train:
    song_data = X[song_idx]
    song_class = Y[song_idx]
    X_train[counter] = song_data
    Y_train[counter] = song_class
    counter += 1

counter = 0
for song_idx in random_song_idx_valid:
    song_data = X[song_idx]
    song_class = Y[song_idx]
    X_valid[counter] = song_data
    Y_valid[counter] = song_class
    counter += 1
   
counter = 0
for song_idx in random_song_idx_test:
    song_data = X[song_idx]
    song_class = Y[song_idx]
    X_test[counter] = song_data
    Y_test[counter] = song_class
    counter += 1

X = None
Y = None


with h5py.File(OUT_FILE, "w") as f:
    f.create_dataset('X_train', data=np.asarray(X_train))
    f.create_dataset('Y_train', data=np.asarray(Y_train))
    f.create_dataset('X_valid', data=np.asarray(X_valid))
    f.create_dataset('Y_valid', data=np.asarray(Y_valid))
    f.close()

with h5py.File(OUT_FILE_TEST, "w") as f:
    f.create_dataset('X_test', data=np.asarray(X_test))
    f.create_dataset('Y_test', data=np.asarray(Y_test))
    f.close()
