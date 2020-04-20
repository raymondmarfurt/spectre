import os
import magic
import hashlib
import pathlib
import numpy as np
from shlex import quote
from subprocess import Popen, PIPE, STDOUT
import mutagen
from mutagen.mp3 import MP3
import librosa
from librosa import display as librosa_display
import matplotlib
import pylab


#Define
currentPath = os.path.dirname(os.path.realpath(__file__)) 

OUT_DIR= "/var/tmp/DATA_2"
DIC_FILE="/var/tmp/DATA_2/DATA_DIC.csv"

def append_entry(dic,entry):
    return np.reshape(np.append(dic,entry),(-1,4))

def write_entries(dic):
        np.savetxt(DIC_FILE, dic, delimiter="|", fmt='%s')         

def load_entries():
    try:
        dic = np.loadtxt(DIC_FILE, delimiter="|", dtype="str")
        return dic
    except:
        with open('/tmp/data_log.txt', 'a') as logfile:
                print('Error reading DIC_FILE', file=logfile)  # python 3.x
        return np.empty(0)

def file_in_entries(dic,file_name):
    if dic.size == 0:
        return False
    return np.max(np.isin(dic[:,0],file_name))


file_dic = load_entries()
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

for root, directories, filenames in os.walk('/mp3'):
    for filename in filenames:
        f = os.path.join(root,filename)
        #in_file = f.replace("'","'\"'\"'") # heavy escaping for single quotes in file names...
        in_file = f
        with open('/tmp/data_log.txt', 'a') as logfile:
                print('processing:', in_file, file=logfile)  # python 3.x

        if file_in_entries(file_dic,f):
            with open('/tmp/data_log.txt', 'a') as logfile:
                print('Skipping:', f, file=logfile)  # python 3.x

            continue

        f_type = magic.from_file(f)
        if not f_type.startswith("Audio file"):
            continue


        print(f)
        #exit(1)

        hash_code = hashlib.md5(f.encode()).hexdigest()
        path_splits = f.split('/')
        n_splits = len(path_splits)
        out_file = (OUT_DIR + "/" + hash_code + "_" + path_splits[n_splits-2] + "_" + filename + ".png").replace(" ", "_").replace("'","")

        entry = [f, str(hash_code), out_file, 'na']
        try:
            audio = MP3(f)
            if audio.info.length < 130:
                #entry[3] = "Too short"
                #file_dic = append_entry(file_dic, entry)
                continue
        except:
            with open('/tmp/data_log.txt', 'a') as logfile:
                print('Skipping :' + in_file + ' because of exception in mutagen.', file=logfile)  # python 3.x
            continue

        file_dic = append_entry(file_dic, entry)



        #path_splits = f.split('/')
        #n_splits = len(path_splits)
        #out_file = (OUT_DIR + "/" + hash_code + "_" + path_splits[n_splits-2] + "_" + filename + ".png").replace(" ", "_").replace("'","")

        music_data, sample_rate = librosa.core.load(in_file, offset=60.0, duration=30.0)
        spectro_data = librosa.feature.melspectrogram(y=music_data, sr=sample_rate)

        #plt.figure(figsize=(10, 4))
        #librosa_display.specshow(librosa.power_to_db(spectro_data, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        #plt.colorbar(format='%+2.0f dB')
        #plt.title('Mel spectrogram')
        #plt.tight_layout()
        #plt.show()
        pylab.axis('off') # no axis
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) 
        librosa_display.specshow(librosa.power_to_db(spectro_data, ref=np.max))
        pylab.savefig(out_file, bbox_inches=None, pad_inches=0)
        pylab.close()

        write_entries(file_dic)
