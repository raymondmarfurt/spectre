import os
import numpy as np
from shlex import quote
from subprocess import Popen, PIPE, STDOUT
import readchar
import time                                                                              
import signal

DIC_FILE="/var/tmp/DATA_2/DATA_DIC.csv"



def load_entries():
    dic = np.loadtxt(DIC_FILE, delimiter="|", dtype="str", encoding="utf-8",
            comments='###')
    return dic

def write_entries(dic):
        np.savetxt(DIC_FILE, dic, delimiter="|", fmt='%s', comments='###')         

file_dic = load_entries()




np.random.shuffle(file_dic)
 
for entry in file_dic:
    # skip already tagged
    if entry[3] in ['n','y']:
        continue

    number_of_y = (file_dic[:,3] == 'y').sum(0)
    number_of_n = (file_dic[:,3] == 'n').sum(0)
    print("Number of rated files y/n: " + str(number_of_y) + "/" +
            str(number_of_n))

    song_file = entry[0]
    #print(song_file)
    command = "play \"{}\" trim 60 30 ".format(song_file)
    p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True , preexec_fn=os.setsid)
    selection = 'o'
    while selection not in ['y','n', 's', 'x']:
        print("y/n/s/x:")
        selection = readchar.readchar()
        if selection == 'p':
            print(song_file)
            continue
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except:
            print("...")

        if selection == 'x':
            exit(0)
        if selection == 's':
            continue
        entry[3] = selection
        write_entries(file_dic)
