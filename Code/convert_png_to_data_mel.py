from scipy import misc
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import numpy as np
import h5py

DIC_FILE="/var/tmp/DATA_2/DATA_DIC.csv"
DATA_PATH="/var/tmp/DATA_2/DATA/"
H5_OUTPUT="/var/tmp/DATA_2/spectro_data_mel.hdf5"


def load_entries():
    dic = np.loadtxt(DIC_FILE, delimiter="|", dtype="str", comments="###")
    return dic


def cat_to_num(category):
    if category == 'y':
        return 0
    if category == 'n':
        return 1
    return -1


file_dic = load_entries()

 
training_data = []
training_labels = []


for entry in file_dic:
    image_path = entry[2]
    print(image_path)
    i_cat = cat_to_num(entry[3])
    if i_cat > -1:
        image = imageio.imread(image_path)
        img_data = np.asarray(image,dtype="uint8")[:,:,:3]
        training_data.append(img_data)
        training_labels.append(i_cat)
        #plt.imshow(img_data,interpolation="bicubic")
        #plt.show()

with h5py.File(H5_OUTPUT, "w") as f:
    f.create_dataset('X', data=np.asarray(training_data))
    f.create_dataset('Y', data=np.asarray(training_labels))
    f.close()
