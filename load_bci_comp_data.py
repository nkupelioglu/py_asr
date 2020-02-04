from scipy.io import loadmat
data_folder = 'D:\\16_bci_competition_IV_1\\data\\'
from os import listdir
from os.path import isfile, join
bci_competition_files = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
#for i in range(len(bci_competition_files)):
#    print(join(data_folder,bci_competition_files[i]))
print(bci_competition_files)
bci_file = loadmat(join(data_folder,bci_competition_files[0]))
print(bci_file)
