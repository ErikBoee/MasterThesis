import numpy as np
import os

newpath = r'../test/New_folder'
if not os.path.exists(newpath):
    os.makedirs(newpath)

eriko = {"name": "Erich",
         "age": 24}
np.save("New_folder/don", eriko, allow_pickle=True)
