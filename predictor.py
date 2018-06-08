import sys
import os
import json
import numpy as np
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

print ("Importing modules...")
import modules
# print ("Done")

##################################################

print ("Reading data from disk...", end=' ')
sys.stdout.flush()

df = pd.read_parquet('./datasets/Metal_all_20180601.parquet')
seqs = np.array(df.sequence)
metals = np.array(df.ligandId)
fingerprints = np.array(df.fingerprint)

print ("Done")

##################################################

print ("Using FOFE encoder...", end=' ')
sys.stdout.flush()
metal_dict = {}
with open("./dictionaries/metal_dict", 'r') as fp:
        metal_dict = json.load(fp)
        
num_to_metal = {e: k for k, e in metal_dict.items()}
print ("Done")

##################################################
print("Loading metal_predictor...", end=' ')

from keras.models import model_from_json
json_file = open('./models/metal_predict.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

metal_predictor = model_from_json(loaded_model_json)
metal_predictor.load_weights("./models/metal_predict.h5")
print ("Done")

##################################################
factor = 2.33
def threshold_func(y_in, factor):
    y_out = np.zeros_like(y_in)
    for i in range(y_in.shape[0]):
        th= np.mean(y_in[i]) + factor * np.std(y_in[i])
        y_out[i] = (y_in[i] > th)
    return y_out

print ("Threshold factor set to " + str(factor))
print ("--------------------------------------------------")

choice = 0
if len(sys.argv) == 1:
    choice = np.random.randint(58207)
    print ("No input is provided. Randomly choose index...[" + str(choice) + "]")

else:
    choice = int(sys.argv[1])
    print ("Choose index [" + str(choice) + "]")
if choice < 0 or choice > 58206:
    sys.exit("Index must be within [0, 58206]")

print ("--------------------------------------------------")
if len(seqs[choice]) > 60:
    print ("The seuqnce is [" + seqs[choice][:30] + "..." + seqs[choice][len(seqs[choice])-30:] + "]\n")
else:
    print ("The seuqnce is [" + seqs[choice] + "]\n")
metal_out = metal_predictor.predict(modules.FOFE(seqs[choice]))

max_index = np.argmax(metal_out)
metal = num_to_metal[max_index]

print ("This sample is binded to [" + metal + "]")
print ("            Ground truth [" + metals[choice] + "]\n")

json_file = open('./models/' + metal + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

MBS_predictor = model_from_json(loaded_model_json)
MBS_predictor.load_weights('./models/' + metal + '.h5')

MBS_out = MBS_predictor.predict(modules.FOFE(seqs[choice]))
MBS_OneHot = threshold_func(MBS_out, factor)
MBS = [np.where(e==1)[0] for e in MBS_OneHot][0]
print ("This sample has [", end="")
print (*MBS, sep=",", end="")
print ("] binding sites")
print ("   Ground truth [", end="")
print (*fingerprints[choice], sep=",", end="")
print ("]")