






import numpy as np
import pandas as pd 
import glob
import json 

def get_evidence(path):

    f=open(path)
    data = json.load(f)

#    print(path, data["log_evidence"])
    return data["log_evidence"]



def parse_filename(f):
    return float(f.split('_')[4]) #the index is specific to these filenames


#def bayes_factor(f1,f2):
#	v1 = get_evidence(f1)
#	ev2 = get_evidence(f2)
#	return ev1 - ev2




list_of_pulsar_files = sorted(glob.glob("*pulsar*result.json"))
list_of_earth_files = sorted(glob.glob("*earth*result.json"))
list_of_null_files = sorted(glob.glob("*null*result.json"))

N = len(list_of_pulsar_files)




print(len(list_of_pulsar_files))
print(len(list_of_earth_files))
print(len(list_of_null_files))


pulsar_strings = []
for f in list_of_pulsar_files:
    h = f.split('_')[4]
    pulsar_strings.append(h)







#sys.exit()







#import sys

#try:
 #   assert len(list_of_pulsar_files) == len(list_of_earth_files) == len(list_of_null_files)
#except:
 #   print("File numbers don't match")

output_data = np.zeros((N,3))
for i in range(N):
    print(N - i)

    h = pulsar_strings[i]

    f2 = f'P3_canonical_bayes_h_{h}_model_pulsar_result.json'
    f1 = f'P3_canonical_bayes_h_{h}_model_earth_result.json'
    f0 = f'P3_canonical_bayes_h_{h}_model_null_result.json'


    assert parse_filename(f2) == parse_filename(f1) == parse_filename(f0)



    ev2 = get_evidence(f2)
    ev1 = get_evidence(f1)
    ev0 = get_evidence(f0)


    output_data[i,0] = parse_filename(f1)
    output_data[i,1] = ev1 - ev0 #earth term bayes factor
    output_data[i,2] = ev2 - ev0 #pulsar term bayes factor

    print(i,parse_filename(f1),ev1-ev0,ev2-ev0)



outfile = 'bayes_factors_n2000_v2'
np.save(outfile, output_data)












