import sys
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--in-file', dest = 'in_file', default = 'labeling-hugo.csv', help = 'the file cith feature labels')
parser.add_argument('--in-path', dest = 'in_path', default = './data_unbalanced_new/', help = 'the folder containing spectrogram data')

args = parser.parse_args() 

labels = np.genfromtxt(args.in_file, delimiter = ',')
predictions_female = np.genfromtxt('predictions-0.csv', delimiter = ',', dtype = None)
predictions_male = np.genfromtxt('predictions-1.csv', delimiter = ',', dtype = None)

print(predictions_male);

male2permuted_raw = np.genfromtxt('male-2-permuted.csv', delimiter = ',', dtype = np.bytes_)
female2permuted_raw = np.genfromtxt('female-2-permuted.csv', delimiter = ',', dtype = np.bytes_)

male2permuted = {}
for i in male2permuted_raw:
    male2permuted[i[0]] = i[1]

female2permuted = {}
for i in female2permuted_raw:
    female2permuted[i[0]] = i[1]

#print(predictions)

#predictions_raw = sorted(predictions, key=lambda x: x[0])
#predictions = np.zeros((len(labels)), dtype = np.int)

non_test = 9621

#last_id = -1

predictions = {}

for p in predictions_male:

    fn = p[0]

    '''
    if int(fn[5:(5 + 9)]) == 3155:    
        print(fn)
    '''
    
    id = fn[1:(1 + 4 + 9)] + b".csv"
    #id = male2permuted[id]
    t = id[0:4]
    #print(t)
    id = int(id[4:(4+9)])
    if t == b"test":
        id = id + non_test

    #print(id)
        
    predictions[id] = int(p[1])

for p in predictions_female:

    fn = p[0]
        
    id = fn[1:(1 + 4 + 9)] + b".csv"
    #id = female2permuted[id]
    t = id[0:4]
    id = int(id[4:(4+9)])
    if t == b"test":
        id = id + non_test
        
    predictions[id] = int(p[1])

    
'''    
    print("this id:", id)
    
    if id != last_id + 1:
        print("id:", id)
    
    last_id = id    

    if t == 'test':
        id = id + non_test
        
    predictions[id] = int(p[1])
'''
        
#print(predictions)

#sys.exit()

#print(labels)

result = []

for x in labels:
    #print(x)
    id = int(x[0])
    
    '''
    if id < non_test:
        basic_data_fn = os.path.join(args.in_path, "data" + str(id).zfill(9))
    else:
        basic_data_fn = os.path.join(args.in_path, "test" + str(id - non_test).zfill(9))
    '''

    basic_data_fn = os.path.join(args.in_path, "data" + str(id).zfill(9))

    
    id_fn = basic_data_fn + ".id"
    data_fn = basic_data_fn + ".csv"
    
    id_file = open(id_fn, "r") 
    m_id = id_file.read()     
    
    if m_id.startswith("Rfem_Afem"):
        m_id = int(m_id[9:])
    else:
        m_id = int(m_id[10:]) + 10
    
    
    #print(basic_data_fn)

    basic_data = np.genfromtxt(data_fn, delimiter = ',')
    
    if not (id in predictions):
        print("not predicted:", id)   
        continue
    
    #print(basic_data)
    basic_data = np.concatenate([[id, m_id], [basic_data[0]], [predictions[id]], basic_data[1:]])
    r = np.concatenate([basic_data, x[1:]])
    result.append(r)
    
result = np.stack(result)

#accuracy = result[:, 2] == result[:, 3] 

#print(np.mean(accuracy))

np.savetxt('features.csv', result, delimiter = ',', fmt = "%.10g")
