import sys
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('--in-file', dest = 'in_file', default = 'labeling-hugo.csv', help = 'the file cith feature labels')
parser.add_argument('--in-path', dest = 'in_path', default = './data_unbalanced_new/', help = 'the folder containing spectrogram data')

args = parser.parse_args() 

labels = np.genfromtxt(args.in_file, delimiter = ',')

total_recordings = 17
total_female = 9
total_male = 8

class_predictions = []

for i in range(total_male + total_female):
    class_predictions.append(np.genfromtxt('predictions-' + str(i) + '.csv', delimiter = ',', dtype = None))

non_test = 9621

#last_id = -1

predictions = {}

for cp in class_predictions:
    for p in cp:

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
            
        # 9th female is missing!
        pv = int(p[1])
        if pv < 8:
            pv += 1
        else:
            pv += 2
            
        
        predictions[id] = pv
    
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

correct = 0

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
    
    if m_id == predictions[id]:
        correct += 1
    
result = np.stack(result)

#accuracy = result[:, 2] == result[:, 3] 

#print(np.mean(accuracy))

np.savetxt('features-recordings-prediction.csv', result, delimiter = ',', fmt = "%.10g")

print("correct:", correct)
