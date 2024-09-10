import dill
import os
import copy

"""This script is used for generating data for CEHMR from records_final.pkl"""

voc_path = '../data/voc_final.pkl'

# 1. Building medication vocabulary of ATC-1 to ATC-2.
voc = dill.load(open(voc_path, 'rb'))
diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
med_h1, med_h2 = {}, {}
for idx in med_voc.id2word:
    if med_voc.id2word[idx][:-1] not in med_h1:
        med_h1[med_voc.id2word[idx][:-1]] = len(med_h1)
    if med_voc.id2word[idx][:1] not in med_h2:
        med_h2[med_voc.id2word[idx][:1]] = len(med_h2)

if not os.path.exists('../data/hire_data'):
    os.mkdir('../data/hire_data')

dill.dump((med_h1, med_h2), open('../data/hire_data/med_h.pkl', 'wb'))

# 2. Building train, dev, test dataset from records_final.pkl
all_data = dill.load(open('../data/records_final.pkl', 'rb'))
split_point = int(2*len(all_data) / 3)

train_data = all_data[:split_point]
test_data = all_data[split_point:]
dev_split = int(len(test_data) / 2)
dev_data, test_data = test_data[:dev_split], test_data[dev_split:]

def add_hire_label(data):
    new_data = []
    for patient in data:
        new_patient = []
        for adm in patient:
            adm.append(sorted(list(set([med_h1[med_voc.id2word[x][:-1]] for x in adm[2]]))))
            adm.append(sorted(list(set([med_h2[med_voc.id2word[x][:1]] for x in adm[2]]))))
            new_patient.append(copy.deepcopy(adm))
        new_data.append(new_patient)

    return new_data

train_data = add_hire_label(train_data)
dev_data = add_hire_label(dev_data)
test_data = add_hire_label(test_data)

dill.dump(train_data, open('../data/hire_data/data_train.pkl', 'wb'))
dill.dump(dev_data, open('../data/hire_data/data_eval.pkl', 'wb'))
dill.dump(test_data, open('../data/hire_data/data_test.pkl', 'wb'))


