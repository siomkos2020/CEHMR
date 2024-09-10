import pandas as pd
import dill
from collections import defaultdict
import numpy as np

med_file = '../data/PRESCRIPTIONS.csv'
diag_file = '../data/DIAGNOSES_ICD.csv'
procedure_file = '../data/PROCEDURES_ICD.csv'

## Drug code mapping files
ndc2atc_file = '../data/ndc2atc_level4.csv'
cid_atc = '../data/drug-atc.csv'
ndc2rxnorm_file = '../data/ndc2rxnorm_mapping.txt'

## Drug-drug interactions
ddi_file = '../data/drug-DDI.csv'
cid_atc_file = '../data/drug-atc.csv'
voc_file = '../data/voc_final.pkl'
data_path = '../data/records_final.pkl'
TOPK = 40


def process_diag():
    diag_pd = pd.read_csv(diag_file, low_memory=False)
    diag_pd.dropna(inplace=True)
    diag_pd.drop(columns=['SEQ_NUM', 'ROW_ID'], inplace=True)  # 删除SEQ_NUM, ROW_ID列
    diag_pd.drop_duplicates(inplace=True)
    diag_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID'], inplace=True)
    return diag_pd.reset_index(drop=True)  # 对DataFrame重置索引


def filter_2000_most_diag(diag_pd):
    diag_count = diag_pd.groupby(by=['ICD9_CODE']).size().reset_index().rename(columns={0: 'count'}) \
        .sort_values(by=['count'], ascending=False).reset_index(drop=True)
    diag_pd = diag_pd[diag_pd['ICD9_CODE'].isin(diag_count.loc[:1999, 'ICD9_CODE'])]

    return diag_pd.reset_index(drop=True)


def process_med():
    med_pd = pd.read_csv(med_file, dtype={'NDC': 'category'}, low_memory=False)
    # filter
    med_pd.drop(columns=[
        'ROW_ID', 'DRUG_TYPE', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC', 'FORMULARY_DRUG_CD', 'GSN',
        'PROD_STRENGTH', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'FORM_UNIT_DISP',
        'ROUTE', 'ENDDATE', 'DRUG'
    ], axis=1, inplace=True)
    med_pd.drop(index=(med_pd[med_pd['NDC'] == '0'].index), axis=0, inplace=True)
    med_pd.fillna(method='pad', inplace=True)
    med_pd.dropna(inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd['ICUSTAY_ID'] = med_pd['ICUSTAY_ID'].astype('int64')
    med_pd['STARTDATE'] = pd.to_datetime(med_pd['STARTDATE'], format='%Y-%m-%d %H:%M:%S')
    med_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'], inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    def filter_first24hour_med(med_pd):
        med_pd_new = med_pd.drop(columns=['NDC'])
        med_pd_new = med_pd_new.groupby(by=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']).head([1]).reset_index(drop=True)
        med_pd_new = pd.merge(med_pd_new, med_pd, on=['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE'])
        med_pd_new = med_pd_new.drop(columns=['STARTDATE'])
        return med_pd_new

    med_pd = filter_first24hour_med(med_pd)
    med_pd.drop(columns=['ICUSTAY_ID'], inplace=True)
    med_pd.drop_duplicates(inplace=True)
    med_pd = med_pd.reset_index(drop=True)

    ## filter records where visit > 1
    def process_visit_lg2(med_pd):
        a = med_pd[['SUBJECT_ID', 'HADM_ID']].groupby(by='SUBJECT_ID')[
            'HADM_ID'].unique().reset_index()  # 将同一SUBJECT_ID下的信息合并至一个单元格
        a['HADM_ID_Len'] = a['HADM_ID'].map(lambda x: len(x))
        a = a[a['HADM_ID_Len'] > 1]
        return a

    med_pd_lg2 = process_visit_lg2(med_pd).reset_index(drop=True)
    med_pd = med_pd.merge(med_pd_lg2[['SUBJECT_ID']], on='SUBJECT_ID', how='inner')

    return med_pd.reset_index(drop=True)


def ndc2arc4(med_pd):
    with open(ndc2rxnorm_file, 'r') as f:
        ndc2rxnorm = eval(f.read())

    med_pd['RXCUI'] = med_pd['NDC'].map(ndc2rxnorm)
    med_pd.dropna(inplace=True)

    rxnorm2atc = pd.read_csv(ndc2atc_file)
    rxnorm2atc = rxnorm2atc.drop(columns=['YEAR', 'MONTH', 'NDC'])
    rxnorm2atc.drop_duplicates(subset=['RXCUI'], inplace=True)
    med_pd.drop(index=med_pd[med_pd['RXCUI'].isin([''])].index, axis=0, inplace=True)

    med_pd['RXCUI'] = med_pd['RXCUI'].astype('int64')
    med_pd = med_pd.reset_index(drop=True)
    med_pd = med_pd.merge(rxnorm2atc, on=['RXCUI'])
    med_pd.drop(columns=['NDC', 'RXCUI'], inplace=True)
    med_pd = med_pd.rename(columns={'ATC4': 'NDC'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: x[:4])
    med_pd = med_pd.drop_duplicates()
    med_pd = med_pd.reset_index(drop=True)
    return med_pd


def process_procedure():
    pro_pd = pd.read_csv(procedure_file, dtype={'ICD9_CODE': 'category'})
    pro_pd.drop(columns=['ROW_ID'], inplace=True)

    pro_pd.drop_duplicates(inplace=True)
    pro_pd.sort_values(by=['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM'], inplace=True)
    pro_pd.drop(columns=['SEQ_NUM'], inplace=True)
    pro_pd.drop_duplicates(inplace=True)
    pro_pd.reset_index(drop=True, inplace=True)

    return pro_pd


def process_all():
    ## 获取Medicine, Diagnoses, Procedure
    med_pd = process_med()
    med_pd = ndc2arc4(med_pd)

    diag_pd = process_diag()
    diag_pd = filter_2000_most_diag(diag_pd)

    pro_pd = process_procedure()

    ## Merge
    med_pd_key = med_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    diag_pd_key = diag_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()
    pro_pd_key = pro_pd[['SUBJECT_ID', 'HADM_ID']].drop_duplicates()

    combined_key = med_pd_key.merge(diag_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    combined_key = combined_key.merge(pro_pd_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    diag_pd = diag_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    med_pd = med_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    pro_pd = pro_pd.merge(combined_key, on=['SUBJECT_ID', 'HADM_ID'], how='inner')

    ## Flatten and merge
    diag_pd = diag_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index()
    med_pd = med_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['NDC'].unique().reset_index()
    pro_pd = pro_pd.groupby(by=['SUBJECT_ID', 'HADM_ID'])['ICD9_CODE'].unique().reset_index() \
        .rename(columns={'ICD9_CODE': 'PRO_CODE'})
    med_pd['NDC'] = med_pd['NDC'].map(lambda x: list(x))
    pro_pd['PRO_CODE'] = pro_pd['PRO_CODE'].map(lambda x: list(x))
    data = diag_pd.merge(med_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data = data.merge(pro_pd, on=['SUBJECT_ID', 'HADM_ID'], how='inner')
    data['NDC_Len'] = data['NDC'].map(lambda x: len(x))
    return data


def statistics(data):
    print('#patients ', data['SUBJECT_ID'].unique().shape)
    print('#clinical events ', len(data))

    diag = data['ICD9_CODE'].values
    med = data['NDC'].values
    pro = data['PRO_CODE'].values

    unique_diag = set([j for i in diag for j in list(i)])
    unique_med = set([j for i in med for j in list(i)])
    unique_pro = set([j for i in pro for j in list(i)])

    print('#diagnosis ', len(unique_diag))
    print('#med ', len(unique_med))
    print('#procedure', len(unique_pro))

    avg_diag = 0
    avg_med = 0
    avg_pro = 0
    max_diag = 0
    max_med = 0
    max_pro = 0
    cnt = 0
    max_visit = 0
    avg_visit = 0

    for subject_id in data['SUBJECT_ID'].unique():
        item_data = data[data['SUBJECT_ID'] == subject_id]
        x = []
        y = []
        z = []
        visit_cnt = 0
        for index, row in item_data.iterrows():
            visit_cnt += 1
            cnt += 1
            x.extend(list(row['ICD9_CODE']))
            y.extend(list(row['NDC']))
            z.extend(list(row['PRO_CODE']))
        x = set(x)
        y = set(y)
        z = set(z)
        avg_diag += len(x)
        avg_med += len(y)
        avg_pro += len(z)
        avg_visit += visit_cnt
        if len(x) > max_diag:
            max_diag = len(x)
        if len(y) > max_med:
            max_med = len(y)
        if len(z) > max_pro:
            max_pro = len(z)
        if visit_cnt > max_visit:
            max_visit = visit_cnt

    print('#avg of diagnoses ', avg_diag / cnt)
    print('#avg of medicines ', avg_med / cnt)
    print('#avg of procedures ', avg_pro / cnt)
    print('#avg of vists ', avg_visit / len(data['SUBJECT_ID'].unique()))

    print('#max of diagnoses ', max_diag)
    print('#max of medicines ', max_med)
    print('#max of procedures ', max_pro)
    print('#max of visit ', max_visit)


def construct_data():
    data = process_all()
    print(data.head())
    statistics(data)
    data.to_pickle('../data/data_final.pkl')
    df = pd.read_pickle('../data/data_final.pkl')
    diag_voc, med_voc, pro_voc = create_str_token_mapping(df)
    records = create_patient_record(df, diag_voc, med_voc, pro_voc)
    print(len(diag_voc.id2word))  # 1958
    print(len(med_voc.id2word))  # 145
    print(len(pro_voc.id2word))  # 1426


class Voc(object):
    def __init__(self):
        self.id2word = {}
        self.word2id = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2id:
                self.id2word[len(self.word2id)] = word
                self.word2id[word] = len(self.word2id)


def create_str_token_mapping(df):
    diag_voc = Voc()
    med_voc = Voc()
    pro_voc = Voc()

    for index, row in df.iterrows():
        diag_voc.add_sentence(row['ICD9_CODE'])
        med_voc.add_sentence(row['NDC'])
        pro_voc.add_sentence(row['PRO_CODE'])

    dill.dump(obj={'diag_voc': diag_voc, 'med_voc': med_voc, 'pro_voc': pro_voc},
              file=open('../data/voc_final.pkl', 'wb'))
    return diag_voc, med_voc, pro_voc


def create_patient_record(df, diag_voc, med_voc, pro_voc):
    records = []
    for subject_id in df['SUBJECT_ID'].unique():
        item_df = df[df['SUBJECT_ID'] == subject_id]
        patient = []
        for index, row in item_df.iterrows():
            admission = list()
            admission.append([diag_voc.word2id[i] for i in row['ICD9_CODE']])
            admission.append([pro_voc.word2id[i] for i in row['PRO_CODE']])
            admission.append([med_voc.word2id[i] for i in row['NDC']])
            patient.append(admission)
        records.append(patient)
    dill.dump(obj=records, file=open('../data/records_final.pkl', 'wb'))

    return records




records = dill.load(open('../data/records_final.pkl', 'rb'))
cid2atc_dic = defaultdict(set)

med_voc = dill.load(open(voc_file, 'rb'))['med_voc']
med_voc_size = len(med_voc.id2word)
med_unique_word = [med_voc.id2word[i] for i in range(med_voc_size)]
atc3_atc4_dic = defaultdict(set)

for item in med_unique_word:
    atc3_atc4_dic[item[:4]].add(item)

with open(cid_atc_file, 'r') as f:
    for line in f:
        line_ls = line[:-1].split(',')
        cid = line_ls[0]
        atcs = line_ls[1:]
        for atc in atcs:
            if len(atc3_atc4_dic[atc[:4]]) != 0:
                cid2atc_dic[cid].add(atc[:4])

#print(cid2atc_dic)
ddi_df = pd.read_csv(ddi_file)
ddi_most_pd = ddi_df.groupby(by=['Polypharmacy Side Effect', 'Side Effect Name']).size().reset_index()\
                    .rename(columns={0:'count'}).sort_values(by=['count'], ascending=False).reset_index(drop=True)
ddi_most_pd = ddi_most_pd.iloc[-TOPK:, :]        #取频率最高的40种不良反应
filter_ddi_df = ddi_df.merge(ddi_most_pd[['Side Effect Name']], how='inner', on=['Side Effect Name'])
ddi_df = filter_ddi_df[['STITCH 1', 'STITCH 2']].drop_duplicates().reset_index(drop=True)

# weighted ehr adj
ehr_adj = np.zeros((med_voc_size, med_voc_size))
for patient in records:
    for adm in patient:
        med_set = adm[2]
        for i, med_i in enumerate(med_set):
            for j, med_j in enumerate(med_set):
                if j <= i:
                    continue
                ehr_adj[med_i, med_j] = 1
                ehr_adj[med_j, med_i] = 1

print(ehr_adj)
dill.dump(ehr_adj, open('../data/ehr_adj_final.pkl', 'wb'))


## ddi adj
ddi_adj = np.zeros((med_voc_size, med_voc_size))
for index, row in ddi_df.iterrows():
    cid1 = row['STITCH 1']
    cid2 = row['STITCH 2']

    for atc_i in cid2atc_dic[cid1]:
        for atc_j in cid2atc_dic[cid2]:
            for i in atc3_atc4_dic[atc_i]:
                for j in atc3_atc4_dic[atc_j]:
                    if med_voc.word2id[i] != med_voc.word2id[j]:
                        ddi_adj[med_voc.word2id[i], med_voc.word2id[j]] = 1
                        ddi_adj[med_voc.word2id[j], med_voc.word2id[i]] = 1

print(ddi_adj)
dill.dump(ddi_adj, open('../data/ddi_A_final.pkl', 'wb'))
print('complete!')