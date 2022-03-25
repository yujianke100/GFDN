import pandas as pd
import os
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
import numpy as np

def save_f_e(df):
    df.sort_values(by="t",axis=0,ascending=False,inplace=True)
    u_feature_dict = dict()
    i_feature_dict = dict()
    for row in tqdm(df.itertuples()):
        u = int(row[3])
        i = int(row[4])
        if(u not in u_feature_dict or i not in i_feature_dict):
            f = row[5].split(',')
            if(u not in u_feature_dict):
                u_feature_dict[u] = np.array(list(map(float, f[72:148])))
            if(i not in i_feature_dict):
                i_feature_dict[i] = np.array(list(map(float, f[:72])))
    u_features = []
    i_features = []
    print('save u feature')
    for i in trange(len(u_feature_dict)):
        u_features.append(u_feature_dict[i])
    u_features = np.array(u_features)
    np.save(root_path + '/bdt_u_features.npy', u_features)
    print('save i feature')
    for i in trange(len(i_feature_dict)):
        i_features.append(i_feature_dict[i])
    i_features = np.array(i_features)
    np.save(root_path + '/bdt_i_features.npy', i_features)

    df = df[['u','i','l']]
    df['u'] = df['u'].astype('int')
    df['i'] = df['i'].astype('int')
    df['l'] = df['l'].astype('int')

    train_data, test_data = train_test_split(df, train_size=0.9, stratify=df['l'], random_state=10)
    
    train_data.sort_values(by=['u','i'],axis=0,ascending=[True, True],inplace=True)
    test_data.sort_values(by=['u','i'],axis=0,ascending=[True, True],inplace=True)
    train_data.to_csv(root_path+'/bdt_edges_train.csv', header=0, index=0, sep='\t', columns=['u','i','l'])
    test_data.to_csv(root_path+'/bdt_edges_test.csv', header=0, index=0, sep='\t', columns=['u','i','l'])

def save_reindex(df, save_name):
    print('reindex...')
    user_list = list(set(df['u']))
    item_list = list(set(df['i']))
    user_list.sort()
    item_list.sort()
    user_dict = dict()
    item_dict = dict()
    for i, id_ in enumerate(user_list):
        user_dict[id_] = i
    for i, id_ in enumerate(item_list):
        item_dict[id_] = i
    print('saving...')
    with open(save_name, 'w') as file_:
        for row in tqdm(df.itertuples()):
            _, uuid, t, u, i, f, l = row
            file_.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(uuid, t.replace('-','').replace(' ', '').replace(':',''), user_dict[u], item_dict[i], f, l))

test_num = None
root_path, _ = os.path.split(os.path.abspath(__file__))
print('reading...')
df = pd.read_csv(root_path+'/antispam_bdt_final_round_train_0.csv', names=['id', 't', 'u', 'i', 'f','l'], delimiter='\t', nrows=test_num, dtype=str)
test = pd.read_csv(root_path+'/antispam_bdt_final_round_pred_0_samples.csv', names=['id', 't', 'u', 'i', 'f'], delimiter='\t', nrows=test_num, dtype=str)
test_l = pd.read_csv(root_path+'/antispam_bdt_final_round_pred_0_labels.csv', names=['id', 'l'], delimiter='\t', nrows=test_num, dtype=str)
test = pd.merge(test,test_l,how='right',on='id')
df = pd.concat([df, test])
print('sorting...')
df.sort_values(by="t",axis=0,ascending=False,inplace=True)
print('dropping...')
df.drop_duplicates(['u', 'i', 'l'], keep="first", inplace=True)
print('washing...')
df = df.loc[(df['u'].str.isdigit() & df['i'].str.isdigit())]
print('relabel...')
u_degrees = df['u'].value_counts()
one_degree_u = u_degrees[u_degrees == 1].axes[0].values
print(sum(df['u'].isin(one_degree_u) * df['l'] == '1'))
df.loc[df['u'].isin(one_degree_u) * df['l'] == '1', ['l']] = '0'

save_f_e(df)

df = pd.read_csv(root_path+'/bdt_edges_train.csv', names=['u', 'i','label'], delimiter='\t', dtype=int)
test_data = pd.read_csv(root_path+'/bdt_edges_test.csv', names=['u', 'i','label'], delimiter='\t', dtype=int)

test_without_label = test_data[test_data['label'] == -1]

df = pd.concat((df, test_without_label))

test_data = test_data[test_data['label'] != -1]
df.to_csv(root_path + '/train.txt', sep='\t', header=0, index=0)
test_data.to_csv(root_path + '/test.txt', sep='\t', header=0, index=0)