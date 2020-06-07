# @Time    : 6/1/20 2:47 AM
# @Author  : 
# @FileName: output_pred.py

import pandas as pd
import numpy as np

# item_data = pd.read_csv('data/' + 'kdd_247' + '/underexpose_item_feat_Test.csv', index_col=False, sep='\,\[|\]\,\[|\]', engine='python', names=['item_id', 'txt_vec', 'img_vec'])
# item_data['txt_vec'] = item_data['txt_vec'].apply(lambda x: map(float,x.split(',')))
#
# item_data['img_vec'] = item_data['img_vec'].apply(lambda x: x.split(','))
# tmp = item_data['txt_vec'].values
# print("finish")

res_recommend = np.load('data/kdd_247/res_idx_old.npy', allow_pickle=True)





u_dict = np.load('data/kdd_247/u_dict.npy', allow_pickle=True).item()
u_dict_rev = np.load('data/kdd_247/u_dict_rev.npy', allow_pickle=True).item()
v_dict = np.load('data/kdd_247/v_dict.npy', allow_pickle=True).item()
v_dict_rev = np.load('data/kdd_247/v_dict_rev.npy', allow_pickle=True).item()

phase = 3
df_all = pd.DataFrame()
for p in range(phase):
    df_tmp = pd.read_csv(
        'data/kdd_247/test/underexpose_test_qtime-{}.csv'.format(p), sep=',', header=None,
        names=['user_id', 'timestamp'],
        dtype={'user_id': np.int32, 'timestamp': np.float64})
    df_all = df_all.append(df_tmp)

df_all_gt = pd.DataFrame()
for p in range(phase):
    df_tmp_gt = pd.read_csv(
        'data/kdd_247/test/underexpose_test_qtime_with_answer-{}.csv'.format(p), sep=',', header=None,
        names=['user_id', 'item_id', 'timestamp'],
        dtype={'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.float64})
    df_all_gt = df_all_gt.append(df_tmp_gt)

users = np.array(df_all['user_id'].values.tolist())
users_gt = np.array(df_all_gt['user_id'].values.tolist())
items_gt = np.array(df_all_gt['item_id'].values.tolist())
N = len(users_gt)

users_idx = np.array(list(map(u_dict.get, users)))
# recommend = res_recommend[users_idx]

cnt = 0
for i, val in enumerate(users_gt):
    u_idx = u_dict[val]
    v_idx = v_dict[items_gt[i]]
    item_pred = res_recommend[u_idx]
    if v_idx in item_pred[:50]:
        cnt += 1
print("prob:{:.2f}".format(cnt/float(N)))

prediction = np.zeros((len(users), 51), dtype=int)
for i, u_id in enumerate(users):
    prediction[i][0] = u_id
    item_pred = res_recommend[u_dict[u_id]][:50]
    prediction[i][1:] = np.array(list(map(v_dict_rev.get, item_pred)))

pd.DataFrame(prediction).to_csv("data/kdd_247/prediction.csv", header=None, index=None)

print("finish")