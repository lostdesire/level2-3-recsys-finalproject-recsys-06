import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import sparse
import pandas as pd
import json

data_dir = '../../files'


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=False)
    count = playcount_groupbyid.size()

    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    """
    특정한 횟수 이상의 리뷰가 존재하는(사용자의 경우 min_uc 이상, 아이템의 경우 min_sc이상)
    데이터만을 추출할 때 사용하는 함수입니다.
    현재 데이터셋에서는 결과적으로 원본그대로 사용하게 됩니다.
    """
    if min_sc > 0:
        itemcount = get_count(tp, 'item_id')
        tp = tp[tp['item_id'].isin(itemcount[itemcount['size'] >= min_sc]['item_id'])]

    if min_uc > 0:
        usercount = get_count(tp, 'user_id')
        tp = tp[tp['user_id'].isin(usercount[usercount['size'] >= min_uc]['user_id'])]

    usercount, itemcount = get_count(tp, 'user_id'), get_count(tp, 'item_id')

    unique_uid = usercount.user_id
    unique_iid = itemcount.item_id

    np.random.seed(42)
    uid_perm = np.random.permutation(unique_uid.size)
    iid_perm = np.random.permutation(unique_iid.size)
    unique_uid, unique_iid = unique_uid[uid_perm], unique_iid[iid_perm]

    return tp, usercount, itemcount, unique_uid, unique_iid


def split_train_test_proportion(data, test_prop=0.2):
    """
    훈련된 모델을 이용해 검증할 데이터를 분리하는 함수입니다.
    100개의 액션이 있다면, 그중에 test_prop 비율 만큼을 비워두고, 그것을 모델이 예측할 수 있는지를
    확인하기 위함입니다.
    """

    data_grouped_by_user = data.groupby('user_id')
    tr_list, te_list = list(), list()

    np.random.seed(42)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te

def numerize(tp, profile2id, show2id):
    uid = tp['user_id'].apply(lambda x: profile2id[x])
    iid = tp['item_id'].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'iid': iid}, columns=['uid', 'iid'])


def split_uid(args, unique_uid):

    # Split Train/Validation/Test User Indices
    n_users = unique_uid.size #31360
    n_heldout_users = args.heldout_users

    tr_users = unique_uid[:(n_users - n_heldout_users * 2)]
    vd_users = unique_uid[(n_users - n_heldout_users * 2): (n_users - n_heldout_users)] # val 3000명
    te_users = unique_uid[(n_users - n_heldout_users):] # test 3000명

    return tr_users, vd_users, te_users

def split_data(args, unique_uid, unique_iid, raw_data):

    tr_users, vd_users, te_users = split_uid(args, unique_uid)
    print("훈련 데이터에 사용될 사용자 수:", len(tr_users))
    print("검증 데이터에 사용될 사용자 수:", len(vd_users))
    print("테스트 데이터에 사용될 사용자 수:", len(te_users))


    ##훈련 데이터에 해당하는 아이템들
    train_plays = raw_data.loc[raw_data['user_id'].isin(tr_users)]

    show2id = dict((int(iid), int(i)) for (i, iid) in enumerate(unique_iid))
    profile2id = dict((int(pid), int(i)) for (i, pid) in enumerate(unique_uid))

    if not os.path.exists(args.pro_dir):
        os.makedirs(args.pro_dir)

    with open(os.path.join(args.pro_dir, 'unique_iid.txt'), 'w') as f:
        for iid in unique_iid:
            f.write('%s\n' % iid)

    #Validation과 Test에는 input으로 사용될 tr 데이터와 정답을 확인하기 위한 te 데이터로 분리되었습니다.
    vad_plays = raw_data.loc[raw_data['user_id'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['item_id'].isin(unique_iid)]
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data['user_id'].isin(te_users)]
    test_plays = test_plays.loc[test_plays['item_id'].isin(unique_iid)]
    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    return train_plays, vad_plays_tr, vad_plays_te, test_plays_tr, test_plays_te, show2id, profile2id


def numerize_write(args, profile2id, show2id, raw_data, train_plays, vad_plays_tr, vad_plays_te, test_plays_tr, test_plays_te):

    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(args.pro_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(args.pro_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(args.pro_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(args.pro_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(args.pro_dir, 'test_te.csv'), index=False)

    inf_data = raw_data[['user_id','item_id']].copy()
    inf_data = numerize(inf_data, profile2id, show2id)
    inf_data.to_csv(os.path.join(args.pro_dir, 'inference.csv'), index=False)

    id2show = {v:k for k,v in show2id.items()}
    id2profile = {v:k for k,v in profile2id.items()}

    show2id = json.dumps(show2id)
    profile2id = json.dumps(profile2id)
    id2show = json.dumps(id2show)
    id2profile = json.dumps(id2profile)
    
    if not os.path.exists(os.path.join(args.pro_dir,'json_id')):
        os.makedirs(os.path.join(args.pro_dir,'json_id'))

    with open(os.path.join(args.pro_dir,'json_id/profile2id.json'), 'w') as json_file :
        json.dump(profile2id, json_file)
    
    with open(os.path.join(args.pro_dir,'json_id/show2id.json'), 'w') as json_file :
        json.dump(show2id, json_file)

    with open(os.path.join(args.pro_dir,'json_id/id2show.json'), 'w') as json_file :
        json.dump(id2show, json_file)
    
    with open(os.path.join(args.pro_dir,'json_id/id2profile.json'), 'w') as json_file :
        json.dump(id2profile, json_file)

## TRAIN/VAL = 9:1 version
def split_uid2(args, unique_uid):

    # Split Train/Validation/Test User Indices
    n_users = unique_uid.size #31360
    n_heldout_users = args.heldout_users

    tr_users = unique_uid[:(n_users - n_heldout_users)]
    vd_users = unique_uid[(n_users - n_heldout_users):] # val 3000명

    return tr_users, vd_users


def split_data2(args, unique_uid, raw_data):

    tr_users, vd_users = split_uid2(args, unique_uid)
    print("훈련 데이터에 사용될 사용자 수:", len(tr_users))
    print("검증 데이터에 사용될 사용자 수:", len(vd_users))


    ##훈련 데이터에 해당하는 아이템들
    train_plays = raw_data.loc[raw_data['user_id'].isin(tr_users)]

    ##아이템 ID
    unique_iid = pd.unique(train_plays['item_id'])

    show2id = dict((int(iid), int(i)) for (i, iid) in enumerate(unique_iid))
    profile2id = dict((int(pid), int(i)) for (i, pid) in enumerate(unique_uid))

    if not os.path.exists(args.pro_dir):
        os.makedirs(args.pro_dir)

    with open(os.path.join(args.pro_dir, 'unique_iid.txt'), 'w') as f:
        for iid in unique_iid:
            f.write('%s\n' % iid)

    #Validation과 Test에는 input으로 사용될 tr 데이터와 정답을 확인하기 위한 te 데이터로 분리되었습니다.
    vad_plays = raw_data.loc[raw_data['user_id'].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays['item_id'].isin(unique_iid)]
    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    return train_plays, vad_plays_tr, vad_plays_te, show2id, profile2id


def numerize_write2(args, profile2id, show2id, raw_data, train_plays, vad_plays_tr, vad_plays_te):

    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_json(os.path.join(args.pro_dir, 'train.json'), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_json(os.path.join(args.pro_dir, 'validation_tr.json'), index=False)

    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_json(os.path.join(args.pro_dir, 'validation_te.json'), index=False)

    inf_data = raw_data[['user_id','item_id']].copy()
    inf_data = numerize(inf_data, profile2id, show2id)
    inf_data.to_json(os.path.join(args.pro_dir, 'inference.json'), index=False)

    id2show = {v:k for k,v in show2id.items()}
    id2profile = {v:k for k,v in profile2id.items()}

    show2id = json.dumps(show2id)
    profile2id = json.dumps(profile2id)
    id2show = json.dumps(id2show)
    id2profile = json.dumps(id2profile)
    
    with open('json_id/show2id_2.json', 'w') as json_file :
        json.dump(show2id, json_file)
    
    with open('json_id/profile2id_2.json', 'w') as json_file :
        json.dump(show2id, json_file)

    with open('json_id/id2show_2.json', 'w') as json_file :
        json.dump(id2show, json_file)
    
    with open('json_id/id2profile_2.json', 'w') as json_file :
        json.dump(id2profile, json_file)