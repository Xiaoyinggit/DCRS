import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import csv

def read_user_item_features(user_file_path,item_file_path, features):
    """ Read features from the given file. """
    if len(features) > 0:
        raise AssertionError
    i = len(features)
    user_file = np.load(user_file_path, allow_pickle=True).item()
    item_file = np.load(item_file_path, allow_pickle=True).item()
     
    # add prefix on UID and ItemID to prevent UID and ItemID all starting from 0. 
    # First map userID/itemID, so that user/item node numbering are consecutive
    for UID in user_file:
        id_f = 'U_'+str(user_file[UID][0][0])
        user_file[UID][0][0] = id_f
        if id_f not in features:
            features[id_f] = i
            i+=1
    for ItemID in item_file:
        id_f = 'I_'+str(item_file[ItemID][0][0])
        item_file[ItemID][0][0] = id_f
        if id_f not in features:
            features[id_f] = i
            i+=1

    # map feature 
    for UID in user_file:
        feat = user_file[UID][0]
        for f in feat:
            if f not in features:
                features[f] = i
                i += 1
    for ItemID in item_file:
        feat = item_file[ItemID][0]
        for f in feat:
            if f not in features:
                features[f] = i
                i += 1
    return user_file, item_file, features

def read_features(file_path, features):
    """ Read features from the given file. """
    i = len(features)
    file = np.load(file_path, allow_pickle=True).item()

    # Let userID/itemID map first
    for ID in file:
        id_f = file[ID][0][0]
        if id_f not in features:
            features[id_f] =i
            i+=1

    for ID in file:
        feat = file[ID][0]
        for f in feat:
            if f not in features:
                features[f] = i
                i += 1

    return file, features

def map_features_din(user_feature_path, item_feature_path, num_groups):
    """
      --user_feature_path, uID: [uFeat, uFeat_val, uHistseq]
      --item_feature_path: itemID:[ID, item_cat]
    """
    features={}
    user_feature, features = read_features(user_feature_path, features)
    print('number of users: {}'.format(len(user_feature)))
    print('number of features: {}'.format(len(features)))
    mapped_user_features = {}
    nUF = 0
    for userID in user_feature.keys():
        feat = user_feature[userID][0]
        nUF = max(nUF, len(feat))
        mapped_feat = []
        for f in feat:
            mapped_feat.append(features[f])

        if(abs(sum(user_feature[userID][1])- len(user_feature[userID][1]))>0.01):
            #make sure all user feature val =1
            raise AssertionError
        mapped_user_features[userID] = [mapped_feat, [],user_feature[userID][2]]
    

    # make sure itemID is indexed 
    item_features = np.load(item_feature_path, allow_pickle=True).item()
    num_items = len(item_features)
    item_cat_map =[]
    for i in range(num_items):
        item_cat_map.append(item_features[i][1][-num_groups:])
    print('[map_features] nUF: %d'%nUF)
    return mapped_user_features, item_cat_map, len(features), nUF

def map_features(user_feature_path, item_feature_path):
    """ Get the number of existing features of users and items."""
    features = {}
    user_feature, item_feature, features = read_user_item_features(user_feature_path, item_feature_path,  features)
    #item_feature, features = read_features(item_feature_path, features)
    print("number of users: {}".format(len(user_feature)))
    print("number of items: {}".format(len(item_feature)))
    print("number of features: {}".format(len(features)))
    num_UF, num_IF = 0, 0
    num_user, num_item = len(user_feature), len(item_feature)
    mapped_user_feature = {}
    for userID in user_feature:
        feat = user_feature[userID][0]
        num_UF = max(num_UF, len(feat))
        mapped_feat = []
        for f in feat:
            mapped_feat.append(features[f])
        mapped_user_feature[userID] = [mapped_feat, user_feature[userID][1]]

    mapped_item_feature = {}
    for itemID in item_feature:
        feat = item_feature[itemID][0]
        num_IF = max(num_IF, len(feat))
        mapped_feat = []
        for f in feat:
            mapped_feat.append(features[f])
        mapped_item_feature[itemID] = [mapped_feat, item_feature[itemID][1]]
    print('[data_utils.map_fature] numUF: %d, numIF:%d, num_user:%d'%(num_UF, num_IF, num_user))

    return mapped_user_feature, mapped_item_feature, len(features), num_UF, num_IF, num_user, num_item,  features 

def loadData(path):
    file = np.load(path, allow_pickle=True).tolist()
    return file 

def loadTestData(path, uid_map=None):
    """load data for testing and validation."""
    file = np.load(path, allow_pickle=True).tolist()
    num_pos, num_all = 0, 0
    user_dp = {}
    for f in file:
        if len(f) >3:
          uID, itemID, label, user_hist_seq  = f[0], f[1], f[2], f[3]
        else:
          uID, itemID, label = f[0], f[1], f[2]
          user_hist_seq = []
        if uid_map is not None:
            uID = uid_map[uID]
        user_dp[uID] = user_dp.get(uID, [])
        user_dp[uID].append([itemID, label, user_hist_seq])
        num_pos +=label
        num_all += 1
    print('~~~~~~~~~~~', path)
    print('num_pos:%d, num_all:%d'%(num_pos, num_all))
    return user_dp

class DINData(data.Dataset):
    """Construct the DIN training dataset"""
    def __init__(self, train_path, user_feature, item_cat_map, max_hist_len=50,  num_groups=None):
        super(DINData, self).__init__()
        self.user_feature = user_feature
        self.item_cat_map = item_cat_map
        self.num_groups = num_groups
        self.max_hist_len = 50

        self.labels = []
        self.user_feat = []
        self.item_ID = []
        self.user_hist_seq = []
        self.user_hist_len = []

        # construct data 
        self.train_list = np.load(train_path, allow_pickle=True).tolist()
        self.train_dict = {}
        for pair in self.train_list:
            userID, itemID, label, user_hist_seq = pair 
            self.train_dict[userID] = self.train_dict.get(userID, [])
            self.train_dict[userID].append([itemID,label])

            self.user_feat.append(np.array(self.user_feature[userID][0]))
            self.item_ID.append(itemID)
            cur_user_seq_l = user_hist_seq
            cur_hist_len = len(cur_user_seq_l)
            seq_len_mask = [1]*cur_hist_len


            if len(cur_user_seq_l) < self.max_hist_len:
                cur_user_seq_l.extend([0]*(self.max_hist_len - cur_hist_len))
                seq_len_mask.extend([0]*(self.max_hist_len - cur_hist_len))  
            
            self.user_hist_len.append(np.array(seq_len_mask))
            self.user_hist_seq.append(np.array(cur_user_seq_l))
            self.labels.append(float(label))
        print('[DIN_DATA]train_data: %d, pos_data:%d'%(len(self.labels), np.sum(np.array(self.labels, dtype=np.float))))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        user_feat = self.user_feat[idx]
        item_id = self.item_ID[idx]
        user_hist_seq = self.user_hist_seq[idx]
        user_hist_len = self.user_hist_len[idx]
        label = self.labels[idx]

        return user_feat, item_id, user_hist_seq, user_hist_len, label



class FMData(data.Dataset):
    """ Construct the FM training dataset."""
    def __init__(self, train_path, user_feature, item_feature, user_hist_cat_file=None,  num_groups=None,  negative_num=1, model_name=None):
        super(FMData, self).__init__()
        self.user_feature = user_feature
        self.item_feature = item_feature
        self.negative_num = negative_num
        self.num_groups = num_groups
        self.model_name = model_name

        # existed pos& negs
        self.known_labels = []
        self.known_user_features = []
        self.known_user_feature_values = []
        self.known_item_features = []
        self.known_item_feature_values = []
        self.known_user_hist_cat_dist =[]

        self.user_hist_cat_info = {}
        # load user_cat_map
        self.user_hist_cat_dist, self.confounder_prior = self.load_user_cat_num(user_hist_cat_file, num_groups, model_name)
        
        # logged for negative sampling. 
        self.train_list = np.load(train_path, allow_pickle=True).tolist()
        self.train_dict = {}
        self.user_max_ID = max(user_feature.keys())
        self.item_max_ID = max(item_feature.keys())
        self.user_item_known_mat = sp.dok_matrix((self.user_max_ID+1, self.item_max_ID+1), dtype=np.float32)
        for pair in self.train_list:
            if len(pair) >3:
               userID, itemID, label, _ = pair
            else:
               userID, itemID, label = pair
            self.user_item_known_mat[userID, itemID] = 1
            if userID not in self.train_dict:
                self.train_dict[userID] = []
            self.train_dict[userID].append([itemID, label])
            
            # features used for model training
            self.known_user_features.append(np.array(self.user_feature[userID][0]))
            self.known_user_feature_values.append(np.array(self.user_feature[userID][1], dtype=np.float32))
            self.known_item_features.append(np.array(self.item_feature[itemID][0]))
            self.known_item_feature_values.append(np.array(self.item_feature[itemID][1], dtype=np.float32))
            
            self.known_user_hist_cat_dist.append(self.user_hist_cat_dist[userID])
            self.known_labels.append(np.float32(label))

        self.label = self.known_labels
        self.user_features = self.known_user_features
        self.user_feature_values = self.known_user_feature_values
        self.item_features = self.known_item_features
        self.item_feature_values = self.known_item_feature_values
        self.user_hist_cat_dist_fs = self.known_user_hist_cat_dist
                
        print('[FM_DATA]train_data: %d, pos_data:%d'%(len(self.label), np.sum(np.array(self.known_labels, dtype=np.float))))
        assert len(self.user_features) == len(self.user_feature_values) == len(self.label)
        assert len(self.item_features) == len(self.item_feature_values) == len(self.label)
        assert len(self.user_hist_cat_dist_fs) == len(self.label)
        assert all(len(item) == len(self.user_features[0]
             ) for item in self.user_features), 'features are of different length'
        assert all(len(item) == len(self.item_features[0]
             ) for item in self.item_features), 'features are of different length'

    def load_user_cat_num(self, data_path, num_groups, model_name):

        user_hist_cat_dict = np.load(data_path, allow_pickle=True).item()
        user_hist_cat_dist = {}

        confounder_prior = np.array([0]*num_groups, dtype=np.float32) 

        for uID, cat_hist_l in user_hist_cat_dict.items():
            cur_user_all = np.array(cat_hist_l[0], dtype=np.float32)
            cur_user_pos = np.array(cat_hist_l[1], dtype=np.float32)

            cur_user_all_dist = cur_user_all/(np.sum(cur_user_all) +0.0001)
            cur_user_pos_dist = cur_user_pos/(np.sum(cur_user_pos) +0.00001)
            self.user_hist_cat_info[uID] = [cur_user_all, cur_user_pos]
 
            if model_name in ['IPS', 'NFM']:
              user_hist_cat_dist[uID] = cur_user_all_dist
            elif model_name in ['DGCN']:
              user_hist_cat_dist[uID] = cur_user_pos_dist
            else:
              cur_user_hist_cat_dist = cur_user_pos_dist / (cur_user_all_dist +0.0001)
              user_hist_cat_dist[uID] = cur_user_hist_cat_dist

            # calculate confounder_prior
            confounder_prior += cur_user_pos_dist 
        
        confounder_prior = confounder_prior/len(user_hist_cat_dist)
        #print('confounder_prior: ', confounder_prior, 'sum: ', np.sum(confounder_prior))

        return user_hist_cat_dist, confounder_prior

 

        
    def ng_sample(self):
        
        # negative sampling
        neg_label = []
        neg_user_features = []
        neg_user_feature_values = []
        neg_item_features = []
        neg_item_feature_values = []
        neg_user_hist_cat_dist = []

        item_list = list(self.item_feature.keys())
        for pair in self.train_list:
            if len(pair) >3:
               userID, itemID, label, _ = pair
            else:
               userID, itemID, label = pair
            if label<0:
                continue
            for i in range(self.negative_num):
                j = item_list[np.random.randint(len(item_list))]
                while (userID, j) in self.user_item_known_mat:
                    j = item_list[np.random.randint(len(item_list))]
 
                neg_user_features.append(np.array(self.user_feature[userID][0]))
                neg_user_feature_values.append(np.array(self.user_feature[userID][1], dtype=np.float32))
                neg_item_features.append(np.array(self.item_feature[j][0]))
                neg_item_feature_values.append(np.array(self.item_feature[j][1], dtype=np.float32))
                neg_user_hist_cat_dist.append(self.user_hist_cat_dist[userID])
                neg_label.append(np.float32(0))
        
        
        self.user_features = self.known_user_features + neg_user_features
        self.user_feature_values = self.known_user_feature_values + neg_user_feature_values
        self.item_features = self.known_item_features + neg_item_features
        self.item_feature_values = self.known_item_feature_values + neg_item_feature_values
        self.user_hist_cat_dist_fs = self.known_user_hist_cat_dist + neg_user_hist_cat_dist
        self.label = self.known_labels + neg_label

        assert len(self.user_features) == len(self.user_feature_values) == len(self.label)
        assert len(self.item_features) == len(self.item_feature_values) == len(self.label)
        assert len(self.user_hist_cat_dist_fs) == len(self.label)
        assert all(len(item) == len(self.user_features[0]
             ) for item in self.user_features), 'features are of different length'
        assert all(len(item) == len(self.item_features[0]
             ) for item in self.item_features), 'features are of different length'
        

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]
        user_features = self.user_features[idx]
        user_feature_values = self.user_feature_values[idx]
        item_features = self.item_features[idx]
        item_feature_values = self.item_feature_values[idx]
        cur_user_hist_cat_dist = self.user_hist_cat_dist_fs[idx]

        
        return user_features, user_feature_values, item_features, item_feature_values,cur_user_hist_cat_dist, label
    
    
    