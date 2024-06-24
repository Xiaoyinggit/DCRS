from ast import Assert
import sys
import random
import numpy as np
np.set_printoptions(suppress=True)

max_seq_len = 50
class Indexer:

    def __init__(self):
        self.index_map = {}
        self.index = 0

    def get_index(self, key, check_add=False):
        try:
            tmp = self.index_map[key]
        except KeyError:
            if check_add == True:
                raise AssertionError
            self.index_map[key]=self.index
            self.index +=1
        return self.index_map[key]

    def size(self):
        return len(self.index_map)

    def save(self, file_name):
        np.save(file_name, self.index_map)
item_indexer = Indexer()

def read_ratings(rating_file):
    user_rating, mid_set = {}, set()
    with open (rating_file, 'r') as fr:
        for line in fr:
            ele = line.split('::')
            if len(ele)!=4:
                raise AssertionError
            uid = int(ele[0])
            try:
               movie_id = item_indexer.get_index(int(ele[1]), check_add=True)
            except:
               print('[read_ratings] cannot find movie_id :%d'%(int(ele[1])))
               continue
            rating = float(ele[2])
            label = 1 if rating>=4 else 0
            time = int(ele[3])

            user_rating[uid] = user_rating.get(uid, [])
            user_rating[uid].append([movie_id, label, time])
            mid_set.add(movie_id)
    
    # dataset_split
    valid_Un =0
    mid_set = list(mid_set)
    user_rating_train, user_rating_valid, user_rating_test = {}, {}, {}

    for uID, u_rl_raw in user_rating.items():
        if len(u_rl_raw) <20:
            continue
        valid_Un +=1
        u_rl = sorted(u_rl_raw, key=lambda v: v[2])
        len_rl = len(u_rl)
        test_num = int(round(len_rl*0.1))
        train_num = len_rl - 2*test_num
        if train_num < 0:
            raise AssertionError
        hist_item = set([pair[0] for pair in u_rl])

        # get user_rating_train data
        user_rating_train[uID] = []
        hist_pos_id = []
        for tmpi in range(train_num):
            tmpitemID, tmplabel, tmp_time = u_rl[tmpi]
            if tmplabel > 0:
                # do neg_sample
                j = mid_set[np.random.randint(len(mid_set))]
                while j in hist_item:
                    j = mid_set[np.random.randint(len(mid_set))]
                hist_item.add(j)
                user_rating_train[uID].append([j, 0, tmp_time,  hist_pos_id[-max_seq_len:]])

            user_rating_train[uID].append([tmpitemID, tmplabel, tmp_time, hist_pos_id[-max_seq_len:]])
            if tmplabel >0:
                hist_pos_id.append(tmpitemID)

        # get user_rating_valid data
        user_rating_valid[uID] = []
        for tmpi in range(train_num, train_num+test_num):
            tmpitemID, tmplabel, tmp_time = u_rl[tmpi]
            if tmplabel > 0:
                # do neg_sample
                j = mid_set[np.random.randint(len(mid_set))]
                while j in hist_item:
                    j = mid_set[np.random.randint(len(mid_set))]
                hist_item.add(j)
                user_rating_valid[uID].append([j, 0, tmp_time,  hist_pos_id[-max_seq_len:]])

            user_rating_valid[uID].append([tmpitemID, tmplabel, tmp_time, hist_pos_id[-max_seq_len:]])
            if tmplabel >0:
                hist_pos_id.append(tmpitemID)

        # get user_rating_test data
        user_rating_test[uID] = []
        for tmpi in range(train_num+test_num, len(u_rl_raw)):
            tmpitemID, tmplabel, tmp_time = u_rl[tmpi]
            if tmplabel > 0:
                # do neg_sample
                j = mid_set[np.random.randint(len(mid_set))]
                while j in hist_item:
                    j = mid_set[np.random.randint(len(mid_set))]
                hist_item.add(j)
                user_rating_test[uID].append([j, 0, tmp_time, hist_pos_id[-max_seq_len:]])

            user_rating_test[uID].append([tmpitemID, tmplabel, tmp_time, hist_pos_id[-max_seq_len:]])
            if tmplabel >0:
                hist_pos_id.append(tmpitemID)
    
    print('valid_Un: %d, all:%d'%(valid_Un, len(user_rating)))
    return user_rating_train, user_rating_valid, user_rating_test

def load_group_info(group_file):
    if item_indexer.size() >0:
        raise AssertionError
    cat_set = set()
    movie_cat_raw = {}
    with open(group_file, 'r', encoding='utf-8') as fr:
        for line in fr:
             ele = line.strip('\n ').split('::')
             if len(ele)!=3:
                 raise AssertionError
             movieID = item_indexer.get_index(int(ele[0]))

             generes = ele[2].split('|')
             for g in generes:
                 cat_set.add(g)

             movie_cat_raw[movieID] = movie_cat_raw.get(movieID, [])
             movie_cat_raw[movieID].extend(generes)

    movie_cat = {}
    cat_l = sorted(list(cat_set))
    cat_ind = {cat_l[i]: i for i in range(len(cat_l))}
    global_m_cat_dist = np.array([0.0 for _ in range(len(cat_l))])
    movie_feat_dict = {}

    for movieID, m_cat in movie_cat_raw.items():
        item_cat = cat_l
        item_cat_val = [0 for _ in range(len(cat_l))]
        for c in m_cat:
            ind = cat_ind[c]
            item_cat_val[ind] = 1.0/len(m_cat)
        movie_cat[movieID] = [item_cat, item_cat_val]
        global_m_cat_dist += np.array(item_cat_val, dtype=np.float32)

        #set movie feature 
        move_feat = [movieID]
        move_feat.extend(['IC%d'%tmp_i for tmp_i in range(len(cat_l))])
        movie_feat_val = [1]
        movie_feat_val.extend(item_cat_val)
        movie_feat_dict[movieID] =[move_feat, movie_feat_val]

    print('movie_num: ', len(movie_cat))
    print(cat_set, 'num: ',len(cat_set))
    print('[load_movie] global_m_cat_dist: ', global_m_cat_dist/len(movie_cat_raw))
    return movie_cat, len(cat_l), movie_feat_dict
            
def calculate_confounded_prior(out_folder, user_rating_in, movie_cat, num_cat):
    confounded_prior = np.array([0 for _ in range(num_cat)], dtype=np.float32)
    
    for uID, rl in user_rating_in.items():
        cur_user_hist_pos_cat_dist = np.array([0 for _ in range(num_cat)], dtype=np.float32)
        cur_pos_num = 0

        for movieID, label, _, _ in rl:
            try:
               item_cat_val = movie_cat[movieID][1]
            except KeyError:
               print('[calculate_confouned_prior] cannot find movieID:%d'%movieID)
               continue 
            if label>0:
                cur_user_hist_pos_cat_dist += np.array(item_cat_val, dtype=np.float32)
                cur_pos_num +=1
        if cur_pos_num >0:
            if abs(np.sum(cur_user_hist_pos_cat_dist) - cur_pos_num) >0.1:
               raise AssertionError
            cur_user_hist_pos_cat_dist = cur_user_hist_pos_cat_dist/np.sum(cur_user_hist_pos_cat_dist)
        
        confounded_prior += cur_user_hist_pos_cat_dist

    confounded_prior = confounded_prior/len(user_rating_in)

    if abs(np.sum(confounded_prior) -1.0) >0.1:
        raise AssertionError
    np.save( out_folder+'/confounder_prior.npy', confounded_prior)

 


def inspect_rating_stat(user_rating_in, movie_cat, num_cat):
    valid_Un = 0 # number of user that has at least 20 ratings
    pos_num, neg_num = 0,0 # pos_ratio
    avg_user_pos_r = 0
    # use_hist_dist
    user_hist_cat_dist = np.array([0 for _ in range(num_cat)], dtype=np.float32)
    user_hist_pos_cat_dist = np.array([0 for _ in range(num_cat)], dtype=np.float32)
    user_hist_neg_cat_dist = np.array([0 for _ in range(num_cat)], dtype=np.float32)
    user_hist_neg_cat_dist = np.array([0 for _ in range(num_cat)], dtype=np.float32)
    pos_neg_div = np.array([0 for _ in range(num_cat)], dtype=np.float32) 
    pos_neg_valid_num = np.array([0 for _ in range(num_cat)], dtype=np.float32) 

    user_dist_dict = {}
    for uID, rl in user_rating_in.items():
        cur_user_hist_cat_dist = np.array([0 for _ in range(num_cat)], dtype=np.float32)
        cur_user_hist_pos_cat_dist = np.array([0 for _ in range(num_cat)], dtype=np.float32)
        cur_user_hist_neg_cat_dist = np.array([0 for _ in range(num_cat)], dtype=np.float32)
        cur_pos_num = 0

        for movieID, label, _, _ in rl:
            try:
                item_cat_val = movie_cat[movieID][1]
            except KeyError:
                print('[inspect_rating_stat] cannot find movie:%d'%movieID)
            cur_user_hist_cat_dist += np.array(item_cat_val, dtype=np.float32)
            if label>0:
                cur_user_hist_pos_cat_dist += np.array(item_cat_val, dtype=np.float32)
                pos_num+=1
                cur_pos_num +=1
            else:
                cur_user_hist_neg_cat_dist += np.array(item_cat_val, dtype=np.float32)
                neg_num +=1

        cur_user_hist_cat_dist = cur_user_hist_cat_dist / len(rl)
        #if cur_pos_num>0:
        #    cur_user_hist_pos_cat_dist = cur_user_hist_pos_cat_dist / cur_pos_num
        #if len(rl)-cur_pos_num >0:
        #    cur_user_hist_neg_cat_dist = cur_user_hist_neg_cat_dist / (len(rl)-cur_pos_num) 
        user_dist_dict[uID] = [ cur_user_hist_cat_dist, cur_user_hist_pos_cat_dist, cur_user_hist_neg_cat_dist]

        for tmp_i in range(num_groups):
            if cur_user_hist_neg_cat_dist[tmp_i] >0:
                pos_neg_div[tmp_i] += np.divide(cur_user_hist_pos_cat_dist[tmp_i], (cur_user_hist_neg_cat_dist[tmp_i]+0.000001))
                pos_neg_valid_num[tmp_i] +=1

        user_hist_cat_dist += cur_user_hist_cat_dist
        user_hist_pos_cat_dist += cur_user_hist_pos_cat_dist
        user_hist_neg_cat_dist += cur_user_hist_neg_cat_dist

        cur_pos_ratio = cur_pos_num/float(len(rl))
        avg_user_pos_r += cur_pos_ratio
    print('pos_num: %d, neg_num:%d, pos_ratio:%f, avg_user_pos_r:%f'%(pos_num, neg_num, pos_num/float(neg_num +pos_num), avg_user_pos_r/len(user_rating_in)))
    print('user_hist_cat_dist:', user_hist_cat_dist/len(user_rating_in))
    print('user_hist_pos_cat_dist:', user_hist_pos_cat_dist/(pos_num))
    print('user_hist_neg_cat_dist:', user_hist_neg_cat_dist/neg_num)
    print('pos_neg_div: ', pos_neg_div/pos_neg_valid_num)
    print('global_pos_neg_div ', (user_hist_pos_cat_dist) /(user_hist_neg_cat_dist))
    return user_dist_dict

def load_user_feature(user_file, user_rating_train):
    user_feat_map = {}

    UF_prefix = [None, 'UG', 'UA', 'UO', 'UZC']
    with open(user_file, 'r') as fr:
        for line in fr:
            ele = line.strip().split('::')
            if len(ele) !=5:
                raise AssertionError
            uID = int(ele[0])
            uFeat_l = [uID]
            uFeat_l.extend([UF_prefix[i]+ele[i] for i in range(1, len(UF_prefix))])
            uFeat_v = [1 for _ in range(len(UF_prefix))]

            # generate user hist seq of last training point
            train_list = user_rating_train[uID]
            # TODO : check the rating is sorted by time

            pos_seq = [i_id for i_id, label,_, _ in train_list if label >0]
            
            user_feat_map[uID] = [uFeat_l, uFeat_v, pos_seq[-max_seq_len:]]


    print('[load_user_feature] user_num:%d'%len(user_feat_map))
    return user_feat_map

def get_fake_user_feat(user_train_rating):
    user_feat_map = {}
    for uID, rl in user_train_rating.items():

        pos_seq = [i_id for i_id, label,_, _ in rl if label >0]
        user_feat_map[uID] = [[uID], [1], pos_seq[-max_seq_len:]]
    print('[get_fake_user_feat] user_num:%d'%len(user_feat_map))
    return user_feat_map

def save_rating_data(user_rating_in, out_file, user_feat_map, movie_feat_map):
    user_rating_l = []
    num , pos = 0, 0
    for uID, u_rl in user_rating_in.items():
        if uID not in user_feat_map.keys():
            raise AssertionError

        for movieID, label, time, hist_seq in u_rl:
            if movieID not in movie_feat_map.keys():
                print('[Warn] movieID %d not in movie_feat_map'%movieID)
                continue
            user_rating_l.append([uID, movieID, label, hist_seq])
            num+=1 
            pos+=label

    np.save(out_file, user_rating_l, allow_pickle=True)
    print(out_file, ', num:%d, pos:%d'%(num, pos))


    



def save_data(out_folder, user_feat_map, movie_feat_dict, user_rating_train, user_rating_valid, user_rating_test, user_hist_cat_num):
    # save user_feat_map
    np.save(out_folder+'/user_feature_file.npy', user_feat_map, allow_pickle=True)

    # save movie_feat_dict 
    np.save(out_folder+'/item_feature_file.npy', movie_feat_dict, allow_pickle=True)

    # save user_hist_cat num
    np.save(out_folder+'/user_hist_cat.npy', user_hist_cat_num, allow_pickle=True)

    # save user-train data
    save_rating_data(user_rating_train, out_folder+'/train_list.npy', user_feat_map, movie_feat_dict)
    save_rating_data(user_rating_valid, out_folder+'/valid_list.npy', user_feat_map, movie_feat_dict)
    save_rating_data(user_rating_test, out_folder+'/test_list.npy', user_feat_map, movie_feat_dict)


def cal_user_hist_cat(user_rating_dict, movie_cat, num_cat):
    user_hist_cat_dict = {}

    for uID, rl in user_rating_dict.items():
        cur_user_hist_cat_all = np.array([0 for _ in range(num_cat)], dtype=np.float32)
        cur_user_hist_cat_pos = np.array([0 for _ in range(num_cat)], dtype=np.float32)
        cur_pos_num = 0

        for movieID, label, _, _ in rl:
            try:
               item_cat_val = movie_cat[movieID][1]
            except KeyError:
                print('[cal_usr_hist_cat] Cannot find movieID:%d'%movieID)
                continue
            cur_user_hist_cat_all += np.array(item_cat_val, dtype=np.float32)
            if label>0:
                cur_user_hist_cat_pos += np.array(item_cat_val, dtype=np.float32)
        user_hist_cat_dict[uID] = [cur_user_hist_cat_all, cur_user_hist_cat_pos]

    return user_hist_cat_dict



            
if __name__ == '__main__':
    in_folder, is_user_info, out_folder = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    
    movie_cat, num_groups, movie_feat_dict = load_group_info(in_folder+'/movies.dat')
    user_rating_train, user_rating_valid, user_rating_test = read_ratings(in_folder+'/ratings.dat')

    user_hist_cat_num = cal_user_hist_cat(user_rating_train, movie_cat, num_groups)
    
    if is_user_info >0:
        user_feat_map = load_user_feature(in_folder+'/users.dat', user_rating_train)
    else:
        user_feat_map = get_fake_user_feat(user_rating_train)
        

    save_data(out_folder, user_feat_map, movie_feat_dict, user_rating_train, user_rating_valid, user_rating_test, user_hist_cat_num)
    item_indexer.save(out_folder+'/item_index.npy')
    # confounding prior for DecNFM
    calculate_confounded_prior(out_folder,user_rating_train, movie_cat, num_groups)

    
    
    print('~~~~~~~~train_stat~~~~~~~~~~')
    user_dist_train_dict = inspect_rating_stat(user_rating_train, movie_cat, num_groups)
    #print('~~~~~~~~valid_stat~~~~~~~~~~~')
    #user_dist_valid_dict = inspect_rating_stat(user_rating_valid, movie_cat, num_groups)
    #print('~~~~~~~~test_stat~~~~~~~~~~~~~~~')
    #user_dist_test_dict = inspect_rating_stat(user_rating_test, movie_cat, num_groups)


    # train_valid_div = np.array([0 for _ in range(num_groups)], dtype=np.float32)
    # train_valid_num = np.array([0 for _ in range(num_groups)], dtype=np.float32)
    # train_test_div = np.array([0 for _ in range(num_groups)], dtype=np.float32)
    # train_test_num = np.array([0 for _ in range(num_groups)], dtype=np.float32)

    # for uID, dist_l in user_dist_train_dict.items():
    #     if np.all(user_dist_valid_dict[uID][0] >0):
    #         train_valid_div += dist_l[0]/(user_dist_valid_dict[uID][0])
    #         train_valid_num += 1
    #     if np.all(user_dist_test_dict[uID][0] >0):
    #         train_test_div += dist_l[0]/(user_dist_test_dict[uID][0])
    #         train_test_num +=1

    # print('train-valid_div: ', train_valid_div/train_valid_num)
    # print('train_test_div: ', train_test_div/train_test_num)


