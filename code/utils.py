

def reverse_user(user_features_t, feature_reverse):
    user_features_new = []
    user_features = user_features_t.cpu().numpy().tolist()
    for uf in user_features:
        uf_r = []
        for f in uf:
            uf_r.append(feature_reverse[f])
        user_features_new.append(uf_r)
    return user_features_new

def reverse_item(item_features_t, item_feature_values_t, feature_reverse, item_index_map=None):
    item_out = []
    item_features = item_features_t.cpu().numpy().tolist()
    item_feature_values = item_feature_values_t.cpu().numpy().tolist()
    for i in range(len(item_features)):
        item_f = item_features[i]
        if_val = item_feature_values[i]
        n_if = []
        itemID = feature_reverse[item_f[0]]
        if item_index_map is not None:
            itemID = item_index_map[itemID]
        n_if.append(itemID)
        for j in range(1, len(item_f)):
            if if_val[j]>0:
                n_if.append('%s\t%f'%(feature_reverse[item_f[j]], if_val[j]))
        item_out.append(n_if)
    return item_out

def split_book_by_avg_ratings(user_rating_dict):
    movie_avg_rating = {}
    for uID, rl in user_rating_dict.items():
        for movieID, rating in rl:
            movie_avg_rating[movieID] = movie_avg_rating.get(movieID, [0, 0])
            movie_avg_rating[movieID][0] += rating
            movie_avg_rating[movieID][1] +=1

    # only insepect movies with >100ratings
    avg_bucket_bookL = {}
    valid_book =0
    for movieID, m_stat in movie_avg_rating.items():
        if m_stat[1] <50:
            continue
        valid_book +=1
        avg_r = round(m_stat[0]/m_stat[1], 1)
        avg_bucket_bookL[avg_r] = avg_bucket_bookL.get(avg_r, [])
        avg_bucket_bookL[avg_r].append(movieID)
    print('After filter len:100', valid_book)

    print('[avg_bucket_stat]: ', sorted([(ar, len(bl)) for ar, bl in avg_bucket_bookL.items()], key=lambda v: v[0]))

    return avg_bucket_bookL

def get_user_candidate(train_dict, target_bookL):

    user_candi = {}
    for userID, rl in train_dict.items():
        t_rl = []
        for itemID, label in rl:
            if itemID in target_bookL:
                t_rl.append([itemID, label])
        if len(t_rl)>1:
            user_candi[userID] = t_rl
    print('[util.get_user_candidate] in_user:%d, out_user:%d'%(len(train_dict), len(target_bookL))) 
    return user_candi 