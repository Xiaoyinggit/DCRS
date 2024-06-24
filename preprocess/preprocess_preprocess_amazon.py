import sys
import random
import numpy as np
np.set_printoptions(suppress=True)
import gzip, json
max_seq_len = 50

class Indexer:

    def __init__(self):
        self.index_map = {}
        self.index = 0

    def get_index(self, key, check_add=False):
        old_index = self.index
        try:
            tmp = self.index_map[key]
        except KeyError:
            self.index_map[key]=self.index
            self.index +=1
        if (check_add==True) and(self.index!=old_index):
            raise AssertionError
        return self.index_map[key]

    def size(self):
        return len(self.index_map)

    def save(self, file_name):
        np.save(file_name, self.index_map)

user_indexer = Indexer()
item_indexer = Indexer()


def read_raw_ratings(file):
    # read raw ratings, and filter user with less 20 ratings.
    user_rating = {}
    with gzip.open(file, 'r') as fr:
        for line in fr:
            js = json.loads(line)
            reviewerID = js['reviewerID']
            bookID = js['asin']
            rating = js['overall']
            label = 1 if rating>4 else 0
            time = js['unixReviewTime']
            user_rating[reviewerID] = user_rating.get(reviewerID, [])
            user_rating[reviewerID].append([bookID, label, time])
    
    # filter books that has less 20 inteaction 
    valid_Un, valid_Bn = 0, 0
    user_rating_cut, valid_book = {}, {}
    for uID, rl in user_rating.items():
        if len(rl) <20:
            continue
        valid_Un +=1
        user_rating_cut[uID] = rl
        for bookID, _, _ in rl:
            valid_book[bookID]=valid_book.get(bookID, 0)
            valid_book[bookID] +=1

    print('[load_raw_ratings]  all_user:%d, valid_UN:%d-%d, valid_book:%d'%( len(user_rating), valid_Un, len(user_rating_cut), len(valid_book)))
    return user_rating_cut, valid_book
            

def load_group_info(group_file, valid_book):
    cat_set = set()
    movie_cat_raw = {}
    no_cat_books=0
    line_c = 0
    cat_book = {}
    with gzip.open(group_file, 'r') as fr:
        for line in fr:
             line_c +=1
             if line_c %100000 ==0:
                 print('[load_group_info] line_c:%d'%line_c)
             j_s = eval(line)
             bookID = j_s['asin']
             try:
                 tmp = valid_book[bookID]
             except KeyError:
                 continue
             # filter books with less 20 ratings
             if valid_book[bookID]<20:
                 continue

             generes = j_s['category']
              
             if len(generes)<1:
                 no_cat_books +=1
                 continue
             # only contains book in valid_book & book_with_category.
            
             if  isinstance(generes[0],list):
                 print(generes)
                 raise AssertionError
             
             g_set = set()
             for g in generes:
                 if 'Books' in g:
                     continue
                 g_set.add(g.replace('amp;', ''))
             if len(g_set) <1:
                 continue

             cat_set = cat_set | set(g_set)

             for g in g_set:
                cat_book[g] = cat_book.get(g, {})
                cat_book[g][bookID] =1

             try:
                 tmp = movie_cat_raw[bookID]
             except KeyError:
                 movie_cat_raw[bookID] = []
             movie_cat_raw[bookID].extend(list(g_set))

    print('[load group info] loading books:%d, no_cat_books:%d'%(len(movie_cat_raw), no_cat_books))
     
    cat_sat = {c: len(bn) for c, bn in cat_book.items()}
    print('cat_num: ', sorted(cat_sat.items(), key=lambda v: v[1]))
    
    
    # keep only categories that link to at least 50 books
    valid_cat_set = {}
    for cat, g in  cat_book.items():
        if len(g)<20:
            print('[load group info] remove cat %s due to low_freq:%f'%(cat, len(g)))
            continue
        elif len(g) > len(movie_cat_raw)*0.7:
            print('[load group info] remove cat %s due to high freq:%f'%(cat ,len(g)))
        else:
            valid_cat_set[cat] =1




    print('[load group info] cat_set: %d, valid_cat_num:%d'%(len(cat_set), len(valid_cat_set)))
    cat_list = list(valid_cat_set.keys())
    cat_ind = {cat_list[i]: i for i in range(len(cat_list))}

    movie_cat = {}
    for movieID, m_cat in movie_cat_raw.items():
        valid_g_cat = []
        for c in m_cat:
            try:
                tmp = valid_cat_set[c]
            except KeyError:
                continue
            valid_g_cat.append(c)
        if len(valid_g_cat) <1:
            continue
        movie_cat[movieID] =  valid_g_cat
        

    print('[load_group_info] movie_num: ', len(movie_cat))
    return movie_cat, cat_ind
            
def get_fake_user_feat(user_train_rating):
    if user_indexer.size()>0:
        raise AssertionError

    user_feat_map = {}

    for uID_raw, rl in user_train_rating.items():
        uID = user_indexer.get_index(uID_raw)
        pos_seq = [iid for iid, label, _, _ in rl if label>0]
        user_feat_map[uID] = [[uID], [1], pos_seq[-max_seq_len:]]
    print('[get_fake_user_feat] user_num:%d'%len(user_feat_map))
    return user_feat_map

def cal_user_hist_cat(user_rating_dict, movie_cat, cat_ind):
    user_hist_cat_dict = {}
    num_cat = len(cat_ind)
    
    for uID_raw, rl in user_rating_dict.items():
        uID = user_indexer.get_index(uID_raw, check_add=True)
        cur_user_hist_cat_all = np.array([0 for _ in range(num_cat)], dtype=np.float32)
        cur_user_hist_cat_pos = np.array([0 for _ in range(num_cat)], dtype=np.float32)
        
        for movieID, label, _, _ in rl:
            g_cat = movie_cat[movieID]
            item_cat_val = [0 for _ in range(num_cat)]
            for c in g_cat:
               ind = cat_ind[c]
               item_cat_val[ind] = 1.0/len(g_cat)
            cur_user_hist_cat_all += np.array(item_cat_val, dtype=np.float32)
            if label>0:
                cur_user_hist_cat_pos += np.array(item_cat_val, dtype=np.float32)
        user_hist_cat_dict[uID] = [cur_user_hist_cat_all, cur_user_hist_cat_pos]

    return user_hist_cat_dict
def print_dist(prefix, in_dist, cat_ind):
    in_dist_sum = np.sum(in_dist)
    dist_d = {cat:in_dist[ind] for cat,ind in cat_ind.items()}
    dist_d_s = sorted(dist_d.items(), key=lambda v: v[1])
    print(prefix, 'sum: ', in_dist_sum, 'detail: ', dist_d_s)

def inspect_rating_stat(user_rating_in, movie_cat, cat_ind):
    num_cat = len(cat_ind)
    valid_Un = 0 # number of user that has at least 20 ratings
    pos_num, neg_num = 0,0 # pos_ratio
    avg_user_pos_r = 0
    # use_hist_dist
    user_hist_cat_dist = np.array([0 for _ in range(num_cat)], dtype=np.float32)
    user_hist_pos_cat_dist = np.array([0 for _ in range(num_cat)], dtype=np.float32)
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
            g_cat = movie_cat[movieID]
            item_cat_val = [0 for _ in range(num_cat)]
            for c in g_cat:
               ind = cat_ind[c]
               item_cat_val[ind] = 1.0/len(g_cat)
 
            cur_user_hist_cat_dist += np.array(item_cat_val, dtype=np.float32)
            if label>0:
                cur_user_hist_pos_cat_dist += np.array(item_cat_val, dtype=np.float32)
                pos_num+=1
                cur_pos_num +=1
            else:
                cur_user_hist_neg_cat_dist += np.array(item_cat_val, dtype=np.float32)
                neg_num +=1

        cur_user_hist_cat_dist_normed = cur_user_hist_cat_dist / len(rl)
        #print('~~~~~~~uid:%s, len_rl:%d~~~~~~'%(str(uID), len(rl)))
        #
        #print('cur_user_hist_cat_dist: ', {i:cur_user_hist_cat_dist[i] for i in range(len(cur_user_hist_cat_dist)) if cur_user_hist_cat_dist[i]>0})
        #print('cur_user_hist_pos_cat_dist: ', {i:cur_user_hist_pos_cat_dist[i] for i in range(len(cur_user_hist_pos_cat_dist)) if cur_user_hist_pos_cat_dist[i]>0})
        #print('cur_user_hist_neg_cat_dist: ', {i:cur_user_hist_neg_cat_dist[i] for i in range(len(cur_user_hist_neg_cat_dist)) if cur_user_hist_neg_cat_dist[i]>0})

        #if cur_pos_num>0:
        #    cur_user_hist_pos_cat_dist_normed = cur_user_hist_pos_cat_dist / cur_pos_num
        #if len(rl)-cur_pos_num >0:
        #    cur_user_hist_neg_cat_dist_normed = cur_user_hist_neg_cat_dist / (len(rl)-cur_pos_num) 
        #user_dist_dict[uID] = [ cur_user_hist_cat_dist, cur_user_hist_pos_cat_dist, cur_user_hist_neg_cat_dist]

        for tmp_i in range(num_cat):
            if cur_user_hist_neg_cat_dist[tmp_i] >0:
                pos_neg_div[tmp_i] += np.divide(cur_user_hist_pos_cat_dist[tmp_i], (cur_user_hist_neg_cat_dist[tmp_i]+0.000001))
                pos_neg_valid_num[tmp_i] +=1

        user_hist_cat_dist += cur_user_hist_cat_dist_normed
        user_hist_pos_cat_dist += cur_user_hist_pos_cat_dist
        user_hist_neg_cat_dist += cur_user_hist_neg_cat_dist

        cur_pos_ratio = cur_pos_num/float(len(rl))
        avg_user_pos_r += cur_pos_ratio
    print('pos_num: %d, neg_num:%d, pos_ratio:%f, avg_user_pos_r:%f'%(pos_num, neg_num, pos_num/float(neg_num +pos_num), avg_user_pos_r/len(user_rating_in)))
    print_dist('user_hist_cat_dist:', user_hist_cat_dist/len(user_rating_in), cat_ind)
    print_dist('user_hist_pos_cat_dist:', user_hist_pos_cat_dist/(pos_num), cat_ind)
    print_dist('user_hist_neg_cat_dist:', user_hist_neg_cat_dist/neg_num, cat_ind)
    print_dist('pos_neg_div: ', pos_neg_div/(pos_neg_valid_num+0.001), cat_ind)
    print_dist('global_pos_neg_div ', (user_hist_pos_cat_dist) /(user_hist_neg_cat_dist+0.0001), cat_ind)
    return user_dist_dict

def gene_user_recall_candidtes(user_rating_train, user_recall_candidates, valid_book_cat, candidates_num):
    # for testing recall, randomly sample candidate set of candidate_num. must contains the testing dataset.
    user_train_item_record = {}
    for uID, rl in user_rating_train.items():
        user_train_item_record[uID] = {}
        for itemID, label, _, _ in rl:
            user_train_item_record[uID][itemID] = 1
    print('[gene_user_recall_candidates] start !')
    user_candidates = {}
    # first insert test/train part
    for uID, rl in user_recall_candidates.items():
        user_candidates[uID]=user_candidates.get(uID, {})
        for itemID, _, _ in rl:
            user_candidates[uID][itemID]=1

    item_list = list(valid_book_cat.keys())
    uc_count = 0
    for uID in user_recall_candidates.keys():
        if uc_count %1000==0:
            print('[generate_recall_candi] uc_count:%d, all:%d'%(uc_count, len(user_recall_candidates)))
        uc_count +=1
        while len(user_candidates[uID]) < candidates_num:
            j = item_list[np.random.randint(len(item_list))]

            while ( j in user_train_item_record[uID]) or (j in user_candidates[uID]):
                 j = item_list[np.random.randint(len(item_list))]
               
            user_candidates[uID][j]=1
        
    return user_candidates




def join_and_split(user_ratings, movie_cat):
    print('[join and split] in_user:%d, in_book:%d'%(len(user_ratings), len(movie_cat)))
    user_rating_all = {}
    missing_book_set = set()
    valid_book_cat, valid_book_num = {}, {}
    for user, rl in user_ratings.items():
        user_rating_all[user] = []
        for bookID, label, time in rl:
            try: 
                tmp= movie_cat[bookID]
            except KeyError:
                missing_book_set.add(bookID)
                continue
            valid_book_cat[bookID] = movie_cat[bookID]
            valid_book_num[bookID] = valid_book_num.get(bookID, 0)
            valid_book_num[bookID] +=1


            user_rating_all[user].append([bookID, label, time])

    print('[join and split] Filter books not in movie_cat: %d, left_books:%d'%(len(missing_book_set), len(valid_book_cat)))
    
#    global_m_cat_dist = np.array([0.0 for _ in range(len(cat_ind))])
#    no_cat_books = 0
#    for bookID, g_cat in valid_book_cat.items():
#        if len(g_cat) <1:
#            raise AssertionError
#        item_cat_val = [0 for _ in range(len(cat_ind))]
#        if ('default_cat' in g_cat) and (len(g_cat)==1):
#            no_cat_books +=1
#        for c in g_cat:
#            ind = cat_ind[c]
#            item_cat_val[ind] = 1.0/len(g_cat)
#        global_m_cat_dist += np.array(item_cat_val, dtype=np.float32)
#    print('[join and split] no_cat_books:%d'%no_cat_books)
#    print_dist('[join and split] global_m_cat_dist: ', global_m_cat_dist/len(valid_book_cat), cat_ind)

    # filter user <2- & split data
    valid_Un =0
    user_rating_train, user_rating_valid, user_rating_test= {}, {}, {}

    valid_book_num, cat_num = {}, {}
    for uID, u_rl_raw in user_rating_all.items():
        if len(u_rl_raw) <20:
            continue
        for itemID, label, time in u_rl_raw:
            valid_book_num[itemID] = valid_book_num.get(itemID, 0)
            valid_book_num[itemID] +=1
    
    item_list = list(valid_book_num.keys())
    for uID, u_rl_raw in user_rating_all.items():
        if len(u_rl_raw)<20:
            continue
        
        u_rl = sorted(u_rl_raw, key=lambda v: v[2])
        len_rl = len(u_rl)
        test_num = int(round(len_rl*0.1))
        train_num = len_rl - 2*test_num
        if (train_num < 1) or (test_num<1):
            continue
        valid_Un +=1
        hist_item = set([pair[0] for pair in u_rl])


        # get user_rating_train data
        user_rating_train[uID] = []
        hist_pos_id = []
        for tmpi in range(train_num):
            tmpitemID, tmplabel, tmp_time = u_rl[tmpi]
            if tmplabel > 0:
                # do neg_sample
                j = item_list[np.random.randint(len(item_list))]
                while j in hist_item:
                    j = item_list[np.random.randint(len(item_list))]
                hist_item.add(j)
                user_rating_train[uID].append([j, 0, tmp_time, hist_pos_id[-max_seq_len:]])
 
            user_rating_train[uID].append([tmpitemID, tmplabel, tmp_time, hist_pos_id[-max_seq_len:]])

        
        # get user_rating_valid data
        user_rating_valid[uID] = []
        for tmpi in range(train_num, train_num+test_num):
            tmpitemID, tmplabel, tmp_time = u_rl[tmpi]
            if tmplabel > 0:
                # do neg_sample
                j = item_list[np.random.randint(len(item_list))]
                while j in hist_item:
                    j = item_list[np.random.randint(len(item_list))]
                hist_item.add(j)
                user_rating_valid[uID].append([j, 0, tmp_time, hist_pos_id[-max_seq_len:]])
             
            user_rating_valid[uID].append([tmpitemID, tmplabel, tmp_time, hist_pos_id[-max_seq_len:]])
 
        # get user_rating_test data
        user_rating_test[uID] = []
        for tmpi in range(train_num+test_num, len(u_rl)):
            tmpitemID, tmplabel, tmp_time = u_rl[tmpi]
            if tmplabel > 0:
                # do neg_sample
                j = item_list[np.random.randint(len(item_list))]
                while j in hist_item:
                    j = item_list[np.random.randint(len(item_list))]
                hist_item.add(j)
                user_rating_test[uID].append([j, 0, tmp_time, hist_pos_id[-max_seq_len:]])
 
            user_rating_test[uID].append([tmpitemID, tmplabel, tmp_time, hist_pos_id[-max_seq_len:]])
 
    print('valid_Un: %d, all:%d, left:%d'%(valid_Un, len(user_rating_all), len(user_rating_train)))

    book_nl =  [n for b,n in valid_book_num.items()] 
    print('valid_book_num_stat: ', min(book_nl), max(book_nl), np.mean(np.array(book_nl)))
    cat_num = {}
    for bookID, _ in valid_book_num.items():
        for g in movie_cat[bookID]:
            cat_num[g] = cat_num.get(g, 0)
            cat_num[g]+=1
    cat_nl = [n for g, n in cat_num.items()]
    print('cat_num_stat: ', min(cat_nl), max(cat_nl), np.mean(np.max(cat_nl)))
    return user_rating_train, user_rating_valid, user_rating_test
    


def save_rating_data(user_rating_in, out_file, user_feat_map, movie_feat_map):
    user_rating_l = []
    num , pos = 0, 0
    for uID_raw, u_rl in user_rating_in.items():
        uID = user_indexer.get_index(uID_raw, check_add=True)
        if uID not in user_feat_map.keys():
            raise AssertionError
        prev_time = u_rl[0][2]
        for movieID_raw, label, time, hs in u_rl:
            if time < prev_time:
                print('prev_time: ', prev_time)
                print(movieID_raw, label, time, hs)
                raise AssertionError
            movieID = item_indexer.get_index(movieID_raw, check_add=True)
            if movieID not in movie_feat_map.keys():
                print('[Warn] movieID %d not in movie_feat_map'%movieID)
                continue
            hi_new = []
            for h_i in hs:
                hi_new.append(item_indexer.get_index(h_i, check_add=True))
            user_rating_l.append([uID, movieID, label, hi_new])
            num+=1 
            pos+=label

    np.save(out_file, user_rating_l, allow_pickle=True)
    print(out_file, ', num:%d, pos:%d'%(num, pos))



def save_data(out_folder, user_feat_map, movie_cat, cat_ind,user_rating_train, user_rating_valid, user_rating_test, user_hist_cat_num):
    
    # save movie_feat_dict
    if item_indexer.size() >0:
        raise AssertionError
    movie_feat_dict = {}
    num_cat = len(cat_ind)
    cat_l = list(cat_ind.keys())
    for movieID_raw, g_cat in movie_cat.items():
        movieID = item_indexer.get_index(movieID_raw)
        cur_feat = [movieID]
        cur_feat.extend((['IC%d'%tmp_i for tmp_i in range(len(cat_l))]))
        item_cat_val = [0 for _ in range(num_cat)]
        for c in g_cat:
            ind = cat_ind[c]
            item_cat_val[ind] = 1.0/len(g_cat)


        cur_feat_val = [1]
        cur_feat_val.extend(item_cat_val)
        
        movie_feat_dict[movieID] =[cur_feat, cur_feat_val]

    np.save(out_folder+'/item_feature_file.npy', movie_feat_dict, allow_pickle=True)

    # save user_feat_map
    user_feat_map_new = {}
    for uID, f_l in user_feat_map.items():
        uFeat, uFeat_v, u_his_seq = f_l
        u_his_seq_new = []
        for hs in u_his_seq:
            u_his_seq_new.append(item_indexer.get_index(hs, check_add=True))
        user_feat_map_new[uID] =[uFeat, uFeat_v, u_his_seq_new]
    np.save(out_folder+'/user_feature_file.npy', user_feat_map_new, allow_pickle=True)


    # save user_hist_cat num
    np.save(out_folder+'/user_hist_cat.npy', user_hist_cat_num, allow_pickle=True)


    # save user-train data
    save_rating_data(user_rating_train, out_folder+'/train_list.npy', user_feat_map_new, movie_feat_dict)
    save_rating_data(user_rating_valid, out_folder+'/valid_list.npy', user_feat_map_new, movie_feat_dict)
    save_rating_data(user_rating_test, out_folder+'/test_list.npy', user_feat_map_new, movie_feat_dict)

    # save index 
    user_indexer.save(out_folder+'/user_index.npy')
    item_indexer.save(out_folder+'/item_index.npy')
    np.save(out_folder+'/cat_ind.npy', cat_ind)



if __name__ == '__main__':

    in_folder, out_folder = sys.argv[1], sys.argv[2]
    
    user_raw_ratings, valid_book= read_raw_ratings(in_folder+'/reviews_Books_5.json.gz')
    movie_cat, cat_ind = load_group_info(in_folder+'/meta_Books.json.gz', valid_book)
    user_rating_train, user_rating_valid, user_rating_test = join_and_split(user_raw_ratings, movie_cat)

    
    user_feat_map = get_fake_user_feat(user_rating_train)
    user_hist_cat_num = cal_user_hist_cat(user_rating_train, movie_cat, cat_ind)
    save_data(out_folder, user_feat_map, movie_cat, cat_ind,  user_rating_train, user_rating_valid, user_rating_test, user_hist_cat_num)
    print('~~~~~~~~train_stat~~~~~~~~~~')
    user_dist_train_dict = inspect_rating_stat(user_rating_train, movie_cat, cat_ind)
    