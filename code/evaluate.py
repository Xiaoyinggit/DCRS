from nis import cat
from pyexpat import model
from token import DEDENT
import numpy as np 
import torch
import math
import time

from torch._C import dtype
from torch.utils import data
import utils
from sklearn.metrics import roc_auc_score
np.set_printoptions(suppress=True)
torch.set_printoptions(precision=5)




def pre_candidate_ranking(item_feature):
    all_item_features = []
    all_item_feature_values = []
    
    item_list = list(item_feature.keys())
    for ind in range(len(item_feature)):
        itemID = item_list[ind]
        all_item_features.append(np.array(item_feature[itemID][0]))
        all_item_feature_values.append(np.array(item_feature[itemID][1], dtype=np.float32))
    
    all_item_features = torch.tensor(all_item_features).cuda()
    all_item_feature_values = torch.tensor(all_item_feature_values).cuda()
    
    return  all_item_features, all_item_feature_values

def selected_concat(user_feat, user_feat_values, user_cat_nums, all_item_features, all_item_feature_values, candidates,  y_true=None):
    candidate_num = len(candidates)
    user_feat = user_feat.expand(candidate_num, -1)
    user_feat_values = user_feat_values.expand(candidate_num, -1)
    user_cat_nums = user_cat_nums.expand(candidate_num, -1)

    candidate_indexes = torch.tensor(candidates).cuda()
    candidate_feature = all_item_features[candidate_indexes]
    candidate_feature_values = all_item_feature_values[candidate_indexes]
    

    if y_true is None: 
       fake_label = torch.tensor([0.0 for _ in range(candidate_num)]).cuda()
    else:
       fake_label = torch.tensor(y_true, dtype=torch.float64).cuda()

    
    return user_feat, user_feat_values, candidate_feature, candidate_feature_values, user_cat_nums, fake_label



def calculate_auc(dataList,model, model_name, user_feature, user_hist_cat_dist, item_idx, all_item_features, all_item_feature_values):
    item_idx_map = {item_idx[i]: i for i in range(len(item_idx))}
    
    all_user_prediction, all_user_label = None, None
    uauc_l =[] 
    for userID, rl in dataList.items():
        user_feat = torch.tensor(np.array(user_feature[userID][0])).cuda()
        user_feat_values = torch.tensor(np.array(user_feature[userID][1], dtype=np.float32)).cuda()
        user_cat_nums = torch.tensor(user_hist_cat_dist[userID]).cuda()

        # dataList contains true itemID, convert to idx
        pred_idx, y_true = [], []
        for itemID, label, _ in rl:
            pred_idx.append(item_idx_map[itemID])
            y_true.append(label)

         
        user_features_l, user_feature_values_l,  item_features_l, item_feature_values_l, user_cat_nums_l, label_l = selected_concat(user_feat, user_feat_values, user_cat_nums, \
                                            all_item_features, all_item_feature_values, pred_idx, y_true)
        
        
        if model_name in ['DCRS']:
            prediction, t_s_loss, ui_loss, _, uc_addi_loss, debug_info = model(user_features_l, user_feature_values_l, item_features_l, item_feature_values_l,label_l) 
        else:
            prediction, _ = model(user_features_l, user_feature_values_l, item_features_l, item_feature_values_l, label_l)

        pred_score = torch.sigmoid(prediction).detach().cpu().numpy()
        #pred_score = torch.sigmoid(debug_info['ui_pred_1']).detach().cpu().numpy()
        label_l = label_l.cpu().numpy()

        # for case only contains one class , just ignore
        if (np.sum(label_l) >0) and (len(rl)-np.sum(label_l) >0):
            tmp_auc = roc_auc_score(y_true=label_l, y_score=pred_score)
            uauc_l.append([tmp_auc, len(rl)])

        if all_user_prediction is None:
            all_user_prediction = pred_score
            all_user_label = label_l
        else:
            all_user_prediction = np.concatenate((all_user_prediction, pred_score), axis=0)
            all_user_label = np.concatenate((all_user_label,label_l), axis=0)

    
    uauc = get_weighted_auc(uauc_l)
    test_auc = roc_auc_score(y_true=all_user_label, y_score=all_user_prediction)

    return [uauc, test_auc]


  

def candidate_ranking(model, model_name, valid_dataList, test_dataList, train_dict, \
                      user_feature,item_feature, all_item_features, all_item_feature_values,\
                       topN, user_hist_cat_dist, num_groups, user_candidate=None, snake_merge_cat=False):
    """evaluate the auc & top-n diversity"""

    # first calculate all auc

    item_idx = list(item_feature.keys())
    item_idx_map = {item_idx[i]: i for i in range(len(item_idx))}

    valid_auc_re = calculate_auc(valid_dataList, model, model_name,  user_feature, user_hist_cat_dist, item_idx, all_item_features, all_item_feature_values)
    test_auc_re = calculate_auc(test_dataList, model, model_name,  user_feature, user_hist_cat_dist, item_idx, all_item_features, all_item_feature_values)

    if user_candidate is not None:
        print('[candiate_ranking] use user_candidate!') 

    # calucate diversity-relatedd metric
    user_pred, user_gt_valid, user_gt_test = [], [], []
    for userID, _ in test_dataList.items():
         
        # get user_gt_valie
        user_gt_valid.append([itemID for itemID, label, _ in valid_dataList[userID] if label>0])
        user_gt_test.append([itemID for itemID, label, _ in test_dataList[userID] if label>0])

        if user_candidate is not None:
           candidates = user_candidate[userID]
        else:
           # only add items that are not train data
           if userID in train_dict:
               trained_item = {e[0]:1 for e in train_dict[userID]}
           else:
               trained_item = {}
           candidates = [tmp_id for tmp_id in item_idx if tmp_id not in trained_item]

        candidates_idx = [item_idx_map[c_itemID] for c_itemID in candidates]
        user_feat = torch.tensor(np.array(user_feature[userID][0])).cuda()
        user_feat_values = torch.tensor(np.array(user_feature[userID][1], dtype=np.float32)).cuda()
        user_cat_nums = torch.tensor(user_hist_cat_dist[userID]).cuda()


        user_features_l, user_feature_values_l,  item_features_l, item_feature_values_l, user_cat_nums_l, fake_label_l = selected_concat(user_feat, user_feat_values, user_cat_nums, \
                                            all_item_features, all_item_feature_values, candidates_idx)

       
        if model_name in ['DCRS']:
            prediction, t_s_loss, ui_loss,_, uc_addi_loss, debug_info = model(user_features_l, user_feature_values_l, item_features_l, item_feature_values_l,fake_label_l) 
        else:
            prediction, _ = model(user_features_l, user_feature_values_l, item_features_l, item_feature_values_l, fake_label_l)

        if snake_merge_cat:
            assert model_name in ['DCRS'] 
            prediction = prediction.detach().cpu().numpy().tolist()
            candidate_cat_pred = debug_info['uc_pred'].detach().cpu().numpy().tolist()
            item_cat_dist = debug_info['item_cat_dist'].detach().cpu().numpy().tolist()
            user_pred_topK = get_predTopK_SM(candidates, prediction, item_cat_dist, candidate_cat_pred, topN)
            user_pred.append(user_pred_topK)

        else:
           pred_prob =  torch.sigmoid(prediction) 

           _, indices = torch.topk(pred_prob, topN[-1])
           pred_items = torch.tensor(candidates)[indices].cpu().numpy().tolist()

           user_pred.append(pred_items)

    
    print('~~~~~~~~test~~~~~~~~~') 
    valid_recall = computeTopNAccuracy(user_gt_valid, user_pred, topN)
    test_recall = computeTopNAccuracy(user_gt_test, user_pred, topN)
    calibration_results = calibration(item_feature, test_dataList, train_dict, user_pred, topN, num_groups)

    # check for recall rate of category
    valid_cat_recall = computeTopNCatRecall(user_gt_valid, user_pred, topN, item_feature, num_group=num_groups) 
    test_cat_recall = computeTopNCatRecall(user_gt_test, user_pred, topN, item_feature, num_group=num_groups) 
                  
    return valid_auc_re, test_auc_re, calibration_results, valid_recall, test_recall, valid_cat_recall, test_cat_recall



def RMSE(model, model_name, dataloader):
    RMSE = np.array([], dtype=np.float32)
    for user_features, user_feature_values, item_features, item_feature_values,cur_user_cat_num_fs, label in dataloader:
        user_features = user_features.cuda()
        user_feature_values = user_feature_values.cuda()
        item_features = item_features.cuda()
        item_feature_values = item_feature_values.cuda()
        cur_user_cat_num_fs = cur_user_cat_num_fs.cuda()
        label = label.cuda()
        
        
        if  model_name in ['DCRS']:
            prediction, t_s_loss, ui_loss, ui_adver_loss, uc_addi_loss, _ = model(user_features, user_feature_values, item_features, item_feature_values,label) 
        else:
           prediction, _ = model(user_features, user_feature_values, item_features, item_feature_values, label)
        
        prediction = prediction.clamp(min=-1.0, max=1.0)
        SE = (prediction - label).pow(2)
        RMSE = np.append(RMSE, SE.detach().cpu().numpy())

    return np.sqrt(RMSE.mean())


   


def get_weighted_auc(uauc_l):
    sum_v, num = 0, 0
    for auc_v, n in uauc_l:
        sum_v += auc_v*n
        num +=n
    return sum_v/float(num+0.00001)



    

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def get_cat_dist(item_feature, item_l , num_groups):
    history_cate = np.array([0]*num_groups, dtype=np.float32)
    for itemID in item_l:
            history_cate += np.array(item_feature[itemID][1][-num_groups:], dtype=np.float32)
    if len(item_l) >0:
        history_cate = history_cate/len(item_l)

    return history_cate

def get_catSet(itemL ,item_features, num_groups):

    cat_set = set()
    for itemID in itemL:
            cat_l = item_features[itemID][1][-num_groups:] 
            cat_rep = ''.join(['1' if cat_l[i]>0 else '0' for i in range(len(cat_l))])
            cat_set.add(cat_rep)
    return cat_set 

def computeTopNCatRecall(GroundTruth, predictedIndices, topN, item_features, num_group):

    #
    user_cat_gt = []
    for i in range(len(predictedIndices)): # for a user
        cat_gt = get_catSet(GroundTruth[i], item_features, num_group)
        user_cat_gt.append(cat_gt)

        
    recall = []
    for index in range(len(topN)):
        sumForRecall = 0
        Valid_GT_Num = 0
        for i in range(len(predictedIndices)): # for a user
            if len(user_cat_gt[i]) >0:
                Valid_GT_Num +=1
                pred_cat_set = get_catSet(predictedIndices[i][0:topN[index]],  item_features, num_group)
                sumForRecall += len(pred_cat_set & user_cat_gt[i]) / len(user_cat_gt[i])

        recall.append(round(sumForRecall /Valid_GT_Num, 4))
    print('[computeTopNCatRecall] Valid_GT_Num: %d, Pred_user:%d'%(Valid_GT_Num, len(predictedIndices)))
    return recall


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    recall = [] 
    NDCG = [] 

    for index in range(len(topN)):
        sumForRecall = 0
        sumForNdcg = 0
        Valid_GT_num = 0

        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                Valid_GT_num +=1
                userHit = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        hit.append(predictedIndices[i][j])
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg


        
        recall.append(round(sumForRecall / Valid_GT_num, 4))
        NDCG.append(round(sumForNdcg / Valid_GT_num, 4))

    
    print('[computeTopNCatRecall] Valid_GT_Num: %d, Pred_user:%d'%(Valid_GT_num, len(predictedIndices)))
        
    return recall, NDCG 


    

def calibration(item_feature, test_dataList, train_dict, user_pred, topN, num_groups, is_din=False):
    C_EN = []
    CC = []
    assert len(test_dataList) == len(user_pred)
    user_cat_collect = {}
    
    for i, userID in enumerate(test_dataList.keys()):

        if userID in train_dict: 
            history_items = train_dict[userID]
        else:
            history_items = {}
        pos_history_cate = np.array([0]*num_groups, dtype=np.float32)
        pos_num = 0
        neg_history_cat = np.array([0]*num_groups, dtype=np.float32)
        all_history_cat = np.array([0]*num_groups, dtype=np.float32)
        
        for itemID, label in history_items:
            if is_din :
                cur_item_cat_v = np.array(item_feature[itemID], dtype=np.float32)
            else:
                cur_item_cat_v = np.array(item_feature[itemID][1][-num_groups:], dtype=np.float32) 
            
            if label>0:
               # only consider postive albels
               pos_history_cate += cur_item_cat_v
               pos_num +=1
            else:
                neg_history_cat += cur_item_cat_v
            all_history_cat += cur_item_cat_v

        test_pos_cat = np.array([0]*num_groups, dtype=np.float32)
        test_all_cat = np.array([0]*num_groups, dtype=np.float32)
        for itemID, label, _ in test_dataList[userID]:
            if label>0:
                test_pos_cat +=  np.array(item_feature[itemID][1][-num_groups:], dtype=np.float32)
            test_all_cat +=  np.array(item_feature[itemID][1][-num_groups:], dtype=np.float32) 
        
        
        C_EN_u = []
        CC_u = [] # cat_coverage
        rec_cate_l = []
        for n in topN:
            rec_list = user_pred[i][:n]
            rec_cate = np.array([0]*num_groups, dtype=np.float32)
            for itemID in rec_list:
                if is_din:
                    rec_cate += np.array(item_feature[itemID], dtype=np.float32)
                else:
                    rec_cate += np.array(item_feature[itemID][1][-num_groups:], dtype=np.float32)
            rec_cate = rec_cate/len(rec_list)
            entropy = -sum([e*np.log(e+1e-12) for e in rec_cate])
            rec_cate_l.append(rec_cate)
            

            C_EN_u.append(entropy)
            cur_cc = np.count_nonzero(rec_cate)/num_groups
            CC_u.append(cur_cc) 

        C_EN.append(C_EN_u)
        CC.append(CC_u)
        user_cat_collect[userID]= [[pos_history_cate, neg_history_cat, all_history_cat], rec_cate_l, test_pos_cat, test_all_cat]
   
    C_EN = np.around(np.mean(C_EN, 0), 4).tolist()
    CC = np.around(np.mean(CC, 0), 4).tolist()
    return  C_EN, user_cat_collect, CC


def calibration_pure(item_feature, test_dataList, user_pred, topN, num_groups, is_din=False):
    C_EN = []
    CC = []
    print('test_dataList: ', len(test_dataList))
    print('user_pred: ', len(user_pred))
    for i, userID in enumerate(test_dataList.keys()):
        C_EN_u = []
        CC_u = [] # cat_coverage

        rec_cate_l = []
        if len(user_pred[i])<1:
            continue
        for n in topN:
            rec_list = user_pred[i][:n]
            rec_cate = np.array([0]*num_groups, dtype=np.float32)
            for itemID in rec_list:
                if is_din:
                    rec_cate += np.array(item_feature[itemID], dtype=np.float32)
                else:
                    rec_cate += np.array(item_feature[itemID][1][-num_groups:], dtype=np.float32)
            rec_cate = rec_cate/len(rec_list)
            entropy = -sum([e*np.log(e+1e-12) for e in rec_cate])
            C_EN_u.append(entropy)

            cur_cc = np.count_nonzero(rec_cate)/num_groups
            CC_u.append(cur_cc) 


        C_EN.append(C_EN_u)
        CC.append(CC_u)
    C_EN = np.around(np.mean(C_EN, 0), 4).tolist()
    CC = np.around(np.mean(CC, 0), 4).tolist()
    return  C_EN, CC

def print_results(train_RMSE, valid_auc_re, test_auc_re, calibration_results, valid_recall, test_recall, valid_cat_recall, test_cat_recall):
    """output the evaluation results."""
    if train_RMSE is not None:
        print("[Train]: RMSE: {:.4f}".format(train_RMSE))
    if valid_auc_re is not None: 
        print("[Valid]: AUC: {} UAUC: {} ".format(valid_auc_re[1], valid_auc_re[0]))
    
    if test_auc_re is not None: 
        print("[Test]: AUC: {} UAUC: {} ".format(test_auc_re[1], test_auc_re[0]))

    if calibration_results is not None:
        print("[Calibration]:  C_EN: {}, CC: {}".format(
                            '-'.join([str(x) for x in calibration_results[0]]),
                            '-'.join([str(x) for x in calibration_results[2]])
                            ))
    if valid_recall is not None:
        print("[Valid_RECALL]: Recall: {} NDCG: {}".format(
                            '-'.join([str(x) for x in valid_recall[0]]), 
                            '-'.join([str(x) for x in valid_recall[1]])))

    if test_recall is not None:
        print("[Test_RECALL]: Recall: {} NDCG: {}".format(
                            '-'.join([str(x) for x in test_recall[0]]), 
                            '-'.join([str(x) for x in test_recall[1]])))

    if valid_cat_recall is not None:
        print("[Valid_Cat_Recall]: catRecall: {} ".format(
                            '-'.join([str(x) for x in valid_cat_recall]))) 

    if test_cat_recall is not None:
        print("[Test_Cat_Recall]: catRecall: {} ".format(
                            '-'.join([str(x) for x in test_cat_recall]))) 

