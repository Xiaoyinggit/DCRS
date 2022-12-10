import os
import time
import argparse
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
#from tensorboardX import SummaryWriter

from models.NFM import NFM
from models.DCRS import DCRS
import evaluate
import data_utils
import utils 

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
    type=str,
    default="ml_1m",
    help="dataset option: 'ml_1m'")
parser.add_argument("--num_groups",
    type=int,
    default=18,
    help="item group number in the dataset, must change for each dataset")
parser.add_argument("--num_user_features",
    type=int,
    default=5,
    help="user feature num in the dataset, must change for each dataset")
parser.add_argument("--model", 
    type=str,
    default="DCRS",
    help="model option: DCRS")
parser.add_argument("--optimizer",
    type=str,
    default="Adagrad",
    help="optimizer option: 'Adagrad', 'Adam', 'SGD', 'Momentum'")
parser.add_argument("--data_path",
    type=str,
    default="../data/",  
    help="load data path")
parser.add_argument("--model_path", 
    type=str,
    default="./saved_models/",
    help="saved model path")
parser.add_argument("--activation_function",
    type=str,
    default="relu",
    help="activation_function option: 'relu', 'sigmoid', 'tanh', 'identity'")
parser.add_argument("--lr", 
    type=float, 
    default=0.05, 
    help="learning rate")
parser.add_argument("--dropout", 
    default='[0.5, 0.2]',  
    help="dropout rate for FM and MLP")
parser.add_argument("--batch_size", 
    type=int, 
    default=128, 
    help="batch size for training")
parser.add_argument("--epochs", 
    type=int,
    default=300, 
    help="training epochs")
parser.add_argument("--hidden_factor", 
    type=int,
    default=64, 
    help="predictive factors numbers in the model")
parser.add_argument("--layers", 
    default='[64]', 
    help="size of layers in MLP model, '[]' is NFM-0")
parser.add_argument("--lamda", 
    type=float, 
    default=0.1, 
    help="regularizer for bilinear layers")
parser.add_argument("--topN", 
    default='[5,10,20,50]',  
    help="the recommended item num")
parser.add_argument("--batch_norm", 
    type=int,
    default=1,   
    help="use batch_norm or not. option: {1, 0}")
parser.add_argument("--out", 
    default=True,
    help="save model or not")
parser.add_argument("--gpu", 
    type=str,
    default="0",
    help="gpu card ID")
parser.add_argument("--snake_merge_cat", 
    type=bool,
    default=False,
    help="whether use snake merge")
parser.add_argument(
    '--DCRSPara',
    default='0:0',
    type=lambda x: {k:float(v) for k,v in (i.split(':') for i in x.split(','))},
    help='comma-separated field:position pairs, e.g. Date:0,Amount:2,Payee:5,Memo:9'
)
args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


#############################  PREPARE DATASET #########################
start_time = time.time()
train_path = args.data_path + '{}/train_list.npy'.format(args.dataset)
valid_path = args.data_path + '{}/valid_list.npy'.format(args.dataset)
test_path = args.data_path + '{}/test_list.npy'.format(args.dataset)
user_hist_cat_path = args.data_path +'{}/user_hist_cat.npy'.format(args.dataset)
user_feature_path = args.data_path + '{}/user_feature_file.npy'.format(args.dataset)
item_feature_path = args.data_path + '{}/item_feature_file.npy'.format(args.dataset)
user_candidate_path = args.data_path + '{}/user_recall_candidates.npy'.format(args.dataset)


user_feature, item_feature, num_features, num_UF, num_IF, num_users, num_items, feature_map = data_utils.map_features(user_feature_path, item_feature_path)
### by default, the last features in item_feature are the group features. the group num is set in the args.
num_groups = args.num_groups

train_dataset = data_utils.FMData(train_path, user_feature, item_feature, user_hist_cat_path, num_groups=args.num_groups, model_name=args.model)
valid_dataList = data_utils.loadTestData(valid_path)
test_dataList = data_utils.loadTestData(test_path)
confounder_prior = train_dataset.confounder_prior 
user_candidates = None

train_loader = data.DataLoader(train_dataset, drop_last=True,
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

all_item_features, all_item_feature_values = evaluate.pre_candidate_ranking(item_feature)
print('data ready. costs ' + time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))
##############################  CREATE MODEL ###########################

model_save_path_perfix = '{}_{}_{}lr_{}bs_{}dropout_{}lamda_{}layers'.format(
                     args.model, args.dataset, args.lr, \
                    args.batch_size, args.dropout, args.lamda, args.layers)
if args.model == 'NFM':
    model = NFM(num_features, args.hidden_factor, 
        args.activation_function, eval(args.layers), args.batch_norm,  eval(args.dropout), num_groups=num_groups, pretrain_FM=None)
elif args.model == 'DCRS':
    model = DCRS(num_features, num_groups, args.hidden_factor, 
        args.activation_function, eval(args.layers), args.batch_norm,  eval(args.dropout), DCRS_para=args.DCRSPara, pretrain_FM=None)
    model_save_path_perfix = '{}_{}_{}lr_{}bs_{}dropout_{}lamda_{}layers_{}DCRSPara'.format(
                    args.model, args.dataset, args.lr, \
                    args.batch_size, args.dropout, args.lamda, args.layers, json.dumps(args.DCRSPara)) 
else:
    raise Exception('model not implemented!')
    
model.cuda()
if args.optimizer == 'Adagrad':
    optimizer = optim.Adagrad(
        model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == 'Momentum':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)

#writer = SummaryWriter('tensorlog') # for visualization

###############################  TRAINING ############################

count, best_recall, best_gt_entropy = 0, -100, 100
best_test_result = []
for epoch in range(args.epochs):
    model.train() # Enable dropout and batch_norm
    start_time = time.time()
    #train_loader.dataset.ng_sample()
    for user_features, user_feature_values, item_features, item_feature_values,cur_user_cat_num_fs, label in train_loader:

        user_features = user_features.cuda()
        user_feature_values = user_feature_values.cuda()
        item_features = item_features.cuda()
        item_feature_values = item_feature_values.cuda()
        cur_user_cat_num_fs = cur_user_cat_num_fs.cuda()
        label = label.cuda()

        model.zero_grad()
        debug_str = ''
        if args.model in ['DCRS']:
            prediction, t_s_loss, ui_loss, ui_adver_loss, uc_addi_loss, debug_info = model(user_features, user_feature_values, item_features, item_feature_values,label) 
            t_loss = t_s_loss + ui_loss + ui_adver_loss + uc_addi_loss
            debug_str = 't_s_loss:%f, ui_loss:%f, ui_adver_loss:%f, uc_addi_loss:%f, ui_adver_hit_r:%f, uc_adver_hit_r:%f:'%(t_s_loss, ui_loss,  ui_adver_loss, uc_addi_loss, debug_info['adver_inverse_1'], debug_info['adver_inverse_0'])
        else:
            prediction, t_loss= model(user_features, user_feature_values, item_features, item_feature_values, label)
        norm_loss =args.lamda * model.embeddings.weight.norm()
        loss = t_loss + norm_loss
        debug_str = 'loss:%f, t_loss:%f, norm_loss:%f'%(loss,t_loss,norm_loss) + ',' + debug_str
        loss.backward()
        optimizer.step()
        # writer.add_scalar('data/loss', loss.item(), count)
        count += 1

    
    if epoch % 1 == 0:
        
        model.eval()
        train_RMSE = evaluate.RMSE(model, args.model, train_loader)
        valid_auc_re, test_auc_re, calibration_results, valid_recall, test_recall, valid_cat_recall, test_cat_recall = evaluate.candidate_ranking(model, args.model, valid_dataList, \
                    test_dataList, train_dataset.train_dict, user_feature, item_feature,\
                    all_item_features, all_item_feature_values, eval(args.topN), \
                    train_dataset.user_hist_cat_dist ,num_groups, user_candidate=user_candidates,
                    snake_merge_cat=args.snake_merge_cat)
 
        print('---'*18)
        print("Runing Epoch {:03d} ".format(epoch) + "costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
        print('[TRAIN] '+debug_str)
        #if 'DNFM' in args.model:
        #    # debug cat ind
        #    for cat_n, id in cat2ind.items():
        #        print('~~~~~~~cat_name:%s: %d~~~~~'%(cat_n, id))
        #        id_emb = model.get_fid_emb(id)
        #       print(id_emb)
 

 
        evaluate.print_results(train_RMSE, valid_auc_re, test_auc_re, calibration_results, valid_recall, test_recall, valid_cat_recall, test_cat_recall)

        if ( valid_recall[0][0] > best_recall): # auc for selection 
            best_recall, best_epoch =  valid_recall[0][0],  epoch
            best_test_result = ( test_auc_re ,calibration_results) 
        
        print("------------ model, saving...------------")
        if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                model_save_path = args.model_path +'/'+ model_save_path_perfix +'{}.epoch.pth'.format(epoch)
                print('model_save_path: ', model_save_path)

                torch.save(model, model_save_path)
                np.save('debug/%s_user_cat_collect_%depoch_%sdataset.npy'%(model_save_path_perfix,epoch,args.dataset), calibration_results[1])

print("End. Best epoch {:03d}".format(best_epoch))
evaluate.print_results(None, None, best_test_result[0], best_test_result[1], None, None, None, None)


