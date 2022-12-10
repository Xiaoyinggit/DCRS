import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
torch.set_printoptions(profile="full")



class DCRS(nn.Module):
    def __init__(self, num_features, num_groups, num_factors, 
        act_function, layers, batch_norm, drop_prob, DCRS_para, pretrain_FM):
        super(DCRS, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        act_function: activation function for MLP layer,
        layers: list of dimension of deep layers,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        pretrain_FM: the pre-trained FM weights.
        """
        self.num_features = num_features
        self.num_groups = num_groups
        self.num_factors = num_factors
        self.act_function = act_function
        self.layers = layers
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        self.pretrain_FM = pretrain_FM
        self.DCRS_para = DCRS_para

        
        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))        
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        MLP_module = []
        in_dim = num_factors
        for dim in self.layers:
            out_dim = dim
            MLP_module.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

            if self.batch_norm:
                MLP_module.append(nn.BatchNorm1d(out_dim))
            if self.act_function == 'relu':
                MLP_module.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                MLP_module.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                MLP_module.append(nn.Tanh())

            MLP_module.append(nn.Dropout(drop_prob[-1]))
        self.deep_layers = nn.Sequential(*MLP_module)

        self.predict_size = layers[-1] if layers else num_factors
        self.ui_prediction = nn.Linear(self.predict_size//2, 1, bias=False)
        self.prediction = nn.Linear(self.predict_size, 1, bias=False)
        self.cat_pred_layer = nn.Linear(self.predict_size//2, self.num_groups, bias=False)

        # criterion
        self.ui_criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
        self.adverse_criterion= nn.CrossEntropyLoss(reduction='none')
        self.debug_info = {}

        self._init_weight_()

    def _init_weight_(self):
        """ Try to mimic the original weight initialization. """
        if self.pretrain_FM:
            self.embeddings.weight.data.copy_(
                            self.pretrain_FM.embeddings.weight)
            self.biases.weight.data.copy_(
                            self.pretrain_FM.biases.weight)
            self.bias_.data.copy_(self.pretrain_FM.bias_)
        else:
            nn.init.normal_(self.embeddings.weight, std=0.01)
            nn.init.constant_(self.biases.weight, 0.0)

        # for deep layers
        if len(self.layers) > 0:
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(self.ui_prediction.weight)
            nn.init.xavier_normal_(self.prediction.weight)
            nn.init.xavier_normal_(self.cat_pred_layer.weight)
        else:
            nn.init.constant_(self.ui_prediction.weight, 1.0)
            nn.init.constant_(self.prediction.weight, 1.0)
            nn.init.constant_(self.cat_pred_layer.weight, 1.0)

    def get_adversary_loss(self, in_emb, cat_dist, is_inverse):

        # is_inverse = 1: cat_label=all zeros 
        if is_inverse ==1:
            in_emb.register_hook(lambda grad: -grad)  

        cat_pred_logits = self.cat_pred_layer(in_emb) / self.DCRS_para['temp']
        loss_vec = self.adverse_criterion(input=cat_pred_logits, target=cat_dist)
        cat_loss = torch.sum(loss_vec) 

        ## for debug, TODO debug.
        # cat_pred_prob = F.softmax(cat_pred_logits, dim=-1)
        # gt_cat_num = torch.sum(torch.gt(cat_dist, 0), dim =1).cpu().numpy()
        # max_cat_num = np.max(gt_cat_num)
        # _, cat_pred_prob_topK_indices = torch.topk(cat_pred_prob, k=max_cat_num, dim=1)
        # cat_pred_prob_topK_indices = cat_pred_prob_topK_indices.cpu().numpy().tolist()
        # pred_ind, wei = [], []
        # for i in range(len(cat_pred_prob_topK_indices)):
        #     for j in range(gt_cat_num[i]):
        #         pred_ind.append([i, cat_pred_prob_topK_indices[i][j]])
        #         wei.append(1.0/gt_cat_num[i])
        # pred_ind, wei = torch.tensor(pred_ind).cuda(), torch.tensor(wei).cuda()
        # hit_r = torch.sum(torch.gt(cat_dist[pred_ind[:,0], pred_ind[:,1]],0)*wei)/(torch.sum(wei)+0.0001)
        self.debug_info['adver_inverse_%d'%is_inverse] = 0 
        
        return cat_loss


    def get_fid_emb(self, id):
        id = torch.tensor(id).cuda()
        emb = self.embeddings(id)
        emb = torch.reshape(emb, (1, -1))
        return emb 


    def forward(self, user_features, user_feature_values, item_features, item_feature_values, label):
        features = torch.cat((user_features, item_features), dim=-1)
        feature_values = torch.cat((user_feature_values, item_feature_values), dim=-1)
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # cat_values
        item_cat_dist = feature_values[:, -self.num_groups:].squeeze(dim=-1)
        self.debug_info['item_cat_dist'] = item_cat_dist 

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM)
        if self.layers: # have deep layers
            FM = self.deep_layers(FM) # [B, layer_d]
        
        user_emb_item, user_emb_cat = torch.split(FM,self.predict_size//2 , dim =1) #[B, layer_d/2]
        
        # loss for each part: 
        ## ui_loss  & ui_adversary loss
        label = torch.reshape(label , (-1,1))
        ui_pred = self.ui_prediction(user_emb_item)
        self.debug_info['ui_pred_1'] = ui_pred.view(-1)
        ui_loss = self.ui_criterion(input=ui_pred, target=label)
        ## for user_emb_item, we hope it cannot predict category
        ui_adver_loss = self.get_adversary_loss(user_emb_item.clone(),item_cat_dist, is_inverse= 1)
        ui_loss = self.DCRS_para['ui']*ui_loss 
        ui_adver_loss = self.DCRS_para['ui_adver']*ui_adver_loss 


        ## uc_loss & uc_adversary_loss
        uc_adver_loss = self.get_adversary_loss(user_emb_cat.clone(),item_cat_dist, is_inverse=0)
        uc_addi_loss =  self.DCRS_para['uc_adver']*uc_adver_loss


        ## combination loss
        if self.DCRS_para['sg'] > 0:
            user_emb = torch.cat((user_emb_item.detach(), user_emb_cat), dim=1)
        else:
            user_emb = torch.cat((user_emb_item, user_emb_cat), dim=1)
        logits = self.prediction(user_emb) # prediction layer_d * 1
        pred_ui, pred_uc = torch.split(self.prediction.weight,self.predict_size//2 , dim =1) #[1, predict_size//2]
        self.debug_info['ui_pred'] = torch.matmul(user_emb_item, pred_ui.t()).view(-1)
        self.debug_info['uc_pred'] =  torch.matmul(user_emb_cat, pred_uc.t()).view(-1)
        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        logits = logits + feature_bias + self.bias_
        t_loss = self.criterion(input=logits, target=label)

        return logits.view(-1), t_loss, ui_loss, ui_adver_loss, uc_addi_loss, self.debug_info
