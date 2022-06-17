import os
import sys
import numpy as np
import pandas as pd

from tqdm import tqdm
import json
import pickle
import natsort

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

from collections import defaultdict

class HEMHelper():
    """
        Help computation ids for Hard Example Mining.
        
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.NON_HEM, self.HEM = (0, 1)
        self.IB_CLASS, self.OOB_CLASS = (0, 1)
        self.bsz = self.args.batch_size
        self.cnt = 0
        self.alpha = self.args.alpha

        self.method_idx = 0

    def set_method(self, method):
        if method in ['hem-softmax-offline', 'hem-voting-offline', 'hem-vi-offline', 'all-offline']:
            self.method = 'hem-vi'
        else:
            self.method = method

    def set_restore_path(self, restore_path):
        self.restore_path = restore_path

    def get_target_hem_count(self):

        with open(os.path.join(self.restore_path, 'DATASET_COUNT.json')) as file:
            try:
                json_data = json.load(file)
                return json_data['target_hem_count']['rs'], json_data['target_hem_count']['nrs']

            except ValueError as e:
                print('Parsing Fail, Error: {}'.format(e))
                return None

    def get_target_patient_hem_count(self, patient_no): # rule for target count of each patients
        # hem
        patient_rs_count, patient_nrs_count = 0,0
        patient_rs_ratio, patient_nrs_ratio = 0,0
        target_rs_cnt, target_nrs_cnt = self.get_target_hem_count()
        
        with open(os.path.join(self.restore_path, 'PATIENTS_DATASET_COUNT.json')) as file:
            try:
                json_data = json.load(file)
                patient_rs_ratio, patient_nrs_ratio = json_data[patient_no]['rs_ratio'], json_data[patient_no]['nrs_ratio']

            except ValueError as e:
                print('Parsing Fail, Error: {}'.format(e))
                return None

            patient_nrs_count = int(target_nrs_cnt * patient_nrs_ratio)
            patient_rs_count = int(patient_nrs_count * self.args.IB_ratio)

        return patient_rs_count, patient_nrs_count

    def set_ratio(self, hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df, patient_no=None):
        
        if patient_no :
            target_rs_cnt, target_nrs_cnt = self.get_target_patient_hem_count(patient_no)
        else : # if patinet_no = None
            target_rs_cnt, target_nrs_cnt = self.get_target_hem_count()

        # df 정렬
        hard_neg_df = hard_neg_df.sort_values(by='Img_path')
        hard_pos_df = hard_pos_df.sort_values(by='Img_path')
        vanila_neg_df = vanila_neg_df.sort_values(by='Img_path')
        vanila_pos_df = vanila_pos_df.sort_values(by='Img_path')

        # HEM 표기
        hard_neg_df['HEM'] = [1]*len(hard_neg_df)
        hard_pos_df['HEM'] = [1]*len(hard_pos_df)
        vanila_neg_df['HEM'] = [0]*len(vanila_neg_df)
        vanila_pos_df['HEM'] = [0]*len(vanila_pos_df)


        # train data 수 만큼 hem data extract
        if len(hard_pos_df) > target_nrs_cnt: hard_pos_df = hard_pos_df.sample(n=target_nrs_cnt, replace=False, random_state=self.args.random_seed)
        if len(hard_neg_df) > target_rs_cnt: hard_neg_df = hard_neg_df.sample(n=target_rs_cnt, replace=False, random_state=self.args.random_seed) 

        target_len_vanila_pos = target_nrs_cnt - len(hard_pos_df)
        target_len_vanila_neg = target_rs_cnt - len(hard_neg_df)

        # if target_len_vanila_pos < 0: target_len_vanila_pos = 0
        # if target_len_vanila_neg < 0: target_len_vanila_neg = 0
        
        try:
            vanila_pos_df = vanila_pos_df.sample(n=target_len_vanila_pos, replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, hem_oob 개
        except:
            vanila_pos_df = vanila_pos_df.sample(frac=1, replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, 전체 oob_df

        try:
            vanila_neg_df = vanila_neg_df.sample(n=target_len_vanila_neg, replace=False, random_state=self.args.random_seed) # 중복뽑기x, random seed 고정, target_ib_assets_df_len 개
        except:
            vanila_neg_df = vanila_neg_df.sample(frac=1, replace=False, random_state=self.args.random_seed)

        method = ['softmax_diff_small', 'softmax_diff_large', 'voting', 'mi_small', 'mi_large']  
        
        save_data = {
            method[self.method_idx]: {
                "rs": {
                    "hem": len(hard_neg_df),
                    "vanila": len(vanila_neg_df)
                },
                "nrs": {
                    "hem": len(hard_pos_df),
                    "vanila": len(vanila_pos_df)
                }
            }
        }

        save_data.update(save_data)

        print('save_data', save_data)

        self.method_idx += 1


        with open(os.path.join(self.restore_path, 'DATASET_COUNT.json'), 'r+') as f:
            data = json.load(f)
            if patient_no not in data :
                data[patient_no] = save_data
            else : 
                data[patient_no].update(save_data)

            f.seek(0)
            json.dump(data, f, indent=2)
            # data.update(save_data)

        # with open(os.path.join(self.restore_path, 'DATASET_COUNT.json'), 'w') as f:
        #     json.dump(data, f, indent=2)


        final_pos_assets_df = pd.concat([hard_pos_df, vanila_pos_df])[['Img_path', 'GT', 'HEM']]
        final_neg_assets_df = pd.concat([hard_neg_df, vanila_neg_df])[['Img_path', 'GT', 'HEM']]

        # sort & shuffle
        final_assets_df = pd.concat([final_pos_assets_df, final_neg_assets_df]).sort_values(by='Img_path', axis=0, ignore_index=True)
        print('\tSORT final_assets HEAD\n', final_assets_df.head(20), '\n\n')

        final_assets_df = final_assets_df.sample(frac=1, random_state=self.args.random_seed).reset_index(drop=True)
        print('\tSHUFFLE final_assets HEAD\n', final_assets_df.head(20), '\n\n')
        
        final_assets_df.columns = ['img_path', 'class_idx', 'HEM']


        return final_assets_df
        
    def compute_hem(self, *args):
        if self.method == 'hem-vi':
            return self.hem_vi(*args)
        elif self.method == 'hem-emb-online':
            if self.args.emb_type == 1 or self.args.emb_type == 2:
                return self.hem_cos_hard_sim(*args)
            elif self.args.emb_type == 3:
                return self.hem_cos_hard_sim2(*args)
            elif self.args.emb_type == 4:
                return self.hem_cos_hard_sim_only(*args)
        else: # exception
            return None

    def hem_vi(self, model, dataset): # MC dropout
        print('hem_mc methods')

        # Function to enable the dropout layers during test-time
        def enable_dropout(model):
            
            ### 2. summary model
            # print('\n\n==== MODEL SUMMARY ====\n\n')
            # summary(model, (3,224,224))

            dropout_layer = []
            for m in model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    dropout_layer.append(m)
            
            if self.args.n_dropout != 1 :
                dropout_layer[-1].train() # only last layer to train

            # print('==== dropout_layer ====')
            # print(dropout_layer)

        # init for parameter for hem methods
        img_path_list = []
        gt_list = []

        d_loader = DataLoader( # validation
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
        )
        
        img_path_list = dataset.img_list
        gt_list = dataset.label_list
        
        ### 0. MC paramter setting
        n_classes = 2
        forward_passes = self.args.n_dropout
        n_samples = len(img_path_list)
        
        dropout_predictions = np.empty((0, n_samples, n_classes)) 
        softmax = nn.Softmax(dim=1)

        ## 1. MC forward
        for cnt in tqdm(range(forward_passes), desc='MC FORWARDING ... '):
            predictions = np.empty((0, n_classes))
            model.eval()
            enable_dropout(model)
            
            print('{}th MC FORWARDING ...'.format(cnt+1))
            
            for data in tqdm(d_loader, desc='processing...'):
                with torch.no_grad():
                    y_hat = model(data[1].cuda())
                    y_hat = softmax(y_hat)

                predictions = np.vstack((predictions, y_hat.cpu().numpy()))

            # dropout predictions - shape (forward_passes, n_samples, n_classes)
            dropout_predictions = np.vstack((dropout_predictions,
                                        predictions[np.newaxis, :, :]))  

        
        '''
        #### mc assets 저장 start
        mc_assets_save_dir = os.path.join(self.restore_path, 'mc_assets')
        os.makedirs(mc_assets_save_dir, exist_ok=True)
        
        dropout_save_path = os.path.join(mc_assets_save_dir, 'dropout_predictions.npy')
        gt_list_save_path = os.path.join(mc_assets_save_dir, 'gt_list.pkl')
        img_path_list_save_path = os.path.join(mc_assets_save_dir, 'img_path_list.pkl')
        
        np.save(dropout_save_path, dropout_predictions)

        with open(gt_list_save_path, 'wb') as f:
            pickle.dump(gt_list, f)

        with open(img_path_list_save_path, 'wb') as f:
            pickle.dump(img_path_list, f)            
        #### mc assets 저장 end
        '''


        #### patient 별 hem extract 구현
        print(dropout_predictions.shape)

        if self.args.hem_per_patient:

            assets_df = pd.DataFrame(img_path_list, columns=['img_path'])
            assets_df['patient'] = assets_df.img_path.str.split('/').str[4]
            
            assets_df['gt'] = gt_list

            patients_list = list(set(assets_df['patient']))
            patients_list = natsort.natsorted(patients_list)

            '''
            softmax_diff_small_dic = defaultdict(list)
            softmax_diff_large_dic = defaultdict(list)
            vointing_dic = defaultdict(list)
            mi_small_dic = defaultdict(list)
            mi_large_dic = defaultdict(list)
            '''

            # it's final df of hem assets per methods (create init df) 
            hem_final_df_columns = ['img_path', 'class_idx', 'HEM'] # same columns of return set_ratio's df
            softmax_diff_small_hem_final_df = pd.DataFrame(columns=hem_final_df_columns)
            softmax_diff_large_hem_final_df = pd.DataFrame(columns=hem_final_df_columns)
            voting_hem_final_df = pd.DataFrame(columns=hem_final_df_columns)
            vi_small_hem_final_df = pd.DataFrame(columns=hem_final_df_columns)
            vi_large_hem_final_df = pd.DataFrame(columns=hem_final_df_columns)


            for patient in tqdm(patients_list, desc='Extract HEM Assets per patients ...'):
                print('Patinet : {}'.format(patient))

                self.method_idx = 0 

                patient_idx = assets_df.index[assets_df['patient'] == patient].tolist()
                
                patient_img_path_list = assets_df['img_path'].iloc[patient_idx].tolist()  
                patient_gt_list = assets_df['gt'].iloc[patient_idx].tolist()  
                patient_dropout_predictions = dropout_predictions[:, patient_idx, :] # patient_dropout_predictions.shape = (5, n_patient, 2)
                
                print(patient_dropout_predictions.shape)

                # extracting hem, apply hem extract mode
                if self.args.hem_extract_mode == 'hem-softmax-offline':
                    print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))
                    
                    hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_softmax_diff(patient_dropout_predictions, patient_gt_list, patient_img_path_list) # diff_small
                    hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df, patient)
                    softmax_diff_small_hem_final_df.append(hem_final_df, ignore_index=True)


                elif self.args.hem_extract_mode == 'hem-voting-offline':
                    print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))

                    hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_voting(patient_dropout_predictions, patient_gt_list, patient_img_path_list)
                    hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df, patient)
                    voting_hem_final_df.append(hem_final_df, ignore_index=True)


                elif self.args.hem_extract_mode == 'hem-vi-offline':
                    print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))

                    hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_mutual_info(patient_dropout_predictions, patient_gt_list, patient_img_path_list) # large
                    hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df, patient)
                    mi_large_dic_final_df.append(hem_final_df, ignore_index=True)

                elif self.args.hem_extract_mode == 'all-offline':
                    print('==> ver_1')
                    hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_softmax_diff(patient_dropout_predictions, patient_gt_list, patient_img_path_list, 'diff_small', ver='ver_1')
                    hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df, patient)
                    softmax_diff_small_hem_final_df = softmax_diff_small_hem_final_df.append(hem_final_df, ignore_index=True) # append per patients
                    '''
                    softmax_diff_small_dic['hard_neg_df'].append(hard_neg_df) # //20개 환자
                    softmax_diff_small_dic['hard_pos_df'].append(hard_pos_df) # //20개 환자
                    softmax_diff_small_dic['vanila_neg_df'].append(vanila_neg_df) # //20개 환자
                    softmax_diff_small_dic['vanila_pos_df'].append(vanila_pos_df) # //20개 환자
                    '''

                    hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_softmax_diff(patient_dropout_predictions, patient_gt_list, patient_img_path_list, 'diff_large', ver='ver_1')
                    hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df, patient)
                    softmax_diff_large_hem_final_df = softmax_diff_large_hem_final_df.append(hem_final_df, ignore_index=True) # append per patients
                    '''
                    softmax_diff_large_dic['hard_neg_df'].append(hard_neg_df) # //20개 환자
                    softmax_diff_large_dic['hard_pos_df'].append(hard_pos_df) # //20개 환자
                    softmax_diff_large_dic['vanila_neg_df'].append(vanila_neg_df) # //20개 환자
                    softmax_diff_large_dic['vanila_pos_df'].append(vanila_pos_df) # //20개 환자
                    '''

                    hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_voting(patient_dropout_predictions, patient_gt_list, patient_img_path_list)
                    hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df, patient)
                    voting_hem_final_df = voting_hem_final_df.append(hem_final_df, ignore_index=True) # append per patients
                    '''
                    vointing_dic['hard_neg_df'].append(hard_neg_df) # //20개 환자
                    vointing_dic['hard_pos_df'].append(hard_pos_df) # //20개 환자
                    vointing_dic['vanila_neg_df'].append(vanila_neg_df) # //20개 환자
                    vointing_dic['vanila_pos_df'].append(vanila_pos_df) # //20개 환자
                    '''

                    hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_mutual_info(patient_dropout_predictions, patient_gt_list, patient_img_path_list, 'small')   
                    hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df, patient)
                    vi_small_hem_final_df = vi_small_hem_final_df.append(hem_final_df, ignore_index=True) # append per patients
                    '''
                    mi_small_dic['hard_neg_df'].append(hard_neg_df) # //20개 환자
                    mi_small_dic['hard_pos_df'].append(hard_pos_df) # //20개 환자
                    mi_small_dic['vanila_neg_df'].append(vanila_neg_df) # //20개 환자
                    mi_small_dic['vanila_pos_df'].append(vanila_pos_df) # //20개 환자
                    '''

                    hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_mutual_info(patient_dropout_predictions, patient_gt_list, patient_img_path_list, 'large')   
                    hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df, patient)
                    vi_large_hem_final_df = vi_large_hem_final_df.append(hem_final_df, ignore_index=True) # append per patients
                    '''
                    mi_large_dic['hard_neg_df'].append(hard_neg_df) # //20개 환자
                    mi_large_dic['hard_pos_df'].append(hard_pos_df) # //20개 환자
                    mi_large_dic['vanila_neg_df'].append(vanila_neg_df) # //20개 환자
                    mi_large_dic['vanila_pos_df'].append(vanila_pos_df) # //20개 환자
                    '''
            '''
            softmax_diff_small_hem_final_df = self.set_ratio(pd.concat(softmax_diff_small_dic['hard_neg_df']), pd.concat(softmax_diff_small_dic['hard_pos_df']), pd.concat(softmax_diff_small_dic['vanila_neg_df']), pd.concat(softmax_diff_small_dic['vanila_pos_df']))
            softmax_diff_large_hem_final_df = self.set_ratio(pd.concat(softmax_diff_large_dic['hard_neg_df']), pd.concat(softmax_diff_large_dic['hard_pos_df']), pd.concat(softmax_diff_large_dic['vanila_neg_df']), pd.concat(softmax_diff_large_dic['vanila_pos_df']))
            voting_hem_final_df = self.set_ratio(pd.concat(vointing_dic['hard_neg_df']), pd.concat(vointing_dic['hard_pos_df']), pd.concat(vointing_dic['vanila_neg_df']), pd.concat(vointing_dic['vanila_pos_df']))
            vi_small_hem_final_df = self.set_ratio(pd.concat(mi_small_dic['hard_neg_df']), pd.concat(mi_small_dic['hard_pos_df']), pd.concat(mi_small_dic['vanila_neg_df']), pd.concat(mi_small_dic['vanila_pos_df']))
            vi_large_hem_final_df = self.set_ratio(pd.concat(mi_large_dic['hard_neg_df']), pd.concat(mi_large_dic['hard_pos_df']), pd.concat(mi_large_dic['vanila_neg_df']), pd.concat(mi_large_dic['vanila_pos_df']))
            '''
            
            print('\nsoftmax_diff_small_hem_final_df\n', softmax_diff_small_hem_final_df)
            print('\n\nsoftmax_diff_large_hem_final_df\n', softmax_diff_large_hem_final_df)
            print('\n\nvoting_hem_final_df\n', voting_hem_final_df)
            print('\n\nvi_small_hem_final_df\n', vi_small_hem_final_df)
            print('\n\nvi_large_hem_final_df\n', vi_large_hem_final_df)

            if self.args.hem_extract_mode == 'all-offline':
                return softmax_diff_small_hem_final_df, softmax_diff_large_hem_final_df, voting_hem_final_df, vi_small_hem_final_df, vi_large_hem_final_df
            
            elif self.args.hem_extract_mode == 'hem-softmax-offline':
                return softmax_diff_small_hem_final_df
            
            elif self.args.hem_extract_mode == 'hem-voting-offline':
                return voting_hem_final_df
            
            elif self.args.hem_extract_mode == 'hem-vi-offline':
                return vi_large_hem_final_df

            else :
                return # 다른 메소드 일 경우 처리~

        # extracting hem, apply hem extract mode
        if self.args.hem_extract_mode == 'hem-softmax-offline':
            print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))
            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_softmax_diff(dropout_predictions, gt_list, img_path_list) # diff_small
            hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

        elif self.args.hem_extract_mode == 'hem-voting-offline':
            print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))
            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_voting(dropout_predictions, gt_list, img_path_list)
            hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)
        
        elif self.args.hem_extract_mode == 'hem-vi-offline':
            print('\ngenerate hem mode : {}\n'.format(self.args.hem_extract_mode))
            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_mutual_info(dropout_predictions, gt_list, img_path_list) # large
            hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

        elif self.args.hem_extract_mode == 'all-offline':
            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_softmax_diff(dropout_predictions, gt_list, img_path_list, 'diff_small')
            softmax_diff_small_hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_softmax_diff(dropout_predictions, gt_list, img_path_list, 'diff_large')
            softmax_diff_large_hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_voting(dropout_predictions, gt_list, img_path_list)
            voting_hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_mutual_info(dropout_predictions, gt_list, img_path_list, 'small')   
            vi_small_hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

            hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.extract_hem_idx_from_mutual_info(dropout_predictions, gt_list, img_path_list, 'large')   
            vi_large_hem_final_df = self.set_ratio(hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df)

            return softmax_diff_small_hem_final_df, softmax_diff_large_hem_final_df, voting_hem_final_df, vi_small_hem_final_df, vi_large_hem_final_df

        return hem_final_df


    # extract hem idx method
    def extract_hem_idx_from_voting(self, dropout_predictions, gt_list, img_path_list):
        hem_idx = []
        
        # 1. extract hem index
        predict_table = np.argmax(dropout_predictions, axis=2) # (forward_passes, n_samples)
        predict_ratio = np.mean(predict_table, axis=0) # (n_samples)

        predict_np = np.around(predict_ratio) # threshold == 0.5, if predict_ratio >= 0.5, predict_class == OOB(1)
        predict_np = np.int8(predict_np) # casting float to int
        predict_list = predict_np.tolist() # to list

        answer = predict_np == np.array(gt_list) # compare with gt list

        hem_idx = np.where(answer == False) # hard example
        hem_idx = hem_idx[0].tolist() # remove return turple

        # 2. split hem/vanila 
        total_df_dict = {
            'Img_path': img_path_list,
            'predict': predict_list,
            'GT': gt_list,
            'voting': predict_ratio.tolist(),
            'hem': [self.NON_HEM] * len(img_path_list) # init hem
        }

        total_df = pd.DataFrame(total_df_dict)
        total_df.loc[hem_idx, ['hem']] = self.HEM # hem index

        hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.split_to_hem_vanila_df(total_df)

        return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df
    

    def get_answer(self, dropout_predictions, gt_list):
        predict_table = np.argmax(dropout_predictions, axis=2) # (forward_passes, n_samples)
        predict_ratio = np.mean(predict_table, axis=0) # (n_samples)

        predict_np = np.around(predict_ratio) # threshold == 0.5, if predict_ratio >= 0.5, predict_class == OOB(1)
        predict_np = np.int8(predict_np) # casting float to int
        predict_list = predict_np.tolist() # to list

        answer = predict_np == np.array(gt_list) # compare with gt list

        return answer, predict_np

    def extract_hem_idx_from_softmax_diff(self, dropout_predictions, gt_list, img_path_list, diff='diff_small', ver='ver_1'):
        # dropout predictions - shape (forward_passes, n_samples, n_classes)
        
        hem_idx = []
        
        cols = ['Img_path', 'GT']
        CORRECT, INCORRECT = (True, False)
        IB_CLASS, OOB_CLASS = (0,1)

        if ver == 'ver_0':
            predict_mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes) //(550, 2)
            predict_diff = np.diff(predict_mean, axis=1) # shape (n_samples, 1) // (550, 1)
            predict_abs = np.abs(predict_diff) # shape (n_samples, 1) // (550, 1)

            '''
            print('\n\npredict_mean {}\n'.format(predict_mean.shape), predict_mean)
            print('\n\npredict_diff {}\n'.format(predict_diff.shape), predict_diff)
            print('\n\npredict_abs {}\n'.format(predict_abs.shape), predict_abs)
            '''

            predict_table = np.squeeze(predict_abs) # shape (n_samples, ) // (550, )

            # 맞았는지, 틀렸는지 (answer) - 평균 취해서 정답 확인. 
            predict_np = np.argmax(predict_mean, axis=1) # 예측값
            answer = predict_np == np.array(gt_list) # 예측값 & 정답 비교. 
            answer_list = answer.tolist()

  
        elif ver == 'ver_1':
            predict_diff = np.diff(dropout_predictions, axis=2) # shape (forward_passes, n_samples, 1) // (3, 550, 1) 
            predict_abs = np.abs(predict_diff) # shape (forward_passes, n_samples, 1) // (3, 550, 1) 
            predict_mean = np.mean(predict_abs, axis=0) # shape (n_samples, 1) // (550, 1) 

            predict_table = np.squeeze(predict_mean) # shape (n_samples, ) // (550, )

            # 맞았는지, 틀렸는지 (answer) - voting 방식.
            answer, _ = self.get_answer(dropout_predictions, gt_list)
            answer_list = answer.tolist()


        vanila_df = pd.DataFrame([x for x in zip(img_path_list, gt_list)],
                columns=cols)

        top_ratio = self.args.top_ratio # 30/100
        top_k = int(len(vanila_df) * top_ratio)

        # softmax hem 추출 시, lower_idx, upper_idx 따로 실험 :) 
        if diff == 'diff_small': # lower
            hard_df_lower_idx = predict_table.argsort()[:top_k].tolist() # 올림차순
            hard_idx = hard_df_lower_idx

            vanila_idx = []

            for i in range(len(vanila_df)):
                if (i not in hard_idx):
                    vanila_idx.append(i)

            hard_df = vanila_df.loc[hard_idx, :]
            vanila_df = vanila_df.loc[vanila_idx, :]

            hard_neg_df = hard_df[hard_df['GT']==IB_CLASS]
            hard_pos_df = hard_df[hard_df['GT']==OOB_CLASS]

            vanila_neg_df = vanila_df[vanila_df['GT']==IB_CLASS]
            vanila_pos_df = vanila_df[vanila_df['GT']==OOB_CLASS]

            return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df

        elif diff == 'diff_large': # higher
            hard_df_upper_idx = (-predict_table).argsort()[:top_k].tolist() # 내림차순

            hard_idx = []
            for i, answer in enumerate(answer_list): # answer_list
                if (i in hard_df_upper_idx) and answer==INCORRECT:
                    hard_idx.append(i)

            vanila_idx = []
            for i in range(len(vanila_df)):
                if (i not in hard_idx):
                    vanila_idx.append(i)

            hard_df = vanila_df.loc[hard_idx, :]
            vanila_df = vanila_df.loc[vanila_idx, :]

            hard_neg_df = hard_df[hard_df['GT']==IB_CLASS]
            hard_pos_df = hard_df[hard_df['GT']==OOB_CLASS]

            vanila_neg_df = vanila_df[vanila_df['GT']==IB_CLASS]
            vanila_pos_df = vanila_df[vanila_df['GT']==OOB_CLASS]

            return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df

    def extract_hem_idx_from_mutual_info(self, dropout_predictions, gt_list, img_path_list, location='large'):

        hem_idx = []

        # 1. extract hem index

        ## Calculate maen and variance
        mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes) 
        variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)
        
        epsilon = sys.float_info.min
        ## calc entropy across multiple MCD forward passes
        entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)

        ## calc mutual information
        # (https://www.edwith.org/medical-20200327/lecture/63144/)
        mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions + epsilon),
                                                                                    axis=-1), axis=0) # shape (n_samples,) // (550, 1)
        # if mutual info is high = relavence // low = non-relavence
        # so in (hard example data selection?), if i'th data is high(relavence), non-indepandence, i'th data has similar so it's hard

        # sort mi index & extract top/btm sample index 
        top_ratio = self.args.top_ratio
        top_k = int(len(mutual_info) * top_ratio)

        btm_ratio = self.args.top_ratio
        btm_k = int(len(mutual_info) * btm_ratio)

        sorted_mi_index = (-mutual_info).argsort() # desecnding index
        top_mi_index = sorted_mi_index[:top_k] # highest 
        btm_mi_index = sorted_mi_index[len(mutual_info) - btm_k:] # lowest

        # # extract wrong anwer from mean softmax // you can also change like voting methods for extracting predict class
        # predict_np = np.argmax(mean, axis=1)
        # predict_list = predict_np.tolist() # to list

        # answer = predict_np == np.array(gt_list) # compare with gt list
        answer, predict_np = self.get_answer(dropout_predictions, gt_list)
        predict_list = predict_np.tolist()

        wrong_idx = np.where(answer == False) # wrong example
        wrong_idx = wrong_idx[0].tolist() # remove return turple

        # append hem idx - high mi & wrong answer
        if location == 'large':
            hem_idx += np.intersect1d(wrong_idx, top_mi_index).tolist()
        
        # append hem idx - low mi
        elif location == 'small':
            hem_idx += btm_mi_index.tolist()
    
        
        # 2. split hem/vanila 
        total_df_dict = {
            'Img_path': img_path_list,
            'predict': predict_list,
            'GT': gt_list,
            'mi': mutual_info.tolist(),
            'hem': [self.NON_HEM] * len(img_path_list) # init hem
        }

        total_df = pd.DataFrame(total_df_dict)
        total_df.loc[hem_idx, ['hem']] = self.HEM # hem index

        hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df = self.split_to_hem_vanila_df(total_df)

        return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df


    def split_to_hem_vanila_df(self, total_df): # total_df should have ['hem', 'GT'] columne
        hem_df = total_df[total_df['hem'] == self.HEM]
        vanila_df = total_df[total_df['hem'] == self.NON_HEM]

        hard_neg_df = hem_df[hem_df['GT'] == self.IB_CLASS]
        hard_pos_df = hem_df[hem_df['GT'] == self.OOB_CLASS]
        
        vanila_neg_df = vanila_df[vanila_df['GT'] == self.IB_CLASS]
        vanila_pos_df = vanila_df[vanila_df['GT'] == self.OOB_CLASS]

        return hard_neg_df, hard_pos_df, vanila_neg_df, vanila_pos_df        
    
    def hem_cos_hard_sim(self, model, x, y, loss_fn):
        emb, y_hat = model(x)
        # emb = B x ch
        # proxies = ch x classes
        
        if self.args.emb_type == 1:
            sim_dist = emb @ model.proxies
            sim_preds = torch.argmax(sim_dist, -1)
        elif self.args.emb_type == 2:
            sim_dist = torch.zeros((emb.size(0), model.proxies.size(1))).to(emb.device)
            
            for d in range(sim_dist.size(1)):
                sim_dist[:, d] = 1 - torch.nn.functional.cosine_similarity(emb, model.proxies[:, d].unsqueeze(0))
            
            sim_preds = torch.argmin(sim_dist, -1)
        
        correct_answer = sim_preds == y
        wrong_answer = sim_preds != y
        
        if sum(correct_answer) > 0:
            pos_y_hat = y_hat[correct_answer]
            pos_y = y[correct_answer]
            
            pos_loss = loss_fn(pos_y_hat, pos_y)
            
            if self.args.use_proxy_all:
                proxy_loss = loss_fn(sim_dist, y)
            else:
                proxy_loss = loss_fn(sim_dist[correct_answer], pos_y)
        else:
            pos_loss = 0
            proxy_loss = 0
        
        if sum(wrong_answer) > 0:
            neg_y_hat = y_hat[wrong_answer]
            neg_y = y[wrong_answer]
            
            wrong_sim_dist = sim_dist[wrong_answer, neg_y]
            wrong_ids = torch.argsort(wrong_sim_dist)
            
            
            w = torch.Tensor(np.array(list(range(len(wrong_ids), 0, -1))) / len(wrong_ids)).cuda()
            
            neg_y_hat = neg_y_hat[wrong_ids]
            neg_y = neg_y[wrong_ids]
            
            neg_loss = 0
            neg_sim_dist = sim_dist[wrong_ids]
            
            if self.args.use_half_neg:
                if len(w) > 1:
                    st = len(w) // 2
                for wi in range(st, len(w)):
                    neg_loss += torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1]) * w[wi:wi+1]
                    
                    if self.args.use_neg_proxy:
                        n_proxy = torch.nn.functional.cross_entropy(neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                        proxy_loss = proxy_loss + n_proxy * w[wi:wi+1]
            else:
                for wi in range(len(w)):
                    if self.args.use_loss_weight:
                        w = torch.exp(self.alpha - wrong_sim_dist[wi])
                    
                        neg_loss += torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1]) * w
                        
                        if self.args.use_neg_proxy:
                            n_proxy = torch.nn.functional.cross_entropy(neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                            proxy_loss = proxy_loss + n_proxy * w
                    else:
                        neg_loss += torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1]) * w[wi:wi+1]
                        
                        if self.args.use_neg_proxy:
                            n_proxy = torch.nn.functional.cross_entropy(neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                            proxy_loss = proxy_loss + n_proxy * w[wi:wi+1]
              
        else:
            neg_loss = 0
            
        return (pos_loss + neg_loss) / 2. + proxy_loss
    
    def hem_cos_hard_sim2(self, model, x, y, loss_fn):
        emb, y_hat = model(x)
        sim_dist = emb @ model.proxies
        sim_preds = torch.argmax(sim_dist, -1)
        
        correct_answer = sim_preds == y
        wrong_answer = sim_preds != y
        
        pos_y_hat = y_hat[correct_answer]
        pos_y = y[correct_answer]
        
        neg_y_hat = y_hat[wrong_answer]
        neg_y = y[wrong_answer]
        
        proxy_loss = loss_fn(sim_dist[correct_answer], pos_y)
        
        correct_sim_dist = sim_dist[correct_answer, pos_y]
        correct_ids = torch.argsort(correct_sim_dist)
        
        wrong_sim_dist = sim_dist[wrong_answer, neg_y]
        wrong_ids = torch.argsort(wrong_sim_dist)
        
        ##################################################
        w = torch.Tensor(np.array(list(range(1, len(correct_ids)+1))) / len(correct_ids)).cuda()
        pos_y_hat = pos_y_hat[correct_ids]
        pos_y = pos_y[correct_ids]
        
        pos_loss = 0
        for wi in range(len(w)):
            pos_loss += torch.nn.functional.cross_entropy(pos_y_hat[wi:wi+1, ], pos_y[wi:wi+1]) * w[wi:wi+1]
        
        w = torch.Tensor(np.array(list(range(len(wrong_ids), 0, -1))) / len(wrong_ids)).cuda()
        neg_y_hat = neg_y_hat[wrong_ids]
        neg_y = neg_y[wrong_ids]
        
        neg_loss = 0
        for wi in range(len(w)):
            neg_loss += torch.nn.functional.cross_entropy(neg_y_hat[wi:wi+1, ], neg_y[wi:wi+1]) * w[wi:wi+1]
        
            
        return (pos_loss + neg_loss) / 2. + proxy_loss
    
    def hem_cos_hard_sim_only(self, model, x, y, loss_fn):
        emb = model(x)
        # emb = B x ch
        # proxies = ch x classes
        
        sim_dist = torch.zeros((emb.size(0), model.proxies.size(1))).to(emb.device)
        
        for d in range(sim_dist.size(1)):
            sim_dist[:, d] = 1 - torch.nn.functional.cosine_similarity(emb, model.proxies[:, d].unsqueeze(0))
        
        sim_preds = torch.argmin(sim_dist, -1)
        
        correct_answer = sim_preds == y
        wrong_answer = sim_preds != y
        
        proxy_loss = torch.zeros(1, requires_grad=True).cuda()
        
        if sum(correct_answer) > 0:
            pos_y = y[correct_answer]
            
            if self.args.use_proxy_all:
                proxy_loss += loss_fn(sim_dist, y)
            else:
                proxy_loss += loss_fn(sim_dist[correct_answer], pos_y)
        
        if sum(wrong_answer) > 0:
            neg_y = y[wrong_answer]
            
            wrong_sim_dist = sim_dist[wrong_answer, neg_y]
            wrong_ids = torch.argsort(wrong_sim_dist)
            
            w = torch.Tensor(np.array(list(range(len(wrong_ids), 0, -1))) / len(wrong_ids)).cuda()
            
            neg_y = neg_y[wrong_ids]
            
            neg_loss = 0
            neg_sim_dist = sim_dist[wrong_ids]
            
            if self.args.use_neg_proxy:
                if self.args.use_half_neg:
                    if len(w) > 1:
                        st = len(w) // 2
                    for wi in range(st, len(w)):
                        n_proxy = torch.nn.functional.cross_entropy(neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                        
                        if self.args.use_loss_weight:
                            w = torch.exp(self.alpha - neg_sim_dist[wi])
                            
                            proxy_loss = proxy_loss + n_proxy * w
                        else:
                            proxy_loss = proxy_loss + n_proxy * w[wi:wi+1]
                else:
                    for wi in range(len(w)):
                        n_proxy = torch.nn.functional.cross_entropy(neg_sim_dist[wi:wi+1, ], neg_y[wi:wi+1])
                        
                        if self.args.use_loss_weight:
                            w = torch.exp(self.alpha - neg_sim_dist[wi])
                        
                            proxy_loss = proxy_loss + n_proxy * w
                        else:
                            proxy_loss = proxy_loss + n_proxy * w[wi:wi+1]
            
        return proxy_loss