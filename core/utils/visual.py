import pandas as pd
import numpy as np
import datetime
from itertools import groupby
import math

from core.utils.metric import MetricHelper

class VisualHelper:
    def __init__(self):
        self.RS_CLASS, self.NRS_CLASS = (0,1) 
        self.NEG_HARD_CLASS, self.POS_HARD_CLASS, self.NEG_VANILA_CLASS, self.POS_VANILA_CLASS, = (2,3,4,5) # only use in calc section sampling

    def fit_to_min_length(self, gt_list, predict_list):
        gt_len, predict_len = len(gt_list), len(predict_list)
        min_len = min(gt_len, predict_len)

        gt_list, predict_list = gt_list[:min_len], predict_list[:min_len]

        return gt_list, predict_list

    def decode_list(self, run_length): # run_length -> [0,1,1,1,1,0 ...]
        decode_list = []

        for length, group in run_length : 
            decode_list += [group] * length

        return decode_list

    def encode_list(self, s_list): # run-length encoding from list
        return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]

    def idx_to_time(self, idx, fps) :
        time_s = idx // fps
        frame = int(idx % fps)

        converted_time = str(datetime.timedelta(seconds=time_s))
        converted_time = converted_time + ':' + str(frame)

        return converted_time

    def calc_section_sampling(self, sampling_list, window_size, section_num):
    
        calc_section_sampling = {
            'start_idx':[],
            'end_idx':[],
            'section_neg_hard_num':[],
            'section_pos_hard_num':[],
            'section_neg_vanila_num':[],
            'section_pos_vanila_num':[],
        }

        sampling_np = np.array(sampling_list)
        total_len = sampling_np.size
        
        slide_window_start_end_idx= [[start_idx * window_size, (start_idx + section_num) * window_size] for start_idx in range(math.ceil(total_len/window_size))] # overlapping section

        # calc metric per section
        for start, end in slide_window_start_end_idx : # slicing            
            section_np = sampling_np[start:end]

            unique, counts = np.unique(section_np, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            
            end = start + section_np.size - 1 # truly end idx

            calc_section_sampling['start_idx'].append(start)
            calc_section_sampling['end_idx'].append(end)
            calc_section_sampling['section_neg_hard_num'].append(counts_dict.get(self.NEG_HARD_CLASS, 0))
            calc_section_sampling['section_pos_hard_num'].append(counts_dict.get(self.POS_HARD_CLASS, 0))
            calc_section_sampling['section_neg_vanila_num'].append(counts_dict.get(self.NEG_VANILA_CLASS, 0))
            calc_section_sampling['section_pos_vanila_num'].append(counts_dict.get(self.POS_VANILA_CLASS, 0))
        return calc_section_sampling
        

    def calc_section_metrics(self, gt_list, predict_list, window_size, section_num):
        metrics_per_section = {
            'start_idx':[],
            'end_idx':[],
            'section_CR':[],
            'section_OR':[],
        }

        data = {
            'GT': gt_list,
            'PREDICT': predict_list,
        }

        total_info_df = pd.DataFrame(data)
        total_len = len(total_info_df)
        slide_window_start_end_idx= [[start_idx * window_size, (start_idx + section_num) * window_size] for start_idx in range(math.ceil(total_len/window_size))] # overlapping section

        # calc metric per section
        for start, end in slide_window_start_end_idx : # slicing            
            section_df = total_info_df.iloc[start:end, ]

            section_gt_list, section_predict_list = section_df['GT'].tolist(), section_df['PREDICT'].tolist()
            section_metrics = self.calc_metrics_with_index(section_gt_list, section_predict_list)
            
            end = start + len(section_df) - 1 # truly end idx

            metrics_per_section['start_idx'].append(start)
            metrics_per_section['end_idx'].append(end)
            metrics_per_section['section_CR'].append(section_metrics['CR'])
            metrics_per_section['section_OR'].append(section_metrics['OR'])

        return metrics_per_section
        
    def calc_metrics_with_index(self, gt_list, predict_list):
        metric_helper = MetricHelper()
        metric_helper.write_preds(np.array(predict_list), np.array(gt_list))
        metrics = metric_helper.calc_metric()
        
        # with metric index
        gt_np = np.array(gt_list)
        predict_np = np.array(predict_list)

        metrics['TP_idx'] = np.where((predict_np == self.NRS_CLASS) & (gt_np == self.NRS_CLASS))[0].tolist()
        metrics['TN_idx'] = np.where((predict_np == self.RS_CLASS) & (gt_np == self.RS_CLASS))[0].tolist()
        metrics['FP_idx'] = np.where((predict_np == self.NRS_CLASS) & (gt_np == self.RS_CLASS))[0].tolist()
        metrics['FN_idx'] = np.where((predict_np == self.RS_CLASS) & (gt_np == self.NRS_CLASS))[0].tolist()

        return metrics