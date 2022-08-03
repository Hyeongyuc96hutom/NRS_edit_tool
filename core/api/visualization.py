import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
import math

from core.utils.metric import MetricHelper # metric helper (for calc CR, OR, mCR, mOR)
from core.utils.visual import VisualHelper

class VisualTool:
    def __init__(self, gt_list, patient_name, save_path): # patinet_name == 'R_100', 'R_10', ...
        self.RS_CLASS, self.NRS_CLASS = (0,1)
        self.FN_CLASS, self.FP_CLASS = (2,3) # only use in visual predict
        self.NEG_HARD_CLASS, self.POS_HARD_CLASS, self.NEG_VANILA_CLASS, self.POS_VANILA_CLASS, = (2,3,4,5) # only use in visual hem

        self.visual_helper = VisualHelper()
        self.EXCEPTION_NUM = -100

        self.gt_list = gt_list
        
        self.patient_name = patient_name
        self.save_path = save_path # "~.png"

    def _get_plt(self, visual_type):
        assert visual_type in ['predict', 'predict_multi', 'sampling', 'sampling_multi'], 'NOT SUPPORT VISUAL MODE'

        if visual_type == 'predict':
            fig, ax = plt.subplots(3,1,figsize=(18,15)) # 1x1 figure matrix 생성, figsize=(가로, 세로) 크기지정

            plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0, 
                    hspace=0.35)

        if visual_type == 'predict_multi':
            fig, ax = plt.subplots(1,1,figsize=(30,5)) # 1x1 figure matrix 생성, figsize=(가로, 세로) 크기지정

            '''
            plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0, 
                    hspace=0.35)
            '''

        elif visual_type == 'sampling':
            fig, ax = plt.subplots(2,1,figsize=(18,14)) # 2x1 figure matrix 생성, 가로(18인치)x세로(13인치) 크기지정

        elif visual_type == 'sampling_multi':
            fig, ax = plt.subplots(1,1,figsize=(15,2)) # 1x1 figure matrix 생성, figsize=(가로, 세로) 크기지정

            plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0, 
                    hspace=0.35)

        return fig, ax
        
    def set_patient_name(self, paitent_name):
        self.patient_name = patient_name

    def set_gt_list(self, gt_list):
        self.gt_list = gt_list

    def set_save_path(self, save_path):
        self.save_path = save_path

    # for text on bar
    def present_text(self, ax, bar, text, color='black'):
        for rect in bar:
            posx = rect.get_x()
            posy = rect.get_y() - rect.get_height()*0.1
            ax.text(posx, posy, text, color=color, rotation=0, ha='left', va='bottom')

    # for section metric
    def draw_plot(self, ax, title, x_value, y_value, y_min, y_max, color='blue'):
        ax.plot(x_value, y_value, marker='o', markersize=4, alpha=1.0, color=color)

        # set title
        ax.set_title(title)
        
        # set x ticks
        ax.set_xticks(x_value)
        xtick_labels = ['{}'.format(frame) if i_th % 2 == 0 else '\n{}'.format(frame) for i_th, frame in enumerate(x_value)]
        ax.set_xticklabels(xtick_labels) # xtick change
        ax.xaxis.set_tick_params(labelsize=6)
        ax.set_xlabel('Start Frame', fontsize=12)
        
        # set y ticks
        ax.set_ylim(ymin=y_min, ymax=y_max)

        # 보조선
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

    def visual_predict(self, predict_list, model_name, inference_interval, window_size = 300, section_num = 2):
        # TO-DO: if call gt_list from self.patient_name, better
        # TO-DO: visualization oob ratio, metrics on ax
        '''
        predict_list: [1,0,0...]
        model_name: str - 'mobilenet .. ' => title name
        inferene_interval: int
        inference_interval: number of elements in each section => title name
        section number: number of elements in each section => title name
        '''
        
        # calc section metric per model

        gt_list, predict_list = self.visual_helper.fit_to_min_length(self.gt_list, predict_list)
        metrics = self.visual_helper.calc_metrics_with_index(gt_list, predict_list)
        metrics_per_section = self.visual_helper.calc_section_metrics(gt_list, predict_list, window_size, section_num)
        
        fig, ax = self._get_plt('predict')
        
        ### for plt variable, it should be pair synk
        label_names = ['RS', 'NRS', 'FN', 'FP']
        colors = ['cadetblue', 'orange', 'blue', 'red']
        height = 0.5 # bar chart thic

        ### prepare data ###
        '''
        visual_data = {'GT': [1, 1, 1, 0, 0, 1, 1, 1], # 0번 ~ 2번 frame , 5번 ~ 7번 frame
	        'Model A': [0, 0, 1, 1, 0, 0, 1, 1], # 2번 ~ 3번 frame
			'Model B': [1, 0, 1, 1, 1, 0, 1, 0] # 0번 ~ 0번 frame, 2번 ~ 4번 frame, 6번 ~ 6번 frame
        }

	    runlength_df = df([[3,0,0,1],
				[2,0,0,0],
				[3,0,0,1],
				[0,2,0,0],
				[0,2,0,1],
				[0,2,0,0],
				[0,2,0,1],
				[0,0,1,1],
				[0,0,1,0],
				[0,0,3,1],
				[0,0,1,0],
				[0,0,1,1],
				[0,0,1,0]], columns= yticks + ['class'])
	    '''
        frame_label = list(range(0, len(gt_list) * inference_interval, inference_interval))
        time_label = [self.visual_helper.idx_to_time(idx, fps=30) for idx in frame_label]
        yticks = ['GT', 'PREDICT'] # y축 names, 순서중요

        visual_data = {
            'GT':gt_list,
            'PREDICT':predict_list,
        }

        # for visualize FP, FN // change class
        predict_df = pd.DataFrame(predict_list)
        predict_df.iloc[metrics['FN_idx'],] = self.FN_CLASS
        predict_df.iloc[metrics['FP_idx'],] = self.FP_CLASS
        visual_data['PREDICT'] = list(np.array(predict_df[0].tolist()))

        encode_data = {}
        for y_name in yticks : # run_length coding
            encode_data[y_name] = pd.DataFrame(data=self.visual_helper.encode_list(visual_data[y_name]), columns=[y_name, 'class']) # [length, value]

        # arrange data
        runlength_df = pd.DataFrame(range(0,0)) # empty df
        for y_name in yticks :
            runlength_df = runlength_df.append(encode_data[y_name])

        # Nan -> 0, convert to int
        runlength_df = runlength_df.fillna(0).astype(int)

        # split data, class // both should be same length
        runlength_class = runlength_df['class'] # class info
        runlength_model = runlength_df[yticks] # run length info of model prediction

        ### draw ###
        ##### initalize label for legned, this code should be write before writing barchart #####
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.RS_CLASS], height=height, color=colors[self.RS_CLASS]) # dummy data
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.NRS_CLASS], height=height, color=colors[self.NRS_CLASS]) # dummy data
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.FN_CLASS], height=height, color=colors[self.FN_CLASS]) # dummy data
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.FP_CLASS], height=height, color=colors[self.FP_CLASS]) # dummy data
	    ##### #### #### #### ##### #### #### #### 

        # data processing for barchart
        data = np.array(runlength_model.to_numpy()) # width
        data_cum = data.cumsum(axis=0) # for calc start index

        # draw bar
        for i, frame_class in enumerate(runlength_class) :
            # print(data[i,:])
            # print(frame_class)

            widths = data[i,:]
            starts= data_cum[i,:] - widths
            
            bar = ax[0].barh(range(len(yticks)), widths, left=starts, height=height, color=colors[frame_class]) # don't input label

        # write text on PREDICT bar
        gt_oob_ratio = metrics['gt_OOB'] / (metrics['gt_IB'] + metrics['gt_OOB'])
        predict_oob_ratio = metrics['predict_OOB'] / (metrics['predict_IB'] + metrics['predict_OOB'])
        text_bar = ax[0].barh(1, 0, height=height) # dummy data        
        self.present_text(ax[0], text_bar, 'CR : {:.3f} | OR : {:.3f} | JACCARD: {:.3f} | OOB_RATIO(GT) : {:.2f} | OOB_RATIO(PD) : {:.2f}'.format(metrics['CR'], metrics['OR'], metrics['Jaccard'], gt_oob_ratio, predict_oob_ratio))

        ### write on figure 
        # set title
        title_name = 'Predict of {}'.format(self.patient_name)
        sub_title_name = 'model: {} | inferene interval: {} | windows size: {} | section num: {}'.format(model_name, inference_interval, window_size, section_num)
        fig.suptitle(title_name, fontsize=16)
        ax[0].set_title(sub_title_name)

        # set xticks pre section size
        ax[0].set_xticks(range(0, len(frame_label), window_size))
        
        xtick_labels = ['{}\n{}'.format(time, frame) if i_th % 2 == 0 else '\n\n{}\n{}'.format(time, frame) for i_th, (time, frame) in enumerate(zip(frame_label[::window_size], time_label[::window_size]))]
        ax[0].set_xticklabels(xtick_labels) # xtick change
        ax[0].xaxis.set_tick_params(labelsize=6)
        ax[0].set_xlabel('Frame / Time (h:m:s:fps)', fontsize=12)

        # set yticks
        ax[0].set_yticks(range(len(yticks)))
        ax[0].set_yticklabels(yticks, fontsize=10)	
        ax[0].set_ylabel('Model', fontsize=12)

        # 8. legend
        box = ax[0].get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
        ax[0].set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax[0].legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

        # 9. 보조선(눈금선) 나타내기
        ax[0].set_axisbelow(True)
        ax[0].xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

        # 10. draw subplot ax (section metrics)
        section_start_idx = np.array(metrics_per_section['start_idx']) * inference_interval
        section_start_idx = section_start_idx.tolist()
        CR_value = metrics_per_section['section_CR']
        OR_value = metrics_per_section['section_OR']

        CR_value = [1.0 if val==self.EXCEPTION_NUM else val for val in CR_value] # -100 EXP 일 경우 1로 처리
        OR_value = [0.0 if val==self.EXCEPTION_NUM else val for val in OR_value] # -100 EXP 일 경우 0로 처리
        
        self.draw_plot(ax[1], 'CR of Predict', section_start_idx, CR_value, y_min=-1.0, y_max=1.0, color='blue')
        self.draw_plot(ax[2], 'OR of Predict', section_start_idx, OR_value, y_min=0.0, y_max=1.0, color='red')

        ### file write
	    # fig.tight_layout() # subbplot 간격 줄이기
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        plt.show()
        plt.savefig(self.save_path, format='png', dpi=500)

        plt.close(fig)
    

    def visual_predict_multi(self, predict_list_dict, model_name, inference_interval, window_size = 300, section_num = 2):
        # TO-DO: if call gt_list from self.patient_name, better
        # TO-DO: visualization oob ratio, metrics on ax
        '''
        predict_list: [1,0,0...]
        model_name: str - 'mobilenet .. ' => title name
        inferene_interval: int
        inference_interval: number of elements in each section => title name
        section number: number of elements in each section => title name
        '''
    
        
        fig, ax = self._get_plt('predict_multi')
        
        ### for plt variable, it should be pair synk
        label_names = ['RS', 'NRS', 'FN', 'FP']
        colors = ['orange', '#4da699', 'blue', 'red']
        # colors = ['navy', 'rosybrown', 'navy', 'rosybrown']
        alpha_ratio = [0.3, 0.3, 1,1]
        edge_colors = ['orange', '#4da699', 'blue', 'red']
        edge_linewidth = [0, 0, 1.0, 1.0]
        height = 0.9 # bar chart thic

        ### prepare data ###
        '''
        visual_data = {'GT': [1, 1, 1, 0, 0, 1, 1, 1], # 0번 ~ 2번 frame , 5번 ~ 7번 frame
	        'Model A': [0, 0, 1, 1, 0, 0, 1, 1], # 2번 ~ 3번 frame
			'Model B': [1, 0, 1, 1, 1, 0, 1, 0] # 0번 ~ 0번 frame, 2번 ~ 4번 frame, 6번 ~ 6번 frame
        }

	    runlength_df = df([[3,0,0,1],
				[2,0,0,0],
				[3,0,0,1],
				[0,2,0,0],
				[0,2,0,1],
				[0,2,0,0],
				[0,2,0,1],
				[0,0,1,1],
				[0,0,1,0],
				[0,0,3,1],
				[0,0,1,0],
				[0,0,1,1],
				[0,0,1,0]], columns= yticks + ['class'])
	    '''
        visual_data = {}

        ### 조정
        lower_bound_frame = 0 # 6138 - 100
        upper_bound_frame = -1

        print('GT len:', len(self.gt_list))
        for k,v in predict_list_dict.items():
            print(len(v))
            visual_data['GT'], visual_data[k] = self.visual_helper.fit_to_min_length(self.gt_list, v)

        for k,v in visual_data.items():
            visual_data[k] = v[lower_bound_frame:upper_bound_frame]

        gt_list = visual_data['GT']

        # print(len(gt_list)) # 6138 / 30908 // # 8361 - 600
        # exit(0)

        frame_label = list(range(0, len(gt_list) * inference_interval, inference_interval))
        time_label = [self.visual_helper.idx_to_time(idx, fps=30) for idx in frame_label]

        # hg fix (section visualizaiton)
        # base_idx = 9000
        # frame_label = [f + base_idx for f in frame_label]
        # time_label = [self.visual_helper.idx_to_time(idx, fps=30) for idx in frame_label]

        yticks = ['GT', 'PREDICT'] # y축 names, 순서중요
        yticks = ['GT'] + list(predict_list_dict.keys())

        # change class for visual FP FN
        for k, v in visual_data.items():
            metrics = self.visual_helper.calc_metrics_with_index(gt_list, v)
            # metrics_per_section = self.visual_helper.calc_section_metrics(gt_list, v, window_size, section_num)

            # for visualize FP, FN // change class
            predict_df = pd.DataFrame(v)
            predict_df.iloc[metrics['FN_idx'],] = self.FN_CLASS
            predict_df.iloc[metrics['FP_idx'],] = self.FP_CLASS

            visual_data[k] = list(np.array(predict_df[0].tolist()))

        # encoding data for visualization
        encode_data = {}
        for y_name in yticks : # run_length coding
            encode_data[y_name] = pd.DataFrame(data=self.visual_helper.encode_list(visual_data[y_name]), columns=[y_name, 'class']) # [length, value]

        # arrange data
        runlength_df = pd.DataFrame(range(0,0)) # empty df
        for y_name in yticks :
            runlength_df = runlength_df.append(encode_data[y_name])

        # Nan -> 0, convert to int
        runlength_df = runlength_df.fillna(0).astype(int)

        # split data, class // both should be same length
        runlength_class = runlength_df['class'] # class info
        runlength_model = runlength_df[yticks] # run length info of model prediction

        ### draw ###
        ##### initalize label for legned, this code should be write before writing barchart #####
        init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.RS_CLASS], height=height, alpha=alpha_ratio[self.RS_CLASS], linewidth=edge_linewidth[self.RS_CLASS], edgecolor=edge_colors[self.RS_CLASS], color=colors[self.RS_CLASS]) # dummy data
        init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.NRS_CLASS], height=height, alpha=alpha_ratio[self.NRS_CLASS], linewidth=edge_linewidth[self.NRS_CLASS], edgecolor=edge_colors[self.NRS_CLASS], color=colors[self.NRS_CLASS]) # dummy data
        init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.FN_CLASS], height=height, alpha=alpha_ratio[self.FN_CLASS], linewidth=edge_linewidth[self.FN_CLASS], edgecolor=edge_colors[self.FN_CLASS], color=colors[self.FN_CLASS]) # dummy data
        init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.FP_CLASS], height=height, alpha=alpha_ratio[self.FP_CLASS], linewidth=edge_linewidth[self.FP_CLASS], edgecolor=edge_colors[self.FP_CLASS], color=colors[self.FP_CLASS]) # dummy data
	    ##### #### #### #### ##### #### #### #### 

        # data processing for barchart
        data = np.array(runlength_model.to_numpy()) # width
        data_cum = data.cumsum(axis=0) # for calc start index

        # draw bar
        for i, frame_class in enumerate(runlength_class) :
            # print(data[i,:])
            # print(frame_class)

            widths = data[i,:]
            starts= data_cum[i,:] - widths
            
            bar = ax.barh(range(len(yticks)), widths, left=starts, height=height, alpha=alpha_ratio[frame_class], linewidth=edge_linewidth[frame_class], edgecolor=edge_colors[frame_class], color=colors[frame_class]) # don't input label

        # write text on PREDICT bar
        '''
        gt_oob_ratio = metrics['gt_OOB'] / (metrics['gt_IB'] + metrics['gt_OOB'])
        predict_oob_ratio = metrics['predict_OOB'] / (metrics['predict_IB'] + metrics['predict_OOB'])
        text_bar = ax.barh(1, 0, height=height) # dummy data        
        self.present_text(ax, text_bar, 'CR : {:.3f} | OR : {:.3f} | JACCARD: {:.3f} | OOB_RATIO(GT) : {:.2f} | OOB_RATIO(PD) : {:.2f}'.format(metrics['CR'], metrics['OR'], metrics['Jaccard'], gt_oob_ratio, predict_oob_ratio))
        '''

        ### write on figure 
        # set title
        title_name = 'Title will be removed [Predict of {}]'.format(self.patient_name)
        # sub_title_name = 'model: {} | inferene interval: {} | windows size: {} | section num: {}'.format(model_name, inference_interval, window_size, section_num)
        sub_title_name =  'Model {}'.format(model_name)
        # fig.suptitle(title_name, fontsize=20)
        # ax.set_title(sub_title_name, fontsize=16)

        # non visual
        '''
        ax.xaxis.set_visible(False) # non visual x label
        ax.spines['left'].set_visible(False) # ax non visual boundary
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        '''

        # set xticks pre section size
        ax.set_xticks(range(0, len(frame_label), window_size))
        
        # xtick_labels = ['{}\n{}'.format(time, frame) if i_th % 2 == 0 else '\n\n{}\n{}'.format(time, frame) for i_th, (time, frame) in enumerate(zip(frame_label[::window_size], time_label[::window_size]))]
        xtick_labels = ['{}\n{}'.format(time, frame) if True else '\n\n{}\n{}'.format(time, frame) for i_th, (time, frame) in enumerate(zip(frame_label[::window_size], time_label[::window_size]))]
        ax.set_xticklabels(xtick_labels) # xtick change
        ax.xaxis.set_tick_params(labelsize=9)
        ax.set_xlabel('Frame / Time (h:m:s:fps)', fontsize=20)


        # set yticks
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=20)
        for label in ax.get_yticklabels(): label.set_fontweight('bold')
        # ax.set_ylabel('Methods', fontsize=20)

        # 8. legend
        print('lening')
        box = ax.get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
        # ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax.legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

        # 9. 보조선(눈금선) 나타내기
        '''
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)
        '''

        # 10. draw subplot ax (section metrics)
        '''
        section_start_idx = np.array(metrics_per_section['start_idx']) * inference_interval
        section_start_idx = section_start_idx.tolist()
        CR_value = metrics_per_section['section_CR']
        OR_value = metrics_per_section['section_OR']

        CR_value = [1.0 if val==self.EXCEPTION_NUM else val for val in CR_value] # -100 EXP 일 경우 1로 처리
        OR_value = [0.0 if val==self.EXCEPTION_NUM else val for val in OR_value] # -100 EXP 일 경우 0로 처리
        
        self.draw_plot(ax[1], 'CR of Predict', section_start_idx, CR_value, y_min=-1.0, y_max=1.0, color='blue')
        self.draw_plot(ax[2], 'OR of Predict', section_start_idx, OR_value, y_min=0.0, y_max=1.0, color='red')
        '''

        ### file write
        fig.tight_layout() # subbplot 간격 줄이기
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        plt.show()
        plt.savefig(self.save_path, format='png', dpi=1000)

        plt.close(fig)
    
    def visual_sampling(self, neg_hard_idx, pos_hard_idx, neg_vanila_idx, pos_vanila_idx, model_name, window_size=9000, section_num=2):
        # visualization for hard example mining

        # set plt
        gt_list = self.gt_list
        fig, ax = self._get_plt('sampling')

        ### for plt variable, it should be pair synk
        label_names = ['RS', 'NRS', 'NEG_HARD', 'POS_HARD', 'NEG_VANILA', 'POS_VANILA']
        colors = ['cadetblue', 'orange', 'royalblue', 'navy', 'darksalmon', 'firebrick']
        alpha_ratio = [0.3,0.3,1,1,1,1]
        edge_colors = ['cadetblue', 'orange', 'royalblue', 'navy', 'darksalmon', 'firebrick']
        edge_linewidth = [0, 0, 0.1, 0.1, 0.1, 0.1]
        height = 0.5 # bar chart thic

        inference_interval = 1
        frame_label = list(range(0, len(gt_list) * inference_interval, inference_interval))
        time_label = [self.visual_helper.idx_to_time(idx, fps=30) for idx in frame_label]

        yticks = ['GT', 'SAMPLING'] # y축 names, 순서중요

        # processing data
        sampling_df = pd.DataFrame(gt_list)

        sampling_df.iloc[neg_hard_idx,] = self.NEG_HARD_CLASS
        sampling_df.iloc[pos_hard_idx,] = self.POS_HARD_CLASS
        sampling_df.iloc[neg_vanila_idx,] = self.NEG_VANILA_CLASS
        sampling_df.iloc[pos_vanila_idx,] = self.POS_VANILA_CLASS

        sampling_list = list(np.array(sampling_df[0].tolist())) # df to list

        visual_data = {
            'GT':gt_list,
            'SAMPLING':sampling_list,
        }

        encode_data = {}
        for y_name in yticks : # run_length coding
            encode_data[y_name] = pd.DataFrame(data=self.visual_helper.encode_list(visual_data[y_name]), columns=[y_name, 'class']) # [length, value]

        # arrange data
        runlength_df = pd.DataFrame(range(0,0)) # empty df
        for y_name in yticks :
            runlength_df = runlength_df.append(encode_data[y_name])

        # Nan -> 0, convert to int
        runlength_df = runlength_df.fillna(0).astype(int)

        # split data, class // both should be same length
        runlength_class = runlength_df['class'] # class info
        runlength_model = runlength_df[yticks] # run length info of model prediction

        ### draw ###
        ##### initalize label for legned, this code should be write before writing barchart #####
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.RS_CLASS], height=height, color=colors[self.RS_CLASS], alpha=alpha_ratio[self.RS_CLASS], linewidth=edge_linewidth[self.RS_CLASS], edgecolor=edge_colors[self.RS_CLASS]) # dummy data
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.NRS_CLASS], height=height, color=colors[self.NRS_CLASS], alpha=alpha_ratio[self.NRS_CLASS], linewidth=edge_linewidth[self.NRS_CLASS], edgecolor=edge_colors[self.NRS_CLASS]) # dummy data
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.NEG_HARD_CLASS], height=height, color=colors[self.NEG_HARD_CLASS], alpha=alpha_ratio[self.NEG_HARD_CLASS], linewidth=edge_linewidth[self.NEG_HARD_CLASS], edgecolor=edge_colors[self.NEG_HARD_CLASS]) # dummy data
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.POS_HARD_CLASS], height=height, color=colors[self.POS_HARD_CLASS], alpha=alpha_ratio[self.POS_HARD_CLASS], linewidth=edge_linewidth[self.POS_HARD_CLASS], edgecolor=edge_colors[self.POS_HARD_CLASS]) # dummy data
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.NEG_VANILA_CLASS], height=height, color=colors[self.NEG_VANILA_CLASS], alpha=alpha_ratio[self.NEG_VANILA_CLASS], linewidth=edge_linewidth[self.NEG_VANILA_CLASS], edgecolor=edge_colors[self.NEG_VANILA_CLASS]) # dummy data
        init_bar = ax[0].barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.POS_VANILA_CLASS], height=height, color=colors[self.POS_VANILA_CLASS], alpha=alpha_ratio[self.POS_VANILA_CLASS], linewidth=edge_linewidth[self.POS_VANILA_CLASS], edgecolor=edge_colors[self.POS_VANILA_CLASS]) # dummy data
	    ##### #### #### #### ##### #### #### #### 

        # data processing for barchart
        data = np.array(runlength_model.to_numpy()) # width
        data_cum = data.cumsum(axis=0) # for calc start index

        # draw bar
        for i, frame_class in enumerate(runlength_class) :
            # print(data[i,:])
            # print(frame_class)

            widths = data[i,:]
            starts= data_cum[i,:] - widths
            
            bar = ax[0].barh(range(len(yticks)), widths, left=starts, height=height, color=colors[frame_class], alpha=alpha_ratio[frame_class], linewidth=edge_linewidth[frame_class], edgecolor=edge_colors[frame_class]) # don't input label

        ### write on figure 
        # set title
        title_name = 'Sampling of {}'.format(self.patient_name)
        sub_title_name = 'model: {} | windows size: {} | section num: {}'.format(model_name, window_size, section_num)
        fig.suptitle(title_name, fontsize=16)
        ax[0].set_title(sub_title_name)

        # set xticks pre section size
        ax[0].set_xticks(range(0, len(frame_label), window_size))
        
        xtick_labels = ['{}\n{}'.format(time, frame) if i_th % 2 == 0 else '\n\n{}\n{}'.format(time, frame) for i_th, (time, frame) in enumerate(zip(frame_label[::window_size], time_label[::window_size]))]
        ax[0].set_xticklabels(xtick_labels) # xtick change
        ax[0].xaxis.set_tick_params(labelsize=6)
        ax[0].set_xlabel('Frame / Time (h:m:s:fps)', fontsize=12)

        # set yticks
        ax[0].set_yticks(range(len(yticks)))
        ax[0].set_yticklabels(yticks, fontsize=10)	
        ax[0].set_ylabel('Sampling', fontsize=12)

        # 8. legend
        box = ax[0].get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
        ax[0].set_position([box.x0, box.y0, box.width * 0.9, box.height])
        ax[0].legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

        # 9. 보조선(눈금선) 나타내기
        ax[0].set_axisbelow(True)
        ax[0].xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

        # 11. write sampling count
        unique, counts = np.unique(sampling_list, return_counts=True)
        counts_dict = dict(zip(unique, counts))

        neg_hard_sampling_count = counts_dict.get(self.NEG_HARD_CLASS, 0)
        pos_hard_sampling_count = counts_dict.get(self.POS_HARD_CLASS, 0)
        neg_vanila_sampling_count = counts_dict.get(self.NEG_VANILA_CLASS, 0)
        pos_vanila_sampling_count = counts_dict.get(self.POS_VANILA_CLASS, 0)

        rs_count = gt_list.count(self.RS_CLASS)
        nrs_count = gt_list.count(self.NRS_CLASS)

        total_sampling_count = neg_hard_sampling_count + pos_hard_sampling_count + neg_vanila_sampling_count + pos_vanila_sampling_count

        text_bar = ax[0].barh(1, 0, height=height) # dummy data
        text = '[GT_RS - {} GT_NRS - {}] || Total Sampling: {} | Neg Hard: {} | Pos Hard: {} | Neg Vanila: {} | Pos Vanila: {}'.format(rs_count, nrs_count, total_sampling_count, neg_hard_sampling_count, pos_hard_sampling_count, neg_vanila_sampling_count, pos_vanila_sampling_count)
        self.present_text(ax[0], text_bar, text)

        # 10. draw subplot ax (section sampling count)
        sampling_per_section = self.visual_helper.calc_section_sampling(sampling_list, window_size, section_num)
        section_start_idx = np.array(sampling_per_section['start_idx']) * inference_interval
        section_start_idx = section_start_idx.tolist()
        
        hard_example_num = np.array(sampling_per_section['section_neg_hard_num']) + np.array(sampling_per_section['section_pos_hard_num'])
        vanila_example_num = np.array(sampling_per_section['section_neg_vanila_num']) + np.array(sampling_per_section['section_pos_vanila_num'])

        # Stacked Bar Chart
        index = np.arange(len(section_start_idx)) # x axis

        p1 = ax[1].bar(index, hard_example_num, color='b', alpha=0.5)
        p2 = ax[1].bar(index, vanila_example_num, color='r', alpha=0.5, bottom=hard_example_num) # stacked bar chart

        ax[1].set_title('Stacked Bar Chart of Count of Sampling')
        ax[1].set_ylabel('Sampling Count', fontsize=12)
        ax[1].set_xlabel('Start Frame', fontsize=12)
        
        # xtick change
        ax[1].set_xticks(index)
        ax[1].set_xticklabels(xtick_labels) 
        ax[1].xaxis.set_tick_params(labelsize=6)

        ax[1].set_axisbelow(True)
        ax[1].xaxis.grid(True, color='gray', linestyle='dashed', linewidth=0.5)

        ax[1].legend((p1[0], p2[0]), ('hard', 'vanila'), loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

        ### file write
	    # fig.tight_layout() # subbplot 간격 줄이기
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        plt.show()
        plt.savefig(self.save_path, format='png', dpi=500)

        plt.close(fig)


    def visual_sampling_multi(self, split_assets_multi, model_name, window_size=9000, section_num=2):
        # visualization for hard example mining

        # set plt
        gt_list = self.gt_list
        fig, ax = self._get_plt('sampling_multi')

        '''
        ### for plt variable, it should be pair synk
        label_names = ['RS', 'NRS', 'NEG_HARD', 'POS_HARD', 'NEG_VANILA', 'POS_VANILA']
        colors = ['cadetblue', 'orange', 'royalblue', 'navy', 'darksalmon', 'firebrick']
        alpha_ratio = [0.3,0.3,1,1,1,1]
        edge_colors = ['cadetblue', 'orange', 'royalblue', 'navy', 'darksalmon', 'firebrick']
        edge_linewidth = [0, 0, 0.1, 0.1, 0.1, 0.1]
        height = 0.5 # bar chart thic
        '''

        ### for plt variable, it should be pair synk
        label_names = ['RS', 'NRS', 'NEG_HARD', 'POS_HARD', 'NEG_VANILA', 'POS_VANILA']
        colors = ['orange', '#4da699', 'firebrick', 'navy', 'firebrick', 'navy']
        alpha_ratio = [0.3,1,1,1,1,1]
        edge_colors = ['orange', '#4da699', 'firebrick', 'navy', 'firebrick', 'navy']
        edge_linewidth = [0.3, 0, 0.3, 0.3, 0.3, 0.3]
        height = 0.9 # bar chart thic

        inference_interval = 1
        frame_label = list(range(0, len(gt_list) * inference_interval, inference_interval))
        time_label = [self.visual_helper.idx_to_time(idx, fps=30) for idx in frame_label]

        yticks = ['GT', 'SAMPLING'] # y축 names, 순서중요
        yticks = ['GT'] + list(split_assets_multi.keys())

        visual_data = {
            'GT':gt_list
        }

        for key, split_assets in split_assets_multi.items():
            neg_hard_idx, pos_hard_idx, neg_vanila_idx, pos_vanila_idx = split_assets['neg_hard_idx'], split_assets['pos_hard_idx'], split_assets['neg_vanila_idx'], split_assets['pos_vanila_idx']


            # processing data
            sampling_df = pd.DataFrame(gt_list)

            sampling_df.iloc[neg_hard_idx,] = self.NEG_HARD_CLASS
            sampling_df.iloc[pos_hard_idx,] = self.POS_HARD_CLASS
            sampling_df.iloc[neg_vanila_idx,] = self.NEG_VANILA_CLASS
            sampling_df.iloc[pos_vanila_idx,] = self.POS_VANILA_CLASS

            sampling_list = list(np.array(sampling_df[0].tolist())) # df to list

            visual_data[key] = sampling_list


        encode_data = {}
        for y_name in yticks : # run_length coding
            encode_data[y_name] = pd.DataFrame(data=self.visual_helper.encode_list(visual_data[y_name]), columns=[y_name, 'class']) # [length, value]

        # arrange data
        runlength_df = pd.DataFrame(range(0,0)) # empty df
        for y_name in yticks :
            runlength_df = runlength_df.append(encode_data[y_name])

        # Nan -> 0, convert to int
        runlength_df = runlength_df.fillna(0).astype(int)

        # split data, class // both should be same length
        runlength_class = runlength_df['class'] # class info
        runlength_model = runlength_df[yticks] # run length info of model prediction

        ### draw ###
        ##### initalize label for legned, this code should be write before writing barchart #####
        init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.RS_CLASS], height=height, color=colors[self.RS_CLASS], alpha=alpha_ratio[self.RS_CLASS], linewidth=edge_linewidth[self.RS_CLASS], edgecolor=edge_colors[self.RS_CLASS]) # dummy data
        init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.NRS_CLASS], height=height, color=colors[self.NRS_CLASS], alpha=alpha_ratio[self.NRS_CLASS], linewidth=edge_linewidth[self.NRS_CLASS], edgecolor=edge_colors[self.NRS_CLASS]) # dummy data
        init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.NEG_HARD_CLASS], height=height, color=colors[self.NEG_HARD_CLASS], alpha=alpha_ratio[self.NEG_HARD_CLASS], linewidth=edge_linewidth[self.NEG_HARD_CLASS], edgecolor=edge_colors[self.NEG_HARD_CLASS]) # dummy data
        init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.POS_HARD_CLASS], height=height, color=colors[self.POS_HARD_CLASS], alpha=alpha_ratio[self.POS_HARD_CLASS], linewidth=edge_linewidth[self.POS_HARD_CLASS], edgecolor=edge_colors[self.POS_HARD_CLASS]) # dummy data
        init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.NEG_VANILA_CLASS], height=height, color=colors[self.NEG_VANILA_CLASS], alpha=alpha_ratio[self.NEG_VANILA_CLASS], linewidth=edge_linewidth[self.NEG_VANILA_CLASS], edgecolor=edge_colors[self.NEG_VANILA_CLASS]) # dummy data
        init_bar = ax.barh(range(len(yticks)), np.zeros(len(yticks)), label=label_names[self.POS_VANILA_CLASS], height=height, color=colors[self.POS_VANILA_CLASS], alpha=alpha_ratio[self.POS_VANILA_CLASS], linewidth=edge_linewidth[self.POS_VANILA_CLASS], edgecolor=edge_colors[self.POS_VANILA_CLASS]) # dummy data
	    ##### #### #### #### ##### #### #### #### 

        # data processing for barchart
        data = np.array(runlength_model.to_numpy()) # width
        data_cum = data.cumsum(axis=0) # for calc start index

        # draw bar
        for i, frame_class in enumerate(runlength_class) :
            # print(data[i,:])
            # print(frame_class)

            widths = data[i,:]
            starts= data_cum[i,:] - widths
            
            bar = ax.barh(range(len(yticks)), widths, left=starts, height=height, color=colors[frame_class], alpha=alpha_ratio[frame_class], linewidth=edge_linewidth[frame_class], edgecolor=edge_colors[frame_class]) # don't input label

        ### write on figure 
        # set title
        title_name = 'Sampling of {}'.format(self.patient_name)
        sub_title_name =  'Model {}'.format(model_name)
        # fig.suptitle(title_name, fontsize=16)
        # ax.set_title(sub_title_name)
        
        # non visual
        ax.xaxis.set_visible(False) # non visual x label
        ax.spines['left'].set_visible(False) # ax non visual boundary
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # set xticks pre section size
        ax.set_xticks(range(0, len(frame_label), window_size))
        
        xtick_labels = ['{}\n{}'.format(time, frame) if i_th % 2 == 0 else '\n\n{}\n{}'.format(time, frame) for i_th, (time, frame) in enumerate(zip(frame_label[::window_size], time_label[::window_size]))]
        ax.set_xticklabels(xtick_labels) # xtick change
        ax.xaxis.set_tick_params(labelsize=6)
        ax.set_xlabel('Frame / Time (h:m:s:fps)', fontsize=12)


        # set yticks
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=12)
        for label in ax.get_yticklabels(): label.set_fontweight('bold')
        # ax.set_ylabel('Methods', fontsize=20)

        # 8. legend
        box = ax.get_position() # 범례를 그래프상자 밖에 그리기위해 상자크기를 조절
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        # ax.legend(label_names, loc='center left', bbox_to_anchor=(1,0.5), shadow=True, ncol=1)

        print('===> VISUALIZATION...')


        # 11. write sampling count
        for i, (key, sampling_list) in enumerate(visual_data.items()):
            unique, counts = np.unique(sampling_list, return_counts=True)
            counts_dict = dict(zip(unique, counts))

            neg_hard_sampling_count = counts_dict.get(self.NEG_HARD_CLASS, 0)
            pos_hard_sampling_count = counts_dict.get(self.POS_HARD_CLASS, 0)
            neg_vanila_sampling_count = counts_dict.get(self.NEG_VANILA_CLASS, 0)
            pos_vanila_sampling_count = counts_dict.get(self.POS_VANILA_CLASS, 0)

            rs_count = gt_list.count(self.RS_CLASS)
            nrs_count = gt_list.count(self.NRS_CLASS)

            total_sampling_count = neg_hard_sampling_count + pos_hard_sampling_count + neg_vanila_sampling_count + pos_vanila_sampling_count

            text_bar = ax.barh(i, 0, height=height) # dummy data
            text = '{} | [GT_RS - {} GT_NRS - {}] || Total Sampling: {} | Neg Hard: {} | Pos Hard: {} | Neg Vanila: {} | Pos Vanila: {}'.format(key, rs_count, nrs_count, total_sampling_count, neg_hard_sampling_count, pos_hard_sampling_count, neg_vanila_sampling_count, pos_vanila_sampling_count)
            print(text)

            # self.present_text(ax, text_bar, text)

        print('===> COUNTING DONE.')

        ### file write
        fig.tight_layout() # subbplot 간격 줄이기
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

        plt.show()
        plt.savefig(self.save_path, format='png', dpi=500)

        plt.close(fig)

        print('==> SAVE DONE.')
