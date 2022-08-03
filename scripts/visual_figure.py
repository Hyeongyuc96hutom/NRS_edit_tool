def visual_for_prediction_results(patient_gt_list, patient_no, model_name, results_dict, save_dir):
     # figure b, predict results per methods
    from core.api.visualization import VisualTool # visual module

    import os
    import pandas as pd
    import glob

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'prediction_results_of_{}.png'.format(patient_no))

    visual_tool = VisualTool(patient_gt_list, patient_no, save_path)
    # visual_tool.visual_predict_multi(results_dict, model_name, inference_interval=30, window_size=300, section_num=2)
    visual_tool.visual_predict_multi(results_dict, model_name, inference_interval=6, window_size=1500, section_num=2)
    # visual_tool.visual_predict_multi(results_dict, model_name, inference_interval=6, window_size=200, section_num=2)


def visual_for_sampling_results(patient_no, model_name, results_dict, save_dir):
     # figure b, predict results per methods
    from core.api.visualization import VisualTool # visual module

    import os
    import pandas as pd
    import glob

    import numpy as np

    from scripts.unit_test import test_visual_sampling

    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'sampling_results_of_{}.png'.format(patient_no))

    split_assets_multi = {}

    upper_bound_frame = 2500

    def get_patient_no(img_db_path):
        cleand_file_name = os.path.splitext(os.path.basename(img_db_path))[0]
        file_info, frame_idx = cleand_file_name.split('-')
        
        hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no = file_info.split('_')
        patient_no = '_'.join([op_method, patient_idx])

        return patient_no

    def get_video_no(img_db_path):
        cleand_file_name = os.path.splitext(os.path.basename(img_db_path))[0]
        file_info, frame_idx = cleand_file_name.split('-')
        
        hospital, surgery_type, surgeon, op_method, patient_idx, video_channel, video_slice_no = file_info.split('_')
        video_no = '_'.join([op_method, patient_idx,video_channel,video_slice_no])

        return video_no

    def get_frame_idx(img_db_path):
        cleand_file_name = os.path.splitext(os.path.basename(img_db_path))[0]
        file_info, frame_idx = cleand_file_name.split('-')

        return int(frame_idx)
    
    def split_hem_vanila(assets_df):
        NON_HEM_CLASS, HEM_CLASS = (0,1)
        RS_CLASS, NRS_CLASS = (0,1)
        split_assets = {
            'neg_hard_idx':[],
            'pos_hard_idx':[],
            'neg_vanila_idx':[],
            'pos_vanila_idx':[],
        }

        split_assets['neg_hard_idx'] += np.array(assets_df.loc[(assets_df['HEM'] == HEM_CLASS) & (assets_df['class_idx'] == RS_CLASS)]['consensus_frame_idx']).tolist()
        split_assets['pos_hard_idx'] += np.array(assets_df.loc[(assets_df['HEM'] == HEM_CLASS) & (assets_df['class_idx'] == NRS_CLASS)]['consensus_frame_idx']).tolist()
        split_assets['neg_vanila_idx'] += np.array(assets_df.loc[(assets_df['HEM'] == NON_HEM_CLASS) & (assets_df['class_idx'] == RS_CLASS)]['consensus_frame_idx']).tolist()
        split_assets['pos_vanila_idx'] += np.array(assets_df.loc[(assets_df['HEM'] == NON_HEM_CLASS) & (assets_df['class_idx'] == NRS_CLASS)]['consensus_frame_idx']).tolist()

        return split_assets
    
    # 0. load patinet gt
    patients_assets = PatientsGT()   

    patient_gt = patients_assets.get_gt(patient_no=patient_no) # get patinets gt
    video_start_idx_list = patients_assets.get_start_idx(patient_no=patient_no) # get video len
    video_no_list = patients_assets.get_video_no(patient_no=patient_no) # get video_no
    
    # 1. load hem assets and concensus 
    for key, hem_assets_path in results_dict.items():
        hem_df = pd.read_csv(hem_assets_path)

        hem_df['patient_no'] = hem_df['img_path'].apply(get_patient_no) # extract patinet_no from image db path
        hem_df['video_no'] = hem_df['img_path'].apply(get_video_no) # extract frame index from image db path
        hem_df['frame_idx'] = hem_df['img_path'].apply(get_frame_idx) # extract frame index from image db path
        hem_df['consensus_frame_idx'] = hem_df['img_path'].apply(get_frame_idx) # init consensus_frame_idx

        # 2. get only patinet df
        patient_hem_df = hem_df[hem_df['patient_no'] == patient_no]

        for video_no, video_start_idx in zip(video_no_list, video_start_idx_list): # consunsus frame index
            is_video_no = patient_hem_df['video_no'] == video_no
            patient_hem_df.loc[is_video_no,'consensus_frame_idx'] = patient_hem_df.loc[is_video_no,'frame_idx'] + video_start_idx
        
        # sorting (저장 할때 편히보려고 sorting)
        patient_hem_df = patient_hem_df.sort_values(by=['consensus_frame_idx'], axis=0)

        ### 조정
        patient_hem_df = patient_hem_df[patient_hem_df['consensus_frame_idx'] < upper_bound_frame]

        # save
        patient_hem_df.to_csv(os.path.join(save_dir, '{}-{}.csv'.format(key, patient_no)))

        # parsing hem/vanila assets info
        split_assets = split_hem_vanila(patient_hem_df)

        # final visualization assets
        split_assets_multi[key] = split_assets # split_assets['neg_hard_idx'], split_assets['pos_hard_idx'], split_assets['neg_vanila_idx'], split_assets['pos_vanila_idx']

    print('===> PREPAREING VISUALIZAION ASSETS DONE.')
    ####### hem assets visualization #######
    ### 조정
    patient_gt = patient_gt[:upper_bound_frame] 
    # visualization
    visual_tool = VisualTool(gt_list=patient_gt, patient_name=patient_no, save_path=os.path.join(save_dir, 'sampling-{}.png'.format(patient_no)))
    visual_tool.visual_sampling_multi(split_assets_multi, model_name=model_name)        

    
def get_inference_results_per_patient2(inference_results_dir, patient_no, post_processing, get_gt=False):
    import os

    from core.utils.parser import FileLoader
    from core.utils.prepare import PatientsGT

    file_loader = FileLoader()    

    # patient_gt = PatientsGT()
    # videos = patient_gt.get_video_no(patient_no)

    patient_predict_results_dir = os.path.join(inference_results_dir, patient_no)

    if post_processing:
        patient_predict_results_path = os.path.join(patient_predict_results_dir, '{}-pp.csv'.format(patient_no))
    else:
        patient_predict_results_path = os.path.join(patient_predict_results_dir, '{}-original.csv'.format(patient_no))

    print(patient_predict_results_path)

    patient_predict_list = file_loader.set_file_path(patient_predict_results_path)
    predict_df = file_loader.load()

    if get_gt :
        patient_predict_list = predict_df['gt'].tolist()
    
    else:
        patient_predict_list = predict_df['predict'].tolist()
    

    return patient_predict_list

def get_inference_results_per_patient3(inference_results_dir, patient_no, get_gt=False):
    import os
    import glob
    import natsort 
    import pandas as pd

    from core.utils.parser import FileLoader

    file_loader = FileLoader()    

    # patient_gt = PatientsGT()
    # videos = patient_gt.get_video_no(patient_no)

    patient_predict_results_dir = os.path.join(inference_results_dir, patient_no)
    pd_paths = natsort.natsorted(glob.glob(os.path.join(patient_predict_results_dir, '*-gt.csv')))

    patient_predict_list = []
    patient_img_list = []
    for pd_path in pd_paths:
        predict_df = pd.read_csv(pd_path)

        if get_gt :
            predict_list = predict_df['gt'].tolist()
            img_list = predict_df['target_img'].tolist()
        
        else:
            predict_list = predict_df['predict'].tolist()
            img_list = predict_df['target_img'].tolist()

        patient_predict_list += predict_list
        patient_img_list += img_list
    
    # pre_idx = int(9000/30 * 5)
    # pro_idx = int(27000/30 * 5)
    # pro_idx = -1
    # patient_predict_list = patient_predict_list[pre_idx:pro_idx]
    
    # patient_img_list = patient_img_list[pre_idx:pro_idx]
    
    print(patient_img_list[0])
    print(patient_img_list[-1])


    return patient_predict_list


def get_inference_results_per_patient(inference_results_dir, patient_no, post_processing, get_gt=False):
    import os
    import glob
    import natsort

    from core.utils.parser import FileLoader
    from core.utils.prepare import PatientsGT

    file_loader = FileLoader()    

    '''
    patient_gt = PatientsGT()
    videos = patient_gt.get_video_no(patient_no)
    '''
    patient_predict_results_dir = os.path.join(inference_results_dir, patient_no)

    print(patient_predict_results_dir)
    
    videos_dir = natsort.natsorted(glob.glob(os.path.join(patient_predict_results_dir, '*')))

    patient_predict_list = []

    print('\n---- ----- ----- \n')
    print(patient_no)

    for video_no in videos_dir:
        if not os.path.isdir(video_no):
            continue
        video_no = os.path.basename(video_no)

        print(video_no)

        video_predict_results_path = os.path.join(patient_predict_results_dir, video_no, '{}.csv'.format(video_no))
        
        file_loader.set_file_path(video_predict_results_path)
        predict_df = file_loader.load()

        if get_gt:
            patient_predict_list += predict_df['gt'].tolist()
        else:
            patient_predict_list += predict_df['predict'].tolist()

    return patient_predict_list

def get_inference_pp_results_per_patient(inference_results_dir, patient_no):
    import os

    from core.utils.parser import FileLoader
    from core.utils.prepare import PatientsGT

    from core.utils.post_processing import FilterBank

    file_loader = FileLoader()    

    patient_gt = PatientsGT()
    videos = patient_gt.get_video_no(patient_no)

    patient_predict_results_dir = os.path.join(inference_results_dir, patient_no)

    patient_predict_list = []

    for video_no in videos:
        video_predict_results_path = os.path.join(patient_predict_results_dir, video_no, '{}.csv'.format(video_no))
        
        file_loader.set_file_path(video_predict_results_path)
        predict_df = file_loader.load()

        patient_predict_list += predict_df['predict'].tolist()
    
    # post processing
    ##### example 1. when you want to apply best pp filter
    fb = FilterBank(patient_predict_list, seq_fps=1)
    patient_predict_list = fb.apply_best_filter()

    ##### example 2. when you want to apply custimizing sequence of pp filter
    # fb2 = FilterBank(seq_list, seq_fps) # seq_fps
    '''
    patient_predict_list = fb.apply_filter(patient_predict_list, "opening", kernel_size=1) 
    patient_predict_list = fb.apply_filter(patient_predict_list, "closing", kernel_size=1)
    '''

    return patient_predict_list

if __name__ == '__main__':
    if __package__ is None:
        import sys
        import os
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core/accessory/RepVGG')
        print(base_path)
    
    
    ### figure b ####
    val_patients = ['R_2', 'R_6', 'R_13', 'R_74', 'R_100', 'R_202', 'R_301', 'R_302', 'R_311', 'R_312', 'R_313', 'R_336', 'R_362', 'R_363', 'R_386', 'R_405', 'R_418', 'R_423', 'R_424', 'R_526']
    val_patients = ['R_301']
    # val_patients = ['R_6', 'R_46', 'R_154', 'R_155', 'R_156', 'R_157', 'R_158','R_160', 'R_161', 'R_162', 'R_163', 'R_164', 'R_165','R_166' , 'R_167', 'R_168', 'R_169', 'R_170', 'R_171', 'R_172']
    # val_patients = ['01_VIHUB1.2_A9_L_5', '01_VIHUB1.2_A9_L_6', '01_VIHUB1.2_A9_L_18', '01_VIHUB1.2_A9_L_19', '01_VIHUB1.2_A9_L_20', '01_VIHUB1.2_A9_L_21', '01_VIHUB1.2_A9_L_24', '01_VIHUB1.2_A9_L_27', '01_VIHUB1.2_A9_L_30', '01_VIHUB1.2_A9_L_35', '01_VIHUB1.2_A9_L_38', '01_VIHUB1.2_A9_L_40', '01_VIHUB1.2_A9_L_47', '01_VIHUB1.2_A9_L_48', '01_VIHUB1.2_A9_L_51', '01_VIHUB1.2_B4_L_1', '01_VIHUB1.2_B4_L_2', '01_VIHUB1.2_B4_L_4', '01_VIHUB1.2_B4_L_6', '01_VIHUB1.2_B4_L_7', '01_VIHUB1.2_B4_L_8', '01_VIHUB1.2_B4_L_9', '01_VIHUB1.2_B4_L_12', '01_VIHUB1.2_B4_L_16', '01_VIHUB1.2_B4_L_17', '01_VIHUB1.2_B4_L_20', '01_VIHUB1.2_B4_L_22', '01_VIHUB1.2_B4_L_24', '01_VIHUB1.2_B4_L_26', '01_VIHUB1.2_B4_L_28', '01_VIHUB1.2_B4_L_29', '01_VIHUB1.2_B4_L_75', '01_VIHUB1.2_B4_L_76', '01_VIHUB1.2_B4_L_82', '01_VIHUB1.2_B4_L_84', '01_VIHUB1.2_B4_L_86', '01_VIHUB1.2_B4_L_87', '01_VIHUB1.2_B4_L_90', '01_VIHUB1.2_B4_L_91', '01_VIHUB1.2_B4_L_94', '01_VIHUB1.2_B4_L_98', '01_VIHUB1.2_B4_L_100', '01_VIHUB1.2_B4_L_103', '01_VIHUB1.2_B4_L_106', '01_VIHUB1.2_B4_L_107', '01_VIHUB1.2_B4_L_108', '01_VIHUB1.2_B4_L_111', '01_VIHUB1.2_B4_L_113', '01_VIHUB1.2_B4_L_115', '01_VIHUB1.2_B4_L_120', '01_VIHUB1.2_B4_L_121', '01_VIHUB1.2_B4_L_123', '01_VIHUB1.2_B4_L_127', '01_VIHUB1.2_B4_L_130', '01_VIHUB1.2_B4_L_131', '01_VIHUB1.2_B4_L_134', '01_VIHUB1.2_B4_L_136', '01_VIHUB1.2_B4_L_137', '01_VIHUB1.2_B4_L_139', '01_VIHUB1.2_B4_L_143', '01_VIHUB1.2_B4_L_144', '01_VIHUB1.2_B4_L_146', '01_VIHUB1.2_B4_L_149', '01_VIHUB1.2_B4_L_150', '01_VIHUB1.2_B4_L_151', '01_VIHUB1.2_B4_L_152', '01_VIHUB1.2_B4_L_153', '01_VIHUB1.2_B5_L_9', '04_GS4_99_L_3', '04_GS4_99_L_4', '04_GS4_99_L_7', '04_GS4_99_L_11', '04_GS4_99_L_12', '04_GS4_99_L_16', '04_GS4_99_L_17', '04_GS4_99_L_26', '04_GS4_99_L_28', '04_GS4_99_L_29', '04_GS4_99_L_37', '04_GS4_99_L_38', '04_GS4_99_L_39', '04_GS4_99_L_40', '04_GS4_99_L_42', '04_GS4_99_L_44', '04_GS4_99_L_46', '04_GS4_99_L_48', '04_GS4_99_L_49', '04_GS4_99_L_50', '04_GS4_99_L_58', '04_GS4_99_L_59', '04_GS4_99_L_60', '04_GS4_99_L_61', '04_GS4_99_L_64', '04_GS4_99_L_65', '04_GS4_99_L_66', '04_GS4_99_L_71', '04_GS4_99_L_72', '04_GS4_99_L_75', '04_GS4_99_L_79', '04_GS4_99_L_82', '04_GS4_99_L_84', '04_GS4_99_L_86', '04_GS4_99_L_87', '04_GS4_99_L_88', '04_GS4_99_L_89', '04_GS4_99_L_92', '04_GS4_99_L_94', '04_GS4_99_L_95', '04_GS4_99_L_96', '04_GS4_99_L_98', '04_GS4_99_L_99', '04_GS4_99_L_102', '04_GS4_99_L_103', '04_GS4_99_L_104', '04_GS4_99_L_106', '04_GS4_99_L_107', '04_GS4_99_L_108', '04_GS4_99_L_114', '04_GS4_99_L_116', '04_GS4_99_L_117', '04_GS4_99_L_120', '04_GS4_99_L_125', '04_GS4_99_L_126']

    # val_patients = ['01_VIHUB1.2_B4_L_20_01'] # win 1 
    # val_patients = ['R_301'] # win 5
    # val_patients = ['R_418'] # win 2
    # val_patients = ['04_GS4_99_L_106_01'] # win 2
    # val_patients = ['R_202'] # win 2
    # val_patients = ['R_336'] # win 5
    # val_patients = ['01_VIHUB1.2_A9_L_35_01'] # win 1

    for patient_no in val_patients:
    # patient_no = 'R_336'
        print('[+] {}'.format(patient_no))
        model_name = 'mobilenet_v3_large'
        save_dir = os.path.join('/NRS_EDIT', 'figures', 'robot-details')

        # patient_gt_list = get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result', 'rs'), patient_no, get_gt=True) # GT

        # results_dict = {
        #     'RaS': get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result', 'rs'), patient_no, get_gt=False), # rs,
        #     'BS': get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result', 'ws'), patient_no, get_gt=False), # ws,
        #     'Theator': get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result', 'theator'), patient_no, get_gt=False), # theator
        #     'Offline(s=1)': get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result', 'offline'), patient_no, get_gt=False), # offline
        #     # 'Offline(s=2)': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'mobilenet_iter200_apply-MC=5', 'mobilenet_iter200_apply-MC=5-softmax_diff_small', 'version_4', 'inference_results_pp-30'), patient_no, post_processing=False), # offline 2
        #     'Online': get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result', 'online'), patient_no, get_gt=False), # online
        # }

        # robot - 5fps
        patient_gt_list = get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result_5fps', 'robot-RS'), patient_no, get_gt=True) # GT

        results_dict = {
            '(a)': get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result_5fps', 'robot-RS'), patient_no, get_gt=False), # rs,
            '(b)': get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result_5fps', 'robot-WS'), patient_no, get_gt=False), # ws,
            '(c)': get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result_5fps', 'robot-theator'), patient_no, get_gt=False), # theator
            '(d)': get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result_5fps', 'robot-offline'), patient_no, get_gt=False), # offline
            # 'Offline(s=2)': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'mobilenet_iter200_apply-MC=5', 'mobilenet_iter200_apply-MC=5-softmax_diff_small', 'version_4', 'inference_results_pp-30'), patient_no, post_processing=False), # offline 2
            '(e)': get_inference_results_per_patient3(os.path.join('/NRS_EDIT', '2022_miccai_rebuttal_assets', 'robot_20', 'inference_result_5fps', 'robot-online'), patient_no, get_gt=False), # online
        }

        '''
        patient_gt = PatientsGT()
        patient_gt_list = patient_gt.get_gt(patient_no)
        patient_gt_list = patient_gt_list[::30]
        '''
        '''
        patient_gt_list = get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'infer_log', 'mobilenet-rs-general-trial_1-fold_1', 'TB_log', 'version_0', 'inference_results_pp-30'), patient_no, post_processing=False, get_gt=True) # GT

        results_dict = {
            'RaS': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'infer_log', 'mobilenet-rs-general-trial_1-fold_1', 'TB_log', 'version_0', 'inference_results_pp-30'), patient_no, post_processing=False), # rs
            # 'BS': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'infer_log', 'mobilenet-ws-general-trial_1-fold_1', 'TB_log', 'version_0', 'inference_results-30'), patient_no, post_processing=True), # ws
            'Theator': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'infer_log', 'mobilenet-ws-replace', 'TB_log', 'version_2', 'inference_results_pp-30'), patient_no, post_processing=False), # thetor
            'Offline(s=1)': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'mobilenet_iter100_apply-MC=5', 'version_4', 'inference_results_pp-30'), patient_no, post_processing=False), # offline 1
            # 'Offline(s=2)': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'mobilenet_iter200_apply-MC=5', 'mobilenet_iter200_apply-MC=5-softmax_diff_small', 'version_4', 'inference_results_pp-30'), patient_no, post_processing=False), # offline 2
            'Online': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'infer_log', 'mobilenet-ws-proxy-4', 'TB_log', 'version_0', 'inference_results_pp-30'), patient_no, post_processing=False), # online            
        }
        '''

        '''
        patient_gt_list = get_inference_results_per_patient(os.path.join('/OOB_RECOG', 'logs-lapa', 'mobilenet-set2-general1', 'TB_log', 'version_0', 'inference_results'), patient_no, post_processing=False, get_gt=True) # GT

        results_dict = {
            'RaS': get_inference_results_per_patient(os.path.join('/OOB_RECOG', 'logs-lapa', 'mobilenet-set2-general1', 'TB_log', 'version_0', 'inference_results'), patient_no, post_processing=False), # rs        
            'Offline(s=1)': get_inference_results_per_patient(os.path.join('/OOB_RECOG', 'logs-lapa', 'mobilenet-set2-apply1', 'TB_log', 'version_4', 'inference_results'), patient_no, post_processing=False), # offline 1
            # 'Offline(s=2)': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'mobilenet_iter200_apply-MC=5', 'mobilenet_iter200_apply-MC=5-softmax_diff_small', 'version_4', 'inference_results_pp-30'), patient_no, post_processing=False), # offline 2
            'Online': get_inference_results_per_patient(os.path.join('/OOB_RECOG', 'logs-lapa', 'mobilenet-proxy-4-lapa', 'TB_log', 'version_0', 'inference_results'), patient_no, post_processing=False), # online            
        }
        '''

        
        
        '''
        ### post procesing figure
        patient_gt_list = get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'infer_log', 'mobilenet-rs-general-trial_1-fold_1', 'TB_log', 'version_0', 'inference_results_pp-30'), patient_no, post_processing=True, get_gt=True) # GT

        results_dict = {
            'RaS': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'infer_log', 'mobilenet-rs-general-trial_1-fold_1', 'TB_log', 'version_0', 'inference_results_pp-30'), patient_no, post_processing=True), # rs
            'BS': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'infer_log', 'mobilenet-ws-general-trial_1-fold_1', 'TB_log', 'version_0', 'inference_results_pp-30'), patient_no, post_processing=True), # ws
            'Theator': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'infer_log', 'mobilenet-ws-replace', 'TB_log', 'version_2', 'inference_results_pp-30'), patient_no, post_processing=True), # thetor
            'Offline(s=1)': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'mobilenet_iter100_apply-MC=5', 'version_4', 'inference_results_pp-30'), patient_no, post_processing=True), # offline 1
            'Offline(s=2)': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'mobilenet_iter200_apply-MC=5', 'mobilenet_iter200_apply-MC=5-softmax_diff_small', 'version_4', 'inference_results_pp-30'), patient_no, post_processing=True), # offline 2
            'Online': get_inference_results_per_patient2(os.path.join('/OOB_RECOG', 'post_processing', 'models', 'infer_log', 'mobilenet-ws-proxy-4', 'TB_log', 'version_0', 'inference_results_pp-30'), patient_no, post_processing=True), # online            
        }
        '''
        ### quantative figure
        
        '''
        # window_size 1
        patient_gt_list = pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'lapa', '01_VIHUB1.2_B4_L_20_01', 'demo_csv', 'rs_01_VIHUB1.2_B4_L_20_01.csv'))['gt'].tolist()

        pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'lapa', '01_VIHUB1.2_B4_L_20_01', 'demo_csv', 'rs_01_VIHUB1.2_B4_L_20_01.csv'))

        results_dict = {
            'RaS': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'lapa', '01_VIHUB1.2_B4_L_20_01', 'demo_csv', 'rs_01_VIHUB1.2_B4_L_20_01.csv'))['predict'].tolist(),
            'Offline': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'lapa', '01_VIHUB1.2_B4_L_20_01', 'demo_csv', 'off_01_VIHUB1.2_B4_L_20_01.csv'))['predict'].tolist(),
            'Online': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'lapa', '01_VIHUB1.2_B4_L_20_01', 'demo_csv', 'on_01_VIHUB1.2_B4_L_20_01.csv'))['predict'].tolist(),
        }
        '''
        
        
        '''
        # window_size 5
        patient_gt_list = pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'robot', 'R_301', 'demo_csv', 'off_R_301_ch1_01.csv'))['gt'].tolist()

        results_dict = {
            'RaS': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'robot', 'R_301', 'demo_csv', 'rs_R_301_ch1_01.csv'))['predict'].tolist(),
            '[3]': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'robot', 'R_301', 'demo_csv', 'theator_R_301_ch1_01.csv'))['predict'].tolist(),
            'Offline': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'robot', 'R_301', 'demo_csv', 'off_R_301_ch1_01.csv'))['predict'].tolist(),
            'Online': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'robot', 'R_301', 'demo_csv', 'on_R_301_ch1_01.csv'))['predict'].tolist(),
        }
        '''
        '''
        # window_size 2
        patient_gt_list = pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'robot', 'R_418', 'demo_csv', 'off_R_418_ch1_02.csv'))['gt'].tolist()

        results_dict = {
            'RaS': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'robot', 'R_418', 'demo_csv', 'rs_R_418_ch1_02.csv'))['predict'].tolist(),
            '[3]': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'robot', 'R_418', 'demo_csv', 'theator_R_418_ch1_02.csv'))['predict'].tolist(),
            'Offline': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'robot', 'R_418', 'demo_csv', 'off_R_418_ch1_02.csv'))['predict'].tolist(),
            'Online': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_A-FP', 'robot', 'R_418', 'demo_csv', 'on_R_418_ch1_02.csv'))['predict'].tolist(),
        }
        '''
        
        
        '''
        # window_size 2
        patient_gt_list = pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'lapa', '04_GS4_99_L_106_01', 'demo_csv', 'off_04_GS4_99_L_106_01.csv'))['gt'].tolist()

        results_dict = {
            'RaS': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'lapa', '04_GS4_99_L_106_01', 'demo_csv', 'rs_04_GS4_99_L_106_01.csv'))['predict'].tolist(),
            'Offline': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'lapa', '04_GS4_99_L_106_01', 'demo_csv', 'off_04_GS4_99_L_106_01.csv'))['predict'].tolist(),
            'Online': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'lapa', '04_GS4_99_L_106_01', 'demo_csv', 'on_04_GS4_99_L_106_01.csv'))['predict'].tolist(),
        }
        '''
        
        '''
        # window_size 2
        
        patient_gt_list = pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'robot', 'R_202', 'demo_csv', 'off_R_202_ch1_01.csv'))['gt'].tolist()

        results_dict = {
            'RaS': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'robot', 'R_202', 'demo_csv', 'rs_R_202_ch1_01.csv'))['predict'].tolist(),
            '[3]': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'robot', 'R_202', 'demo_csv', 'theator_R_202_ch1_01.csv'))['predict'].tolist(),
            'Offline': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'robot', 'R_202', 'demo_csv', 'off_R_202_ch1_01.csv'))['predict'].tolist(),
            'Online': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'robot', 'R_202', 'demo_csv', 'on_R_202_ch1_01.csv'))['predict'].tolist(),
        }
        '''
        
        

        # window_size 5
        '''
        patient_gt_list = pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'robot', 'R_336', 'demo_csv', 'off_R_336_ch1_01.csv'))['gt'].tolist()

        results_dict = {
            'RaS': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'robot', 'R_336', 'demo_csv', 'rs_R_336_ch1_01.csv'))['predict'].tolist(),
            '[3]': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'robot', 'R_336', 'demo_csv', 'theator_R_336_ch1_01.csv'))['predict'].tolist(),
            'Offline': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'robot', 'R_336', 'demo_csv', 'off_R_336_ch1_01.csv'))['predict'].tolist(),
            'Online': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'robot', 'R_336', 'demo_csv', 'on_R_336_ch1_01.csv'))['predict'].tolist(),
        }
        '''

        '''
        # window_size 5
        patient_gt_list = pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'lapa', '01_VIHUB1.2_A9_L_35_01', 'demo_csv', 'rs_01_VIHUB1.2_A9_L_35_01.csv'))['gt'].tolist()

        results_dict = {
            'RaS': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'lapa', '01_VIHUB1.2_A9_L_35_01', 'demo_csv', 'rs_01_VIHUB1.2_A9_L_35_01.csv'))['predict'].tolist(),
            'Offline': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'lapa', '01_VIHUB1.2_A9_L_35_01', 'demo_csv', 'off_01_VIHUB1.2_A9_L_35_01.csv'))['predict'].tolist(),
            'Online': pd.read_csv(os.path.join('/OOB_RECOG', 'DEMO_VIDEO_no_video', 'CASE_B-TP', 'lapa', '01_VIHUB1.2_A9_L_35_01', 'demo_csv', 'on_01_VIHUB1.2_A9_L_35_01.csv'))['predict'].tolist(),
        }
        '''
        

        visual_for_prediction_results(patient_gt_list, patient_no, model_name, results_dict, save_dir) # figure b, predict results per methods
        print('[-] {}'.format(patient_no))
        print('------' * 5)

    '''
    ### figure c ###
    # post process
    val_patients = ['R_2', 'R_6', 'R_13', 'R_74', 'R_100', 'R_202', 'R_301', 'R_302', 'R_311', 'R_312', 'R_313', 'R_336', 'R_362', 'R_363', 'R_386', 'R_405', 'R_418', 'R_423', 'R_424', 'R_526']

    val_patients = ['R_13'] # (8361) # 10분가량
    for patient_no in val_patients:
        # patient_no = 'R_336'
        model_name = 'mobilenet_v3_large'
        save_dir = '/OOB_RECOG/figures/predictions_post_part'

        patient_gt = PatientsGT()
        patient_gt_list = patient_gt.get_gt(patient_no)
        patient_gt_list = patient_gt_list[::30]
    
    
        results_dict = {
            '(a)': get_inference_results_per_patient(os.path.join('/OOB_RECOG', 'logs_robot-offline-iter100', 'mobilenet_iter100_apply-MC=5', 'TB_log', 'version_4', 'inference_results'), patient_no),
            '(b)': get_inference_pp_results_per_patient(os.path.join('/OOB_RECOG', 'logs_robot-offline-iter100', 'mobilenet_iter100_apply-MC=5', 'TB_log', 'version_4', 'inference_results'), patient_no), # post processing
        }

        visual_for_prediction_results(patient_gt_list, patient_no, model_name, results_dict, save_dir) # figure b, predict results per methods
    '''

    







