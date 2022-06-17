def visual_flow_for_sampling(final_assets_df, model_name, save_dir, window_size=9000, section_num=2):
    import os
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    from core.utils.prepare import PatientsGT
    from core.api.visualization import VisualTool
    
    print('======= \t VISUAL FLOW FOR SAMPLING \t =======')
    print(final_assets_df)

    sampling_visual_save_dir = os.path.join(save_dir, 'sampling_visualization')
    sampling_assets_save_dir = os.path.join(save_dir, 'sampling_assets')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sampling_visual_save_dir, exist_ok=True)
    os.makedirs(sampling_assets_save_dir, exist_ok=True)

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
    
    final_assets_df['patinet_no'] = final_assets_df['img_path'].apply(get_patient_no) # extract patinet_no from image db path
    final_assets_df['video_no'] = final_assets_df['img_path'].apply(get_video_no) # extract frame index from image db path
    final_assets_df['frame_idx'] = final_assets_df['img_path'].apply(get_frame_idx) # extract frame index from image db path
    final_assets_df['consensus_frame_idx'] = final_assets_df['img_path'].apply(get_frame_idx) # init consensus_frame_idx

    patients_assets = PatientsGT()    
    patients_grouped = final_assets_df.groupby('patinet_no') # grouping by patinets no

    # visualziation
    for patient_no, patient_df in tqdm(patients_grouped, desc='Sampling Visualization'): # per each patient
        patient_df = patient_df.reset_index(drop=True) # (should) reset index

        print('Patient:{} - Sampling: {}'.format(patient_no, len(patient_df)))
        
        patient_gt = patients_assets.get_gt(patient_no=patient_no) # get patinets gt
        video_start_idx_list = patients_assets.get_start_idx(patient_no=patient_no) # get video len
        video_no_list = patients_assets.get_video_no(patient_no=patient_no) # get video_no

        for video_no, video_start_idx in zip(video_no_list, video_start_idx_list): # consunsus frame index
            is_video_no = patient_df['video_no'] == video_no
            patient_df.loc[is_video_no,'consensus_frame_idx'] = patient_df.loc[is_video_no,'frame_idx'] + video_start_idx

        # sorting (저장 할때 편히보려고 sorting)
        patient_df = patient_df.sort_values(by=['consensus_frame_idx'], axis=0)
        print(patient_df)

        # parsing hem/vanila assets info
        split_assets = split_hem_vanila(patient_df)
        
        # save
        patient_df.to_csv(os.path.join(sampling_assets_save_dir, '{}.csv'.format(patient_no)))

        # visual
        visual_tool = VisualTool(gt_list=patient_gt, patient_name=patient_no, save_path=os.path.join(sampling_visual_save_dir, 'sampling-{}.png'.format(patient_no)))
        visual_tool.visual_sampling(split_assets['neg_hard_idx'], split_assets['pos_hard_idx'], split_assets['neg_vanila_idx'], split_assets['pos_vanila_idx'], model_name=model_name, window_size=window_size, section_num=section_num)

def main():
    import pandas as pd
    from glob import glob
    import natsort
    import os

    save_dir = '/OOB_RECOG/results_sampling'
    restore_path = '/OOB_RECOG/logs/hem-softmax-offline-IBRATIO=7/TB_log/version_4'

    # self.args.restore_path 에서 version0, 1, 2, 3 에 대한 hem.csv 읽고
    restore_path = os.path.dirname(restore_path)
    print(restore_path)
    
    # csv_path 내부의 모든 hem.csv (fold2, fold3, fold4, fold5) ==> 하나로 합침
    read_hem_csv = glob(os.path.join(restore_path, '*', '*-*-*-*.csv'))

    read_hem_csv = natsort.natsorted(read_hem_csv)
    
    print(read_hem_csv)

    hem_df_list = []
    cols = ['img_path', 'class_idx', 'HEM']

    for csv_file in read_hem_csv:
        df = pd.read_csv(csv_file, names=cols)
        hem_df_list.append(df)

    hem_assets_df = pd.concat(hem_df_list, ignore_index=True).reset_index(drop=True)

    hem_assets_df.to_csv(os.path.join(save_dir, 'hem_assets.csv'))    

    visual_flow_for_sampling(hem_assets_df, 'mobilenet', save_dir) # cols = ['img_path', 'class_idx', 'HEM'] 형태로 df만들어진 hem_assets_df 만 넘기면 돌아갑니다. // save_dir은 각 확자별 plt와 csv 결과를 저장할 parents directory입니다.

if __name__ == '__main__':
    
    if __package__ is None:
        import sys
        import os
        
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(base_path)

    main()

