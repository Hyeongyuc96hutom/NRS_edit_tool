
import os
import csv
import glob
import json
import natsort

from collections import defaultdict


def save_list_to_csv(save_filepath, list, mode):
    with open(save_filepath, mode) as f:
        writer = csv.writer(f)
        for row in list:
            writer.writerow(row)



def parsing_last_video_last_annotation(json_file):
    with open(json_file) as json_file:
        time_list = []
        json_data = json.load(json_file)
        json_anno = json_data["annotations"]
        
        for list in json_anno:
            temp_list = []
            temp_list.append(int(list.get("start"))) # float -> int
            temp_list.append(int(list.get("end"))) # float -> int
            time_list.append(temp_list)

        oob_list = []
        for ids, time in enumerate(time_list):
            if ids == len(time_list)-1:
                return time[1]

def parsing_frame_length(json_file):
    with open(json_file) as json_file:
        json_data = json.load(json_file)
        json_total_frame = json_data["totalFrame"]
    
    return json_total_frame

def save_log(save_path, log_txt):
    with open(save_path, 'a') as f:
        f.write(log_txt+'\n')
    
# annotation_version_base_path, Device (ROBOT, LAPA)
def last_video_annotation_parity_check(anno_base_path, img_base_path, save_path, device):
    '''
        대상 : 각 환자의 마지막 비디오
        각 환자의 마지막 비디오의 마지막 어노테이션 (oob 표기) 이 끝까지 되어 있는지 확인. 
    '''
    total_annotation_list = glob.glob(anno_base_path + '/*')
    os.makedirs(save_path, exist_ok=True)

    if device.lower() == 'robot':
        anno_list = [i for i in total_annotation_list if '_R_' in i]
        anno_list = natsort.natsorted(anno_list)
        # print('anno_list : {}\n\n'.format(anno_list))
        target_img_base_path = os.path.join(img_base_path, 'ROBOT')

    elif device.lower() == 'lapa':
        anno_list = [i for i in total_annotation_list if '_L_' in i]
        anno_list = natsort.natsorted(anno_list)
        # print('anno_list : {}\n\n'.format(anno_list))
        target_img_base_path = os.path.join(img_base_path, 'LAPA')
    

    patients_dict = defaultdict(list)
    
    for anno_file in anno_list:
        patients_dict['_'.join(anno_file.split('/')[-1].split('_')[:5])].append(anno_file)
        patients_list = list(patients_dict.values())
    
    check_list = []
    # inbody_list = []

    for patient_list in patients_list: # ['./01_G_01_R_1_ch1_01_TBE_30.json', './01_G_01_R_1_ch1_03_TBE_30.json', './01_G_01_R_1_ch1_06_TBE_30.json']
    
        for i, anno_path in enumerate(patient_list): # './01_G_01_R_1_ch1_01_TBE_30.json'
            oob_list = []

            if i == len(patient_list)-1:
                # 마지막 비디오
                last_annotation_point = parsing_last_video_last_annotation(anno_path)
                total_annotation_cnt = parsing_frame_length(anno_path)

                if last_annotation_point < total_annotation_cnt:
                    check_list.append(anno_path)


    for i in check_list:
        print(i)



            
    # save_list_to_csv(os.path.join(save_path, 'oob_assets_outofbody.csv'), outbody_list, 'w')
    # save_list_to_csv(os.path.join(save_path, 'oob_assets_inbody.csv'), inbody_list, 'w')




if __name__ == '__main__':
    import sys
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from core.config.assets_info import annotation_path, img_db_path, oob_assets_save_path

    last_video_annotation_parity_check(anno_base_path=annotation_path['annotation_v3_base_path'], img_base_path=img_db_path['12'], save_path='./anno_parity_check', device='robot')