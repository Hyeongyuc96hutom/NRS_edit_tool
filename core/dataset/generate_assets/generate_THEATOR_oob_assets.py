
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

def parsing_oob_list_mid_video(json_file):
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
        for time in time_list:
            for i in range(time[0], time[1] + 1):
                oob_list.append('{:010d}'.format(i))

    return oob_list

def parsing_oob_list_first_video(json_file):
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
        for time in time_list:
            if time[0] == 0:
                for i in range(time[0], time[1] + 1):
                    oob_list.append('{:010d}'.format(i))

    return oob_list

def parsing_oob_list_last_video(json_file):
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
                for i in range(time[0], time[1] + 1):
                    oob_list.append('{:010d}'.format(i))

    return oob_list

def frame_length_parity_check(json_file):
    with open(json_file) as json_file:
        json_data = json.load(json_file)
        json_total_frame = json_data["totalFrame"]
    
    return json_total_frame

def save_log(save_path, log_txt):
    with open(save_path, 'a') as f:
        f.write(log_txt+'\n')
    
# annotation_version_base_path, Device (ROBOT, LAPA)
def make_oob_csv(anno_base_path, img_base_path, save_path, device):
    total_annotation_list = glob.glob(anno_base_path + '/*')
    os.makedirs(save_path, exist_ok=True)

    if device.lower() == 'robot':
        anno_list = [i for i in total_annotation_list if '_R_' in i]
        anno_list = natsort.natsorted(anno_list)
        print('anno_list : {}\n\n'.format(anno_list))
        target_img_base_path = os.path.join(img_base_path, 'ROBOT')

    elif device.lower() == 'lapa':
        anno_list = [i for i in total_annotation_list if '_L_' in i]
        anno_list = natsort.natsorted(anno_list)
        print('anno_list : {}\n\n'.format(anno_list))
        target_img_base_path = os.path.join(img_base_path, 'LAPA')
    

    patients_dict = defaultdict(list)
    
    for anno_file in anno_list:
        patients_dict['_'.join(anno_file.split('/')[-1].split('_')[:5])].append(anno_file)
        patients_list = list(patients_dict.values())
    
    outbody_list = []
    inbody_list = []

    for patient_list in patients_list: # ['./01_G_01_R_1_ch1_01_TBE_30.json', './01_G_01_R_1_ch1_03_TBE_30.json', './01_G_01_R_1_ch1_06_TBE_30.json']
    
        for i, anno_path in enumerate(patient_list): # './01_G_01_R_1_ch1_01_TBE_30.json'
            oob_list = []

            if len(patient_list) == 1:
                oob_list_1 = parsing_oob_list_first_video(anno_path)
                oob_list_2 = parsing_oob_list_last_video(anno_path)
                
                oob_list = oob_list_1 + oob_list_2
            
            else:
                if i == 0:
                    # 첫번째 비디오
                    oob_list = parsing_oob_list_first_video(anno_path)
                    print(anno_path)
                    print('첫번째 비디오 oob_list ===> ', oob_list)

                elif i == len(patient_list)-1:
                    # 마지막 비디오
                    oob_list = parsing_oob_list_last_video(anno_path)
                    print(anno_path)
                    print('마지막 비디오 oob_list ===> ', oob_list)

                else:
                    # TODO 중간 비디오  
                    oob_list = []   


            patient_folder_name = '_'.join(anno_path.split('/')[-1].split('_')[3:5]) # R_94
            video_folder_name = '_'.join(anno_path.split('/')[-1].split('_')[:7]) # 01_G_01_R_94_ch1_03

            target_img_path = os.path.join(target_img_base_path, patient_folder_name, video_folder_name)
            target_img_list = glob.glob(os.path.join(target_img_path, '*.jpg'))
            
            target_img_list_length = len(target_img_list)

            ## if annotation total frame < target img list
            ## annotation total frame 만큼 target img list 자르기. (img db 에 프레임이 더 있다는 의미 -> img db 프레임 자름)
            if frame_length_parity_check(anno_file) < target_img_list_length:
                print('========\tframe length parity check START\t========')
                target_img_list = natsort.natsorted(target_img_list)
                print('ORIGIN FRAME LENGTH ====> {}'.format(target_img_list_length))

                target_img_list  = target_img_list[:frame_length_parity_check(anno_file)]
                print('CONVERT FRAME LENGTH ====> {}'.format(len(target_img_list)))
                print('\t====>', frame_length_parity_check(anno_file), len(target_img_list))
                print('========\tframe length parity check END\t========')

            print('target_img_path : {}\n'.format(target_img_path))

            for target_img in target_img_list:
                target_img_idx = target_img.split('-')[1][:-4]
                
                if int(target_img_idx) % 30 == 0: # train step : 30
                    if target_img_idx in oob_list:
                        outbody_list.append([target_img, 1])

                    else:
                        inbody_list.append([target_img, 0])


        inbody_list = natsort.natsorted(inbody_list)
        outbody_list = natsort.natsorted(outbody_list)

        save_list_to_csv(os.path.join(save_path, 'oob_assets_outofbody.csv'), outbody_list, 'w')
        save_list_to_csv(os.path.join(save_path, 'oob_assets_inbody.csv'), inbody_list, 'w')




if __name__ == '__main__':
    import sys
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from core.config.assets_info import annotation_path, img_db_path, oob_assets_save_path

    make_oob_csv(anno_base_path=annotation_path['annotation_v3_base_path'], img_base_path=img_db_path['12'], save_path=oob_assets_save_path['theator-oob_assets_v3_robot_save_path'], device='robot')
