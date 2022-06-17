
import os
import csv
import glob
import json
import natsort

def save_list_to_csv(save_filepath, list, mode):
    with open(save_filepath, mode) as f:
        writer = csv.writer(f)
        for row in list:
            writer.writerow(row)

def parsing_oob_list(json_file):
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
    set_fps = 5
    cal = 30 // set_fps
    
    print('\n\n', '==='*10)
    print('\tCAL ===> {}'.format(cal))
    print('==='*10, '\n\n')
    
    total_annotation_list = glob.glob(anno_base_path + '/*')

    if device.lower() == 'robot':
        anno_list = [i for i in total_annotation_list if '_R_' in i]
        anno_list = natsort.natsorted(anno_list)
        print('anno_list : {}\n\n'.format(anno_list))
        target_img_base_path = img_base_path

    elif device.lower() == 'lapa':
        anno_list = [i for i in total_annotation_list if '_L_' in i]
        anno_list = natsort.natsorted(anno_list)
        print('anno_list : {}\n\n'.format(anno_list))
        target_img_base_path = os.path.join(img_base_path, 'LAPA')

    error_list = []

    outbody_list = []
    inbody_list = []

    for anno_file in anno_list:
        os.makedirs(save_path, exist_ok=True)
        
        try:
            if 'json' in anno_file:
                print('Processing in ====> {}\n'.format(anno_file))
                oob_list = parsing_oob_list(anno_file)
                print('oob_list : {}\n\n'.format(oob_list))
                save_log(os.path.join(save_path, 'oob_assets_log-fps={}.txt'.format(set_fps)), anno_file)
            else:
                continue

        except:
            print('====' * 5)
            print('ERROR: cannot parsing oob_list ====> {}'.format(anno_file))
            print('====' * 5)
            error_list.append([anno_file])
            save_log(os.path.join(save_path, 'oob_assets_log-fps={}.txt'.format(set_fps)), anno_file+'\t ====> ERROR!')

        patient_folder_name = '_'.join(anno_file.split('/')[-1].split('_')[3:5]) # R_94
        video_folder_name = '_'.join(anno_file.split('/')[-1].split('_')[:7]) # 01_G_01_R_94_ch1_03

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
            
            # if int(target_img_idx) % 30 == 0: # fps = 1
            ## fps = 30
            if int(target_img_idx) % cal == 0: # fps = 5
                if target_img_idx in oob_list:
                    outbody_list.append([target_img, 1])

                else:
                    inbody_list.append([target_img, 0])


    inbody_list = natsort.natsorted(inbody_list)
    outbody_list = natsort.natsorted(outbody_list)

    save_list_to_csv(os.path.join(save_path, 'oob_assets_outofbody-fps={}.csv'.format(set_fps)), outbody_list, 'w')
    save_list_to_csv(os.path.join(save_path, 'oob_assets_inbody-fps={}.csv'.format(set_fps)), inbody_list, 'w')

    if error_list:
        print('ERROR list ====> ', error_list)
        save_list_to_csv(os.path.join(output_save_path, 'ERROR_img-fps={}.csv'.format(set_fps)), error_list, 'w')


if __name__ == '__main__':
    import sys
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from core.config.assets_info import annotation_path, img_db_path, oob_assets_save_path

    make_oob_csv(anno_base_path=annotation_path['annotation_v3_base_path'], img_base_path=img_db_path['robot'], save_path=oob_assets_save_path['oob_assets_v3_robot_save_path'], device='robot')
