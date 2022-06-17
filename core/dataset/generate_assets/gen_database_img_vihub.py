import os
import subprocess
import natsort

import glob

import re

import csv


video_base_path = '/data3/DATA/IMPORT/211220/12_14'
dest_base_path = '/raid/img_db/VIHUB'

def convert_patient_num(patient_num):
    return ''.join(re.findall('[1-9]\d*', patient_num))

def gen_dataset_using_ffmpeg(input_video_path, output_dir_path):
    print('\n\tProcessing ====>\t {}\n'.format(os.path.join(output_dir_path)))
    
    output_img_path = os.path.join(output_dir_path, '{}-%010d.jpg'.format(output_dir_path.split('/')[-1]))

    cmd = ['ffmpeg', '-i', input_video_path, '-start_number', '0', '-vf', 'scale=512:512']
    cmd += [output_img_path]

    print('Running: ', " ".join(cmd))
    subprocess.run(cmd)


def save_log(save_path, log_txt):
    with open(save_path, 'a') as f:
        f.write(log_txt)


def main(target_folder='gangbuksamsung_127case'):
    files = glob.glob(os.path.join(video_base_path, target_folder, '*', '*'))
    files = natsort.natsorted(files)

    for target_file in files:
        # print(file)

        ## 저장될 파일 이름
        ## L_100/04_GS4_99_L_100_01/01_G_01_R_1_ch1_01-0000000000.jpg
        target_patient = '_'.join(target_file.split('/')[-1].split('_')[3:5]) # L_100
        target_video = '.'.join(target_file.split('/')[-1].split('.')[:-1]) # 04_GS4_99_L_3_01

        # /raid/img_db/VIHUB/gangbuksamsung_127case/L_3/04_GS4_99_L_3_01
        output_path = os.path.join(dest_base_path, target_folder, target_patient, target_video)
        os.makedirs(output_path, exist_ok=True)
        
        gen_dataset_using_ffmpeg(target_file, output_path)

        save_log(os.path.join(dest_base_path, '{}_database_log.txt'.format(target_folder)), target_file+'\n')


def get_frame_count(target_path='/raid/img_db/VIHUB/gangbuksamsung_127case'):
    restore_list = []

    for (root, dirs, files) in os.walk(target_path):
    
        for dir in dirs:
            f_list = glob.glob(os.path.join(root, dir, '*.jpg'))
            # if len(f_list):
            print(dir, len(f_list))
            restore_list.append([dir, len(f_list)])


    restore_list = natsort.natsorted(restore_list)
    with open('{}.csv'.format(target_path.split('/')[-1]), 'w',newline='') as f: 
        write = csv.writer(f) 
        write.writerows(restore_list)

def annotation_check():
    video_path = '/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case'
    anno_path = '/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS'

    total_video = glob.glob(os.path.join(video_path, '*', '*'))
    total_anno = glob.glob(os.path.join(anno_path, '*'))

    for video in total_video:
        flag = False
        for anno in total_anno:
            if video.split('/')[-1].split('.')[0] == '_'.join(anno.split('/')[-1].split('.')[0].split('_')[:6]):
                flag = True

        if flag == False:
            print(video)
            



        




if __name__ == '__main__':
    # main('gangbuksamsung_127case')

    # main('severance_1st/01_VIHUB1.2_A9')
    # main('severance_1st/01_VIHUB1.2_B4')
    # main('severance_1st/L_9')

    # main('severance_1st/01_VIHUB1.2_B5')
    
    # main('severance_2nd/01_VIHUB1.2_A9')
    # main('severance_2nd/01_VIHUB1.2_B4')

    # get_frame_count('/raid/img_db/VIHUB/severance_2nd/01_VIHUB1.2_A9')
    # print('\n\n')
    # get_frame_count('/raid/img_db/VIHUB/severance_2nd/01_VIHUB1.2_B4')

    # get_frame_count('/data3/Public/NRS_Recog/img_db/ROBOT/ETC/ETC24')

    annotation_check()