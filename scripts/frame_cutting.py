

import os 
import glob
import sys

import natsort

def frame_cutting(target_video, frame_save_path):   
    print('\nframe cutting ... ')

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    sys.path.append(base_path)
    sys.path.append(base_path+'core')

    from core.utils.ffmpegHelper import ffmpegHelper

    # save_path 
    os.makedirs(frame_save_path, exist_ok=True)

    # frame cutting -> save to '$ frame_save_path~/frame-000000000.jpg' -> 변경 : '$ frame_save_path~/000000000.jpg'
    ffmpeg_helper = ffmpegHelper(target_video, frame_save_path)
    ffmpeg_helper.cut_frame_total()

def frame_cnt_parity_check():

    

if __name__ == '__main__':

    import os 
    import glob
    import sys
    import natsort

    import json
    import csv 

    video_list = glob.glob(os.path.join('/nas1/ViHUB-pro/vihub_robot_videos/B1_1st', '*', '*.mp4')) # 16건, 87개
    anno_list = glob.glob(os.path.join('/nas1/ViHUB-pro/vihub_robot_annos/NRS', '*.json')) # 16건, 44개

    video_list = natsort.natsorted(video_list)
    anno_list = natsort.natsorted(anno_list)

    list_items = []
    
    for anno in anno_list:
        anno_f = '_'.join(os.path.splitext(anno.split('/')[-1])[0].split('_')[:-2])
        for video in video_list:
            video_f = os.path.splitext(video.split('/')[-1])[0]

            if anno_f == video_f:
        
                patient_no = '_'.join(video_f.split('_')[3:5])
                output_path = os.path.join('/nas1/ViHUB-pro/vihub_robot_frames', patient_no, video_f)
                
                '''
                ## 비디오 전처리 (frmae 추출)
                frame_cutting(target_video=video, frame_save_path = output_path) 
                '''

                ## 프레임 수 일치 여부 확인
                with open(anno, 'r') as file:
                    data = json.load(file)

                    totalFrame = data['totalFrame']

                video_len = len(glob.glob(os.path.join(output_path, '*.jpg')))

                list_item = [video_f, totalFrame, video_len]
                list_items.append(list_item)

    
    with open('./vihub_robot.csv', 'w') as file:
        write = csv.writer(file)
        write.writerows(list_items)
