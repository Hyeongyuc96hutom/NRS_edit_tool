
def main():
    import os
    import glob
    import shutil

    video_path = '/nas1/Robot' 
    video_list = glob.glob(os.path.join(video_path, '*', '*'))

    for video in video_list:
        # print(video)
        patient_no = '_'.join(video.split('/')[-1].split('_')[3:5])
        video_name = video.split('/')[-1]



        if patient_no in ['R_2', 'R_6', 'R_13', 'R_74', 'R_100', 'R_202', 'R_301', 'R_302', 'R_311', 'R_312', 'R_313', 'R_336', 'R_362', 'R_363', 'R_386', 'R_405', 'R_418', 'R_423', 'R_424', 'R_526']:
            restore_path = os.path.join('/nas1/ViHUB-pro/input_video/train_100', patient_no)
            os.makedirs(restore_path, exist_ok=True)

            # # copy target_video
            print('COPY {} \n==========> {}\n'.format(video, os.path.join(restore_path, video_name))) # /data3/Public/ViHUB-pro/results/inference_json/train_100/Dataset1/R_2/01_G_01_R_2_ch1_01/01_G_01_R_2_ch1_01.mp4
            shutil.copy2(video, os.path.join(restore_path, video_name))


if __name__ == '__main__':
    main()