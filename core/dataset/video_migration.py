
import os
import yaml
import shutil

def save_log(save_path, log_txt):
    with open(save_path, 'a') as f:
        f.write(log_txt)

def copy_video(src_path, dest_path):
    shutil.copy2(src_path, dest_path)

def main():
    with open('patients_aseets_0111.yaml') as f:
        patients = yaml.load(f, Loader=yaml.FullLoader)

    migration_video_list = []
    os.makedirs('/data3/Public/ViHUB-pro/input_video/train_100', exist_ok=True)

    for video in patients['patients']:
        # print(video['path_info'])

        for i in range(len(video['path_info'])):
            
            video_path = video['path_info'][i]['video_path']
            patient_no = '_'.join(video['path_info'][i]['video_name'].split('_')[:2])
            video_name = '01_G_01_'+video['path_info'][i]['video_name']
            video_ext = video_path.split('.')[-1]
            
            dest_base_path = os.path.join('/data3/Public/ViHUB-pro/input_video/train_100', patient_no)
            os.makedirs(dest_base_path, exist_ok=True)

            dest_path = os.path.join(dest_base_path, video_name+'.'+video_ext)
            
            migration_video_list.append([video_path, dest_path])

    print('***'*20)
    print(patients['patients'])
    print('***'*20)

    for target_migration_video in migration_video_list:
        log_txt = 'COPY FROM: {} ==> TO: {}\n'.format(target_migration_video[0], target_migration_video[1])
        print(log_txt)

        copy_video(target_migration_video[0], target_migration_video[1])
        save_log(os.path.join('/data3/Public/ViHUB-pro/input_video', 'train_100_video_migration_log.csv'), log_txt)

    # print(len(migration_video_list)) # 225개

if __name__ == '__main__':
    main()