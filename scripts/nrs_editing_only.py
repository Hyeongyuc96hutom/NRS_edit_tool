
'''
    프레임 추출, 인퍼런스 모두 생략
    비디오 편집만을 하기 위함.  
'''

def extract_target_video_path(target_video_path):
    import os
    '''
    target_video_path = '/data3/Public/ViHUB-pro/input_video/train_100/R_2/01_G_01_R_2_ch1_01.mp4'
    core_output_path = 'train_100/R_2/01_G_01_R_2_ch1_01'
    '''

    split_path_list = target_video_path.split('/')
    edited_path_list = []

    for i, split_path in enumerate(split_path_list):
        if split_path == 'input_video':
            edited_path_list = split_path_list[i+1:]

    core_output_path = '/'.join(edited_path_list)
    core_output_path = os.path.splitext(core_output_path)[0]

    return core_output_path

def get_video_meta_info(target_video, base_output_path):
    import os
    import glob
    import datetime

    inference_json_output_path = os.path.join(base_output_path, 'inference_json', extract_target_video_path(target_video))

    video_name = target_video.split('/')[-1]
    video_path = target_video    
    frame_list = glob.glob(os.path.join(inference_json_output_path, 'frames', '*.jpg'))
    date_time = str(datetime.datetime.now())

    return video_name, video_path, len(frame_list), date_time


def save_meta_log(save_f, target_video, base_output_path):
    import os
    import json
    from collections import OrderedDict

    '''
    	"04_GS4_99_L_1_01.mp4": {
            "video_path": "/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01.mp4",
            "frame_cnt": 108461,
            "date": "2022-01-06 17:55:36.291518"
	}
    '''

    print('\nmeta log saving ...')

    meta_log_path = os.path.join(base_output_path, 'logs')
    os.makedirs(meta_log_path, exist_ok=True)
    
    meta_data = OrderedDict()
    video_name, video_path, frame_cnt, date_time = get_video_meta_info(target_video, base_output_path)

    meta_data[video_name] = {
        'video_path': video_path,
        'frame_cnt': frame_cnt,
        'date': date_time
    }

    print(json.dumps(meta_data, indent='\t'))

    try: # existing editing_log.json 
        with open(os.path.join(meta_log_path, 'editing_logs', 'editing_log-{}.json'.format(save_f)), 'r+') as f:
            data = json.load(f)
            data.update(meta_data)

            f.seek(0)
            json.dump(data, f, indent=2)
    except:
        os.makedirs(os.path.join(meta_log_path, 'editing_logs'), exist_ok=True)

        with open(os.path.join(meta_log_path, 'editing_logs', 'editing_log-{}.json'.format(save_f)), 'w') as f:
            json.dump(meta_data, f, indent=2)


def get_video_meta_info_from_ffmpeg(video_path):
    from core.utils.ffmpegHelper import ffmpegHelper

    print('\n\n \t\t <<<<< GET META INFO FROM FFMPEG >>>>> \t\t \n\n')

    ffmpeg_helper = ffmpegHelper(video_path)
    fps = ffmpeg_helper.get_video_fps()
    video_len = ffmpeg_helper.get_video_length()
    width, height = ffmpeg_helper.get_video_resolution()

    return fps, video_len, width, height


def extract_video_clip(video_path, results_dir, start_time, duration):
    from core.utils.ffmpegHelper import ffmpegHelper

    ffmpeg_helper = ffmpegHelper(video_path, results_dir)
    ffmpeg_helper.extract_video_clip(start_time, duration)


def parse_clips_paths(clips_root_dir):
    import os
    import glob
    from natsort import natsort

    target_clip_paths = glob.glob(os.path.join(clips_root_dir, 'clip-*.mp4'))
    target_clip_paths = natsort.natsorted(target_clip_paths)

    save_path = os.path.join(clips_root_dir, 'clips.txt')

    logging = ''

    for clip_path in target_clip_paths:
        txt = 'file \'{}\''.format(os.path.abspath(clip_path))
        logging += txt + '\n'
    
    print(logging)

    # save txt
    with open(save_path, 'w') as f :
        f.write(logging)

    return save_path


def clips_to_video(clips_root_dir, merge_path):
    from core.utils.ffmpegHelper import ffmpegHelper

    # parsing clip video list 
    input_txt_path = parse_clips_paths(clips_root_dir)

    ffmpeg_helper = ffmpegHelper("dummy", "dummy")
    ffmpeg_helper.merge_video_clip(input_txt_path, merge_path)


## 5. 22.01.07 hg write, inference_interval, video_fps는 clips의 extration time 계산시 필요
def video_editing(video_path, event_sequence, editted_video_path, inference_interval, video_fps):
    import os
    from core.utils.misc import get_rs_time_chunk

    print('\nvideo editing ...')

    # temp process path
    temp_process_dir = os.path.dirname(editted_video_path) # clips, 편한곳에 생성~(현재는 edit되는 비디오 밑에 ./clips에서 작업)
    temp_clip_dir = os.path.join(temp_process_dir, 'clips')
    os.makedirs(temp_clip_dir, exist_ok=True)

    # 0. inference vector(sequence) to rs chunk
    target_clipping_time = get_rs_time_chunk(event_sequence, video_fps, inference_interval)


    print('\n\n \t\t <<<<< EXTRACTING CLIPS >>>>> \t\t \n\n')
    # 1. clip video from rs chunk
    for i, (start_time, duration) in enumerate(target_clipping_time, 1):
        print('\n\n[{}] \t {} - {}'.format(i, start_time, duration))
        extract_video_clip(video_path, temp_clip_dir, start_time, duration)

    # 2. merge video
    print('\n\n \t\t <<<<< MERGEING CLIPS >>>>> \t\t \n\n')
    clips_to_video(clips_root_dir = temp_clip_dir, merge_path = editted_video_path)

    # 3. TODO: delete temp process path (delete clips)


def video_copy_to_save_dir(target_video, output_path):
    import os
    import shutil

    print('\nVideo copying ...')

    video_name = output_path.split('/')[-1] # 01_G_01_R_2_ch1_01
    video_ext = target_video.split('.')[-1] # .mp4

    video_name = video_name + '.' + video_ext

    # copy target_video
    print('COPY {} \n==========> {}\n'.format(target_video, os.path.join(output_path, video_name))) # /data3/Public/ViHUB-pro/results/inference_json/train_100/Dataset1/R_2/01_G_01_R_2_ch1_01/01_G_01_R_2_ch1_01.mp4
    shutil.copy2(target_video, os.path.join(output_path, video_name))


def anno_copy_to_save_dir(target_anno, output_path):
    import os
    import shutil

    print('\Anno copying ...')

    f_anno_name = target_anno.split('/')[-1]

    # copy target_video
    print('COPY {} \n==========> {}\n'.format(target_anno, os.path.join(output_path, f_anno_name))) # /data3/Public/ViHUB-pro/results/inference_json/train_100/Dataset1/R_2/01_G_01_R_2_ch1_01/01_G_01_R_2_ch1_01.json
    shutil.copy2(target_anno, os.path.join(output_path, f_anno_name))

    predict_csv_path = os.path.join(output_path, f_anno_name)

    return predict_csv_path # /data3/Public/ViHUB-pro/results/inference_json/train_100/Dataset1/R_2/01_G_01_R_2_ch1_01/01_G_01_R_2_ch1_01.json
    

def check_exist_dupli_video(target_video, inference_json_output_path, edited_video_output_path):
    import os

    if os.path.exists(inference_json_output_path) and os.path.exists(edited_video_output_path):
        return True
    
    return False

def get_event_sequence_from_csv(predict_csv_path):
    import pandas as pd

    return pd.read_csv(predict_csv_path)['predict'].values.tolist()

def set_inference_interval(origin_fps):
    return int(round(origin_fps))

# predict csv to applied-pp predict csv
def apply_post_processing(predict_csv_path, seq_fps):
    import os
    import pandas as pd
    from core.api.post_process import FilterBank
    
    # 1. load predict df
    predict_df = pd.read_csv(predict_csv_path, index_col=0)
    event_sequence = predict_df['predict'].tolist()

    # 2. apply pp sequence
    #### use case 1. best filter
    fb = FilterBank(event_sequence, seq_fps)
    best_pp_seq_list = fb.apply_best_filter()

    #### use case 2. custimize filter
    '''
    best_pp_seq_list = fb.apply_filter(event_sequence, "opening", kernel_size=3)
    best_pp_seq_list = fb.apply_filter(best_pp_seq_list, "closing", kernel_size=3)
    '''

    predict_df['predict'] = best_pp_seq_list

    # 3. save pp df
    d_, f_ = os.path.split(predict_csv_path)
    f_name, _ = os.path.splitext(f_)

    results_path = os.path.join(d_, '{}-pp.csv'.format(f_name)) # renaming file of pp
    predict_df.to_csv(results_path)

    return results_path

## 4. 22.01.07 hg new add, save annotation by inference (hvat form)
def report_annotation(frameRate, totalFrame, width, height, name, event_sequence, inference_interval, result_save_path):
    from core.utils.report import ReportAnnotation
    from core.utils.misc import get_nrs_frame_chunk, get_current_time
    
    ### static meta data ###
    createdAt = get_current_time()[0]
    updatedAt = createdAt
    _id = "temp"
    annotationType = "NRS"
    annotator = "temp"
    label = {"1": "NonRelevantSurgery"}
    ### ### ### ### ### ###

    nrs_frame_chunk = get_nrs_frame_chunk(event_sequence, inference_interval)

    annotation_report = ReportAnnotation(result_save_path) # load Report module

    # set meta info
    annotation_report.set_total_report(totalFrame, frameRate, width, height, _id, annotationType, createdAt, updatedAt, annotator, name, label)
    
    # add nrs annotation info
    nrs_cnt = len(nrs_frame_chunk)
    for i, (start_frame, end_frame) in enumerate(nrs_frame_chunk, 1):
        # print('\n\n[{}] \t {} - {}'.format(i, start_frame, end_frame))

        # check over totalFrame on last annotation (because of quntization? when set up inference_interval > 1)
        if nrs_cnt == i and end_frame >= totalFrame: 
            end_frame = totalFrame - 1

        annotation_report.add_annotation_report(start_frame, end_frame, code=1)

    annotation_report.save_report()
    
def main(inference_csv, model_path, seq_fps, input_path, base_output_path):
    import os
    import glob
    import re

    for (root, dirs, files) in os.walk(input_path):
        for file in files:
            
            target_video = os.path.join(root, file)
            core_output_path = extract_target_video_path(target_video) # 'train_100/R_2/01_G_01_R_2_ch1_01'
            inference_json_output_path = os.path.join(base_output_path, 'inference_json', core_output_path) # '/data3/Public/ViHUB-pro/results + inference_json + train_100/R_2/01_G_01_R_2_ch1_01'
            edited_video_output_path = os.path.join(base_output_path, 'edited_video', core_output_path) # '/data3/Public/ViHUB-pro/results + edited_video + train_100/R_2/01_G_01_R_2_ch1_01'

            #########################################################################################
            
            ## 중복 비디오 확인
            if check_exist_dupli_video(target_video, inference_json_output_path=inference_json_output_path, edited_video_output_path=edited_video_output_path): # True: 중복된 비디오 있음. False : 중복된 비디오 없음.
                print('[NOT RUN] ALREADY EXITST IN OUTPUT PATH {}'.format(os.path.join(root, file)))
                continue

            # extention 예외 처리 ['mp4'] 
            if file.split('.')[-1] not in ['mp4']: # 22.01.07 hg comment, sanity한 ffmpeg module 사용을 위해 우선 .mp4 만 사용하는 것이 좋을 것 같습니다.
                continue

            target_csv_anno = None

            for f_anno in inference_csv: # target_video 에 대해 csv 를 가지고 있는 경우에만 수행.
                if core_output_path.split('/')[-1] == os.path.splitext(f_anno)[0].split('/')[-1]:
                    target_csv_anno = f_anno

            if target_csv_anno == None:
                continue


            #########################################################################################

            print('\n', '+++++++++'*10)
            print('*[target video] Processing in {}'.format(target_video))
            print('*[output path 1] {}'.format(inference_json_output_path))
            print('*[output path 2] {}\n'.format(edited_video_output_path))

            os.makedirs(inference_json_output_path, exist_ok=True)
            os.makedirs(edited_video_output_path, exist_ok=True)

            ## get video meta info
            frameRate, totalFrame, width, height = get_video_meta_info_from_ffmpeg(target_video) # from ffmpeg
            video_name = os.path.splitext(os.path.basename(target_video))[0]

            # 22.03.22 jh 추가
            # inference_interval = 1fps (기존 30 fps 비디오 기준으로 고정 사용에서 -> 비디오에 따라 유동적으로 변경), inference_interval 은 30, 60 만 사용할 것으로 예상 (예외 처리)
            inference_interval = set_inference_interval(frameRate)
            assert inference_interval in [30, 60], 'Not supported inference interval size (only use 30 or 60): {}'.format(inference_interval)

            ## 비디오 복사
            video_copy_to_save_dir(target_video=target_video, output_path=inference_json_output_path)
            predict_csv_path = anno_copy_to_save_dir(target_anno=target_csv_anno, output_path=inference_json_output_path)

            # Post-processing prepare 2. apply PP module
            predict_pp_csv_path = apply_post_processing(predict_csv_path, seq_fps) # csv vector to applid pp csv vector
            event_sequence = get_event_sequence_from_csv(predict_pp_csv_path) # reload - applied pp csv vector         

            ## save annotation by inference (hvat form)
            report_annotation(frameRate, totalFrame, width, height, video_name, event_sequence, inference_interval, result_save_path=os.path.join(inference_json_output_path, '{}-annotation_by_inference.json'.format(video_name)))

            ## 비디오 편집 (ffmpep script)
            video_editing(video_path=target_video, event_sequence=event_sequence, editted_video_path=os.path.join(edited_video_output_path, '{}-edit.mp4'.format(video_name)), inference_interval=inference_interval, video_fps=frameRate)
            
            ## meta_log 파일 생성 & 임시 디렉토리 삭제
            save_meta_log(save_f=core_output_path.split('/')[0], target_video=target_video, base_output_path=base_output_path)

            # TODO : clip 영상 삭제
            #########################################################################################



if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core')

    import os, glob

    # set params
    inference_csv_path = '/NRS_editing/mobilenet-set1-apply1/TB_log/version_4/inference_results'
    inference_csv = glob.glob(os.path.join(inference_csv_path, '*', '*', '*.csv')) # 66건, 152개

    model_path = '/NRS_editing/logs_sota/mobilenetv3_large_100-theator_stage100-offline-sota/version_4/checkpoints/epoch=60-Mean_metric=0.9875-best.ckpt'
    seq_fps = 1 # pp module (1 fps 로 inference) - pp에서 사용 (variable이 고정되어 30 fps인 비디오만 사용하는 시나이오로 적용, 비디오에 따라 유동적으로 var이 변하도록 계산하는게 필요해보임) -> 질문!

    input_path = '/nas1/ViHUB-pro/input_video/ViHUB_test_case'

    main(inference_csv=inference_csv, model_path=model_path, seq_fps=seq_fps, input_path=input_path, base_output_path='/nas1/ViHUB-pro/results')

