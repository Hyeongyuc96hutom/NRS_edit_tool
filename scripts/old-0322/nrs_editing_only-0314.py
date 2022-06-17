
## IDC NAS 에 업로드된 데이터 -> 사내 NAS 로 이동하기 위한 작업. 
## 프레임 추출, 인퍼런스 모두 생략 -> 비디오 편집만을 하기 위함.  

def get_experiment_args():
    from core.config.base_opts import parse_opts

    parser = parse_opts()

    args = parser.parse_args()

    ### model basic info opts
    args.pretrained = True
    # TODO 원하는대로 변경 하기
    # 전 그냥 save path와 동일하게 가져갔습니다. (bgpark)
    args.save_path = args.save_path + '-trial:{}-fold:{}'.format(args.trial, args.fold)
    # args.save_path = args.save_path + '-model:{}-IB_ratio:{}-WS_ratio:{}-hem_extract_mode:{}-top_ratio:{}-seed:{}'.format(args.model, args.IB_ratio, args.WS_ratio, args.hem_extract_mode, args.top_ratio, args.random_seed) # offline method별 top_ratio별 IB_ratio별 실험을 위해
    args.experiments_sheet_dir = args.save_path

    ### dataset opts
    args.data_base_path = '/raid/img_db'

    ### train args
    args.num_gpus = 1
    
    ### etc opts
    args.use_lightning_style_save = True # TO DO : use_lightning_style_save==False 일 경우 오류해결 (True일 경우 정상작동)

    return args


def extract_target_video_path(target_video_path):
    '''
    # input_video 이후의 core 한 path 추출 #

    target_video_path = '/data3/Public/ViHUB-pro/input_video/train_100/R_2/01_G_01_R_2_ch1_01.mp4'
    core_output_path = 'train_100/R_2/01_G_01_R_2_ch1_01'
    '''
    
    import os

    split_path_list = target_video_path.split('/')
    edited_path_list = []

    for i, split_path in enumerate(split_path_list):
        if split_path == 'input_video':
            edited_path_list = split_path_list[i+1:]

    core_output_path = '/'.join(edited_path_list)
    core_output_path = '.'.join(core_output_path.split('.')[:-1])

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


def save_meta_log(target_video, base_output_path):
    import json
    from collections import OrderedDict

    import os
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
        with open(os.path.join(meta_log_path, 'editing_logs', 'editing_log-vihub.json'), 'r+') as f:
            data = json.load(f)
            data.update(meta_data)

            f.seek(0)
            json.dump(data, f, indent=2)
    except:
        os.makedirs(os.path.join(meta_log_path, 'editing_logs'), exist_ok=True)

        with open(os.path.join(meta_log_path, 'editing_logs', 'editing_log-vihub.json'), 'w') as f:
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

    print('Anno copying ...')

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

# 22.01.07 hg modify, 기존 target_video 내부에 생성 => 설정된 frame_save_path 생성되도록 변경
def frame_cutting(target_video, frame_save_path):
    import sys
    import os    

    print('\nframe cutting ... ')

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
    sys.path.append(base_path)
    sys.path.append(base_path+'core')

    from core.utils.ffmpegHelper import ffmpegHelper

    # save_path 
    os.makedirs(frame_save_path, exist_ok=True)

    # frame cutting -> save to '$ frame_save_path~/frame-000000000.jpg'
    ffmpeg_helper = ffmpegHelper(target_video, frame_save_path)
    ffmpeg_helper.cut_frame_total()

    return frame_save_path

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


def main():
    import os
    import glob
    import re
    import natsort

    '''
    # annotation_by_inference_json = glob.glob(os.path.join(annotation_by_inference_path, '*', '*', '*' ,'*.json'))
    # annotation_by_inference_total_csv = glob.glob(os.path.join(annotation_by_inference_path, '*', '*', '*', '*.csv'))
    
    # annotation_by_inference_csv = []
    # annotation_by_inference_pp_csv = []

    # for f_csv in annotation_by_inference_total_csv:
    #     split_list = f_csv.split('/')[-1].split('-')
    #     if len(split_list) == 1:
    #         annotation_by_inference_csv.append(f_csv)

    #     if len(split_list) == 2:
    #         annotation_by_inference_pp_csv.append(f_csv)

    # print('******************'*3)
    # print('inference_json', len(annotation_by_inference_json))
    # for i in annotation_by_inference_json:
    #     print(i)
    # print('\n\n')
    
    # print('inference_csv', len(annotation_by_inference_csv))
    # for i in annotation_by_inference_csv:
    #     print(i)
    # print('\n\n')
    
    # print('inference_pp_csv', len(annotation_by_inference_pp_csv))
    # for i in annotation_by_inference_pp_csv:
    #     print(i)

    # print('\n\n')
    # print('******************'*3)
    '''
    
    # inference_csv_path = '/NRS_EDITING/mobilenet-set2-sota/inference_results'
    inference_csv_path = '/NRS_EDITING/mobilenet-set1-apply1/TB_log/version_4/inference_results'
    inference_csv = glob.glob(os.path.join(inference_csv_path, '*', '*', '*.csv')) # 123건, 234개

    # input_path = '/nas1/ViHUB-pro/input_video' # 비디오 path
    input_path = '/nas1/ViHUB-pro/input_video/ViHUB_0314' # 비디오 path
    base_output_path = '/nas1/ViHUB-pro/results'

    # 추후 args로 받아올 경우 해당 변수를 args. 로 초기화
    seq_fps = 1 # pp module (1 fps 로 inference) - pp에서 사용 (variable이 고정되어 30 fps인 비디오만 사용하는 시나이오로 적용, 비디오에 따라 유동적으로 var이 변하도록 계산하는게 필요해보임)
    inference_interval = 30 # frame inference interval - 전체사용 (variable이 고정되어 30 fps인 비디오를 1fps로 Infeence 하는 시나리오로 적용, 비디오에 따라 유동적으로 var이 변하도록 계산하는게 필요해보임)
    ## model_path = '/NRS_EDITING/logs_sota/mobilenetv3_large_100-theator_stage100-offline-sota/version_4/checkpoints/epoch=60-Mean_metric=0.9875-best.ckpt'
    # model_path = '/NRS_EDITING/mobilenet-set2-sota/checkpoints/epoch=48-Mean_metric=0.9892-best.ckpt'
    model_path = '/NRS_editing/mobilenet-set1-apply1/TB_log/version_4/checkpoints/epoch=52-Mean_metric=0.9894-best.ckpt'

    for (root, dirs, files) in os.walk(input_path): # 비디오

        for file in files:
            
            target_video = os.path.join(root, file) # 타깃 비디오
            core_output_path = extract_target_video_path(target_video) # ~./input_video 이후의 core 한 path 추출: 'train_100/R_2/01_G_01_R_2_ch1_01'

            inference_json_output_path = os.path.join(base_output_path, 'inference_json', core_output_path) # '/nas1/ViHUB-pro/results + inference_json + train_100/R_2/01_G_01_R_2_ch1_01'
            edited_video_output_path = os.path.join(base_output_path, 'edited_video', core_output_path) # '/nas1/ViHUB-pro/results + edited_video + train_100/R_2/01_G_01_R_2_ch1_01'

            #########################################################################################
            
            ## 중복 비디오 확인
            if check_exist_dupli_video(target_video, inference_json_output_path=inference_json_output_path, edited_video_output_path=edited_video_output_path): # True: 중복된 비디오 있음. False : 중복된 비디오 없음.
                # print('[NOT RUN] ALREADY EXITST IN OUTPUT PATH {}'.format(os.path.join(root, file)))
                continue

            # extention 예외 처리 ['mp4'] 
            if file.split('.')[-1] not in ['mp4']: # 22.01.07 hg comment, sanity한 ffmpeg module 사용을 위해 우선 .mp4 만 사용하는 것이 좋을 것 같습니다.
                continue

            target_csv_anno = None

            for f_anno in inference_csv: # target_video 에 대해 csv 를 가지고 있는 경우에만 수행. 
                # print(core_output_path.split('/')[-1])
                # print('.'.join(f_anno.split('/')[-1].split('.')[:-1]))
                if core_output_path.split('/')[-1] == '.'.join(f_anno.split('/')[-1].split('.')[:-1]):
                    target_csv_anno = f_anno
                    # print(target_csv_anno)

            if target_csv_anno == None:
                # print('* csv 없어서 실행 안 함 *', target_video)
                continue

            # #########################################################################################

            print('\n\n\n', '+++++++++'*10)
            print('*[target video] Processing in {}'.format(target_video))
            print('*[output path 1] {}'.format(inference_json_output_path))
            print('*[output path 2] {}\n'.format(edited_video_output_path))

            os.makedirs(inference_json_output_path, exist_ok=True)
            os.makedirs(edited_video_output_path, exist_ok=True)

            ## get video meta info
            frameRate, totalFrame, width, height = get_video_meta_info_from_ffmpeg(target_video) # from ffmpeg
            video_name = os.path.splitext(os.path.basename(target_video))[0]

            ## 비디오 복사
            video_copy_to_save_dir(target_video=target_video, output_path=inference_json_output_path)
            predict_csv_path = anno_copy_to_save_dir(target_anno=target_csv_anno, output_path=inference_json_output_path)

            '''
            # annotation_by_inference 복사
            predict_json_path = anno_copy_to_save_dir(target_anno=target_json_anno, output_path=inference_json_output_path)
            predict_csv_path = anno_copy_to_save_dir(target_anno=target_csv_anno, output_path=inference_json_output_path)
            predict_pp_csv_path = anno_copy_to_save_dir(target_anno=target_pp_csv_anno, output_path=inference_json_output_path)
            '''

            ## 비디오 전처리 (frmae 추출) -> 임시 디렉토리
            # frame_save_path = frame_cutting(target_video=target_video, frame_save_path = os.path.join(inference_json_output_path, 'frames')) 
            
            ## inference (비디오 단위) -> 저장 디렉토리 & result csv 생성 
            # predict_csv_path = inference(target_dir = frame_save_path, inference_interval = inference_interval, result_save_path = inference_json_output_path, model_path = model_path) 

            # Post-processing prepare 1. csv to sequence vector
            event_sequence = get_event_sequence_from_csv(predict_csv_path)

            # Post-processing prepare 2. apply PP module
            predict_pp_csv_path = apply_post_processing(predict_csv_path, seq_fps) # csv vector to applid pp csv vector
            event_sequence = get_event_sequence_from_csv(predict_pp_csv_path) # reload - applied pp csv vector

            # # Post-processing prepare 3. check unmatching total frame
            # frame_cnt = len(glob.glob(os.path.join(frame_save_path, '*.jpg'))) # 실제 frame 개수
            # if frame_cnt != totalFrame: # 
            #     print('>>>>> UNMATCH FRAME CNT <<<<< \t extrated frame_cnt : {} \t != \t totalFrame by ffmpeg : {} '.format(frame_cnt, totalFrame))  # TODO - logging
            #     totalFrame = frame_cnt           

            ## save annotation by inference (hvat form)
            report_annotation(frameRate, totalFrame, width, height, video_name, event_sequence, inference_interval, result_save_path=os.path.join(inference_json_output_path, '{}-annotation_by_inference.json'.format(video_name)))

            ## 비디오 편집 (ffmpep script)
            video_editing(video_path=target_video, event_sequence=event_sequence, editted_video_path=os.path.join(edited_video_output_path, '{}-edit.mp4'.format(video_name)), inference_interval=inference_interval, video_fps=frameRate)
            
            ## meta_log 파일 생성
            save_meta_log(target_video=target_video, base_output_path=base_output_path)

            ## 임시 디렉토리 삭제
            # del_frames(frame_save_path)


if __name__ == '__main__':
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        sys.path.append(base_path+'/core')

    main()