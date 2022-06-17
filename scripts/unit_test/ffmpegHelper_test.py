def get_video_meta_info_test(video_path):
    from core.utils.ffmpegHelper import ffmpegHelper

    ffmpeg_helper = ffmpegHelper(video_path)
    fps = ffmpeg_helper.get_video_fps()
    video_len = ffmpeg_helper.get_video_length()
    width, height = ffmpeg_helper.get_video_resolution()
    print(fps, video_len, width, height)

    return fps, video_len, width, height

def extract_video_clip_test(video_path, results_dir, start_time, duration):
    from core.utils.ffmpegHelper import ffmpegHelper

    ffmpeg_helper = ffmpegHelper(video_path, results_dir)
    ffmpeg_helper.extract_video_clip(start_time, duration)


def parse_clips_paths(clips_root_dir):
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


if __name__ == '__main__':

    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
        sys.path.append(base_path)
        sys.path.append(base_path + '/core/accessory/RepVGG')
        

    import os
    import glob

    from core.utils.parser import AnnotationParser
    from core.utils.misc import get_rs_time_chunk

    # input path
    input_video_path = '/data3/DATA/IMPORT/211220/12_14/gangbuksamsung_127case/L_1/04_GS4_99_L_1_01.mp4'
    
    # output path
    results_dir = '/NRS_EDITING/results' # json, editted video
    annotation_by_inference_json_path = os.path.join(results_dir, 'annotation_by_inference.json') # save annotation json path
    editted_video_path = os.path.join(results_dir, 'editted_video.mp4') # editted video
    os.makedirs(results_dir, exist_ok=True)

    # process path
    temp_process_dir = '/NRS_EDITING/results/temp' # clip, frame
    temp_clip_dir = os.path.join(temp_process_dir, 'clips')
    os.makedirs(temp_clip_dir, exist_ok=True)

    inference_interval = 1
    video_fps= 30.0

    ### inferene vector by test annotation ###
    annotation_parser = AnnotationParser('/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS/04_GS4_99_L_1_01_NRS_30.json')
    event_sequence = annotation_parser.get_event_sequence(inference_interval)
    ### ###

    # 0. inference vector(sequence) to rs chunk
    target_clipping_time = get_rs_time_chunk(event_sequence, video_fps, inference_interval)

    # 1. clip video from rs chunk
    for i, (start_time, duration) in enumerate(target_clipping_time, 1):
        print('\n\n[{}] \t {} - {}'.format(i, start_time, duration))
        extract_video_clip_test(input_video_path, temp_clip_dir, start_time, duration)

    # 2. merge video
    clips_to_video(clips_root_dir = temp_clip_dir, merge_path = editted_video_path)