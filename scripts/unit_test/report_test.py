def report_anno(video_path, save_path, event_sequence, inference_interval):
    from core.utils.report import ReportAnnotation
    from scripts.unit_test.ffmpegHelper_test import get_video_meta_info_test
    from core.utils.misc import get_nrs_frame_chunk, get_current_time

    frameRate, totalFrame, width, height = get_video_meta_info_test(video_path)
    # frameRate, totalFrame, width, height = 10000, 20000, 30000, 40000

    _id = "61efa"
    annotationType = "NRS"
    createdAt = get_current_time()[0]
    updatedAt = createdAt
    annotator = "30"
    name = os.path.splitext(os.path.basename(video_path))[0]
    label = {"1": "NonRelevantSurgery"}

    nrs_frame_chunk = get_nrs_frame_chunk(event_sequence, inference_interval)

    annotation_report = ReportAnnotation(save_path)

    # set meta info
    annotation_report.set_total_report(totalFrame, frameRate, width, height, _id, annotationType, createdAt, updatedAt, annotator, name, label)
    
    # add nrs annotation info
    nrs_cnt = len(nrs_frame_chunk)
    for i, (start_frame, end_frame) in enumerate(nrs_frame_chunk, 1):
        # print('\n\n[{}] \t {} - {}'.format(i, start_frame, end_frame))

        # check over totalFrame on last annotation (becaseu of quntization? when set up inference_interval > 1)
        if nrs_cnt == i and end_frame >= totalFrame: 
            end_frame = totalFrame - 1

        annotation_report.add_annotation_report(start_frame, end_frame, code=1)

    annotation_report.save_report()

    print(save_path)

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

    inference_interval = 30

    ### inferene vector by test annotation ###
    annotation_parser = AnnotationParser('/data3/Public/NRS_Recog/annotation/Gastrectomy/Lapa/v3/gangbuksamsung_127case/NRS/04_GS4_99_L_1_01_NRS_30.json')
    event_sequence = annotation_parser.get_event_sequence(inference_interval)
    ### ###

    event_sequence = [1] * len(event_sequence)

    # report
    report_anno(input_video_path, annotation_by_inference_json_path, event_sequence, inference_interval)