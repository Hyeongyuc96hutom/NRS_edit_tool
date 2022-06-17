import numpy as np
import pandas as pd
from core.utils.report import ReportAnnotation
from core.utils.parser import AnnotationParser
from core.utils.misc import encode_list, decode_list
from core.utils.ffmpegHelper import ffmpegHelper

import os
from core.utils.misc import idx_to_time, time_to_sec, sec_to_time

class QAHelper():
    def __init__(self, extract_interval):
    
        self.extract_interval = extract_interval

        self.CORRECT_CLS = 100
        self.OVER_GT_CLS = 0
        self.OVER_INFER_CLS = 1
        self.ONLY_INFER_CLS = 2
        self.RS_CLS, self.NRS_CLS = (0,1)

        self.cnt = 0

    def set_extract_interval(extract_interval):
        self.extract_interval = extract_interval

    
    def excute(self, target_video_path, gt_json_path, inference_json_path, output_dir):
        # set argu
        video_name = os.path.splitext(os.path.basename(target_video_path))[0]
        compare_save_path = os.path.join(output_dir, '{}-compare.json'.format(video_name))
        marking_clip_save_dir = os.path.join(output_dir, video_name)

        os.makedirs(output_dir, exist_ok=True)

        # excute
        self._compare_to_json(gt_json_path, inference_json_path, self.extract_interval, compare_save_path)
        log_df = self._extract_video_from_json_w_gt(target_video_path, gt_json_path, compare_save_path, marking_clip_save_dir, padding_sec=3)

        # logging
        return log_df

    
    def _compare_to_json(self, gt_json_path, inference_json_path, extract_interval, output_path):

        # prepare
        gt_anno=AnnotationParser(gt_json_path)
        inference_anno=AnnotationParser(inference_json_path)

        gt_event_seq = gt_anno.get_event_sequence(self.extract_interval)
        inference_event_seq = inference_anno.get_event_sequence(self.extract_interval)

        # compare
        gt_event_seq = np.array(gt_event_seq)
        inference_event_seq = np.array(inference_event_seq)

        # adjust min length
        adj_len = min(len(gt_event_seq), len(inference_event_seq))
        gt_event_seq, inference_event_seq= gt_event_seq[:adj_len], inference_event_seq[:adj_len]
        
        sub_seq = gt_event_seq - inference_event_seq # [1,0,0,-1,-1,0,1, ...]
        
        # change cls
        sub_seq_enco = encode_list(sub_seq.tolist())
        results_seq_enco = []

        start_idx = 0
        end_idx = 0
        for value, enco_cls in sub_seq_enco:
            end_idx = start_idx + value
            
            if enco_cls == 0:
                results_seq_enco.append([value, self.CORRECT_CLS])
            elif enco_cls == 1:
                results_seq_enco.append([value, self.OVER_GT_CLS])
            elif enco_cls == -1: # => -1 or 2
                # check only, over infer class
                is_only_infer_case = False

                pre_idx = 0 if start_idx - 1 < 0 else start_idx - 1 # exception for overindexing
                pro_idx = adj_len - 1 if end_idx + 1 >= adj_len else end_idx # exception for overindexing
                
                pre_gt_cls, pro_gt_cls = gt_event_seq[pre_idx], gt_event_seq[pro_idx]
                pre_inf_cls, pro_inf_cls = inference_event_seq[pre_idx], inference_event_seq[pro_idx]
                
                if (pre_gt_cls == self.RS_CLS) and (pre_inf_cls == self.NRS_CLS): # FP
                    is_only_infer_case = True
                
                if (pro_gt_cls == self.RS_CLS) and (pro_inf_cls == self.NRS_CLS): # FP
                    is_only_infer_case = True
                    
                if is_only_infer_case: # 2
                    results_seq_enco.append([value, self.ONLY_INFER_CLS])
                else: # -1
                    results_seq_enco.append([value, self.OVER_INFER_CLS])                
            
            start_idx = end_idx
        
        results_seq = decode_list(results_seq_enco) # [0,1,1,0,0,1,1,1,..]

        # save to json
        # meta info
        totalFrame = gt_anno.get_totalFrame()
        frameRate = gt_anno.get_fps()
        width="temp"
        height="temp"
        name="temp"
        createdAt = "temp"
        updatedAt = "temp"
        _id = "temp"
        annotationType = "NRS"
        annotator = "temp"
        label = {str(self.CORRECT_CLS): "CORRECT",
                    str(self.OVER_GT_CLS): "OVER_GT",
                    str(self.OVER_INFER_CLS): "OVER_INFERENCE",
                    str(self.ONLY_INFER_CLS): "ONLY_INFERENCE"}

        compare_report = ReportAnnotation(output_path)
        compare_report.set_total_report(totalFrame, frameRate, width, height, _id, annotationType, createdAt, updatedAt, annotator, name, label)

        # add annotation
        results_chunk_cnt = len(results_seq_enco)
        start_frame = 0
        end_frame = 0
        for i, (value, enco_cls) in enumerate(results_seq_enco):
            end_frame = start_frame + (value * self.extract_interval) - 1
            
            if enco_cls == self.CORRECT_CLS:
                pass

            else:
                # check over totalFrame on last annotation (because of quntization? when set up extract_interval > 1)
                if i == results_chunk_cnt and end_frame >= totalFrame: 
                    end_frame = totalFrame - 1

                compare_report.add_annotation_report(start_frame, end_frame, code=enco_cls)


            start_frame = end_frame + 1

        compare_report.save_report()

    def _extract_video_from_json(self, video_path, json_path, editted_video_dir, padding_sec=3):

        print('\nvideo editing ...')

        ffmpeg_helper = ffmpegHelper(video_path, editted_video_dir)

        # prepare
        anno_parser = AnnotationParser(json_path)
        target_clipping_time = anno_parser.get_annotations_info()
        video_fps = anno_parser.get_fps() # get from annotation
        # video_fps = ffmpeg_helper.get_fps() # get from video

        print('\n\n \t\t <<<<< EXTRACTING CLIPS >>>>> \t\t \n\n')
        for i, (start_idx, end_idx, code) in enumerate(target_clipping_time, 1):

            process_dir = os.path.join(editted_video_dir, '{}'.format(code))
            
            os.makedirs(process_dir, exist_ok=True)

            mark_start_time, mark_end_time = ffmpeg_helper.idx_to_time(start_idx, video_fps), ffmpeg_helper.idx_to_time(end_idx, video_fps) # conver to time

            print('\n\n[{}] \t {} - {} - {}'.format(i, mark_start_time, mark_end_time, code))
            
            # extract_video_marking_clip(video_path, process_dir, mark_start_time, mark_end_time, padding_sec, mark=code)

            # calc paddingsec
            start_sec, end_sec = ffmpeg_helper.time_to_sec(mark_start_time) - padding_sec , ffmpeg_helper.time_to_sec(mark_end_time) + padding_sec
            start_time, end_time = ffmpeg_helper.sec_to_time(start_sec), ffmpeg_helper.sec_to_time(end_sec)

            # set save file name
            save_name = '{}-{}[{}-{}]'.format(start_time, end_time, mark_start_time, mark_end_time)

            ffmpeg_helper.set_results_dir(process_dir)
            ffmpeg_helper.extract_video_marking_clip(start_time, end_time, ffmpeg_helper.time_to_sec(mark_start_time), ffmpeg_helper.time_to_sec(mark_end_time), save_name, mark)

    def _get_gt_marking(self, gt_json_path):

        gt_mark = []
        anno_parser = AnnotationParser(gt_json_path)
        gt_anno = anno_parser.get_annotations_info()
        fps = anno_parser.get_fps()

        for start_idx, end_idx, code in gt_anno:
            gt_mark.append([
                time_to_sec(idx_to_time(start_idx, fps)),
                time_to_sec(idx_to_time(end_idx, fps)),
            ])

        return gt_mark

    def _extract_video_from_json_w_gt(self, video_path, gt_json_path, extract_json_path, editted_video_dir, padding_sec=3):

        print('\nvideo editing ...')

        ffmpeg_helper = ffmpegHelper(video_path, editted_video_dir)

        # prepare
        anno_parser = AnnotationParser(extract_json_path)
        target_clipping_time = anno_parser.get_annotations_info()
        video_fps = anno_parser.get_fps() # get from annotation
        totalFrame = anno_parser.get_totalFrame()

        # video_fps = ffmpeg_helper.get_fps() # get from video

        anno_parser.set_annotation_path(gt_json_path)
        gt_anno = anno_parser.get_annotations_info()

        gt_mark_sec = self._get_gt_marking(gt_json_path)

        # logging
        log_df = pd.DataFrame([])
        print('\n\n \t\t <<<<< EXTRACTING CLIPS >>>>> \t\t \n\n')
        for i, (mark_start_idx, mark_end_idx, code) in enumerate(target_clipping_time, 1):

            process_dir = os.path.join(editted_video_dir, '{}'.format(code))
            
            os.makedirs(process_dir, exist_ok=True)

            mark_start_time, mark_end_time = idx_to_time(mark_start_idx, video_fps), idx_to_time(mark_end_idx, video_fps) # conver to time
            mark_start_sec, mark_end_sec = time_to_sec(mark_start_time), time_to_sec(mark_end_time) # convert to sec

            # calc paddingsec
            start_sec, end_sec = mark_start_sec - padding_sec , mark_end_sec + padding_sec

            # exception
            start_sec = 0 if start_sec < 0 else start_sec

            start_time, end_time = sec_to_time(start_sec), sec_to_time(end_sec)

            # set save file name
            save_name = '{}-{}'.format(mark_start_time.replace('.', '_').replace(':', '_'), mark_end_time.replace('.', '_').replace(':', '_'))

            ffmpeg_helper.set_results_dir(process_dir)

            print('\n\n[{}] \t {} - {} - {}'.format(i, mark_start_time, mark_end_time, code))

            ffmpeg_helper.extract_video_multi_marking_clip(
                start_time=start_time,
                end_time=end_time,
                start_mark_sec=mark_start_sec,
                end_mark_sec=mark_end_sec,
                mark=code,
                save_name=save_name,
                nrs_mark_sec=gt_mark_sec,
            )

        
            # logging
            log_col = {
                'Video File Name': os.path.splitext(os.path.basename(video_path))[0],
                'Campared Class': code,
                'start': mark_start_time.replace('.', ':'),
                'end': mark_end_time.replace('.', ':'),
            }
            
            log_df = log_df.append(log_col, ignore_index=True)

        return log_df

        
        


        


            

        
        




    

        

        

