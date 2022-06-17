STAGE_LIST = ['mini_fold_stage_0', 'mini_fold_stage_1', 'mini_fold_stage_2', 'mini_fold_stage_3', 'hem_train', 'general_train']

def get_experiment_args():
    from core.config.base_opts import parse_opts
    import os

    parser = parse_opts()

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_list

    ### model basic info opts
    args.pretrained = True
    # TODO 원하는대로 변경 하기
    # 전 그냥 save path와 동일하게 가져갔습니다. (bgpark)
    args.save_path = args.save_path + '-trial:{}-fold:{}'.format(args.trial, args.fold)
    args.experiments_sheet_dir = args.save_path

    ### dataset opts
    args.data_base_path = '/raid/img_db'

    ### train args
    args.num_gpus = 1
    
    ### etc opts
    args.use_lightning_style_save = True # TO DO : use_lightning_style_save==False 일 경우 오류해결 (True일 경우 정상작동)

    return args

def inference_main(args):
    print('inference_main')
    
    ### test inference module
    from core.api.trainer import CAMIO
    from core.api.inference import InferenceDB # inference module
    from core.api.evaluation import Evaluator # evaluation module
    from core.utils.metric import MetricHelper # metric helper (for calc CR, OR, mCR, mOR)
    from core.utils.logger import Report # report helper (for experiments reuslts and inference results)

    from core.api.visualization import VisualTool # visual module

    import os
    import pandas as pd
    import glob

    # from pretrained model
    '''
    model = CAMIO(args)
    model = model.cuda()
    '''

    print('restore : ', args.restore_path)

    # from finetuning model
    model_path = get_inference_model_path(args.restore_path)
    print('model path : ', model_path)
    model = CAMIO.load_from_checkpoint(model_path, args=args)
    model = model.cuda()

    # inference block
    os.makedirs(args.restore_path, exist_ok=True)

    inference_assets_save_path = args.restore_path
    details_results_path = os.path.join(args.restore_path, 'inference_results')
    patients_results_path = os.path.join(details_results_path, 'patients_report.csv')
    videos_results_path = os.path.join(details_results_path, 'videos_report.csv')

    report_path = os.path.join(args.restore_path, 'Report.json')


    # 1. load inference dataset
    # inference case(ROBOT, LAPA), anno_ver(1,2,3), inference fold(1,2,3,4,5)에 따라 환자별 db. gt_json 잡을 수 있도록 set up
    case = args.dataset # ['ROBOT', LAPA]
    anno_ver = '3'
    inference_fold = args.inference_fold

    inference_assets = prepare_inference_aseets(case=case , anno_ver=anno_ver, inference_fold=inference_fold, save_path=inference_assets_save_path)
    patients = inference_assets['patients']
    
    patients_count = len(patients)

    # 2. for record metrics
    # Report
    report = Report(report_path)

    patients_metrics_list = [] # save each patients metrics
    for idx in range(patients_count): # per patients
        patient = patients[idx]
        patient_no = patient['patient_no']
        patient_video = patient['patient_video']

        videos_metrics_list = [] # save each videos metrics
        
        patient_gt_list = [] # for visual 
        patient_predict_list = [] # for visual

        # for save patients results
        each_patients_save_dir = os.path.join(details_results_path, patient_no)
        os.makedirs(each_patients_save_dir, exist_ok=True)

        for video_path_info in patient['path_info']: # per videos
            video_name = video_path_info['video_name']
            video_path = video_path_info['video_path']
            annotation_path = video_path_info['annotation_path']
            db_path = video_path_info['db_path']

            # Inference module
            inference = InferenceDB(model, db_path, args.inference_interval) # Inference object
            predict_list, target_img_list, target_frame_idx_list = inference.start() # call start
  
            # for save video results
            each_videos_save_dir = os.path.join(each_patients_save_dir, video_name)
            os.makedirs(each_videos_save_dir, exist_ok=True)

            # save predict list to csv
            predict_csv_path = os.path.join(each_videos_save_dir, '{}.csv'.format(video_name))
            predict_df = pd.DataFrame({
                            'frame_idx': target_frame_idx_list,
                            'predict': predict_list,
                            'target_img': target_img_list,
                        })
            predict_df.to_csv(predict_csv_path)

            # Evaluation module
            evaluator = Evaluator(predict_csv_path, annotation_path, args.inference_interval)
            gt_list, predict_list = evaluator.get_assets() # get gt_list, predict_list by inference_interval

            # save predict list to csv
            predict_csv_path = os.path.join(each_videos_save_dir, '{}.csv'.format(video_name))
            predict_df = pd.DataFrame({
                            'frame_idx': target_frame_idx_list,
                            'predict': predict_list,
                            'gt': gt_list,
                            'target_img': target_img_list,
                        })
            predict_df.to_csv(predict_csv_path)

            # for visulization per patients
            patient_gt_list += gt_list
            patient_predict_list += predict_list

            # metric per video
            video_metrics = evaluator.calc() # same return with metricHelper
            video_CR, video_OR = video_metrics['CR'], video_metrics['OR']
            video_TP, video_FP, video_TN, video_FN = video_metrics['TP'], video_metrics['FP'], video_metrics['TN'], video_metrics['FN']
            video_TOTAL = video_FP + video_TP + video_FN + video_TN

            video_gt_IB, video_gt_OOB, video_predict_IB, video_predict_OOB = video_metrics['gt_IB'], video_metrics['gt_OOB'], video_metrics['predict_IB'], video_metrics['predict_OOB']

            video_jaccard = video_metrics['Jaccard']

            print('\t => video_name: {}'.format(video_name))
            print('\t    video_CR: {:.3f} | video_OR: {:.3f}'.format(video_CR, video_OR))
            print('\t    video_TP: {} | video_FP: {} | video_TN: {} | video_FN: {}'.format(video_TP, video_FP, video_TN, video_FN))

            # save video metrics
            video_results_dict = report.add_videos_report(patient_no=patient_no, video_no=video_name, FP=video_FP, TP=video_TP, FN=video_FN, TN=video_TN, TOTAL=video_TOTAL, CR=video_CR, OR=video_OR, gt_IB=video_gt_IB, gt_OOB=video_gt_OOB, predict_IB=video_predict_IB, predict_OOB=video_predict_OOB, jaccard=video_jaccard)
            save_dict_to_csv(video_results_dict, videos_results_path)

            # for calc patients metric
            videos_metrics_list.append(video_metrics)
        
        # calc each patients CR, OR
        patient_metrics = MetricHelper().aggregate_calc_metric(videos_metrics_list)
        patient_CR, patient_OR = patient_metrics['CR'], patient_metrics['OR']
        patient_TP, patient_FP, patient_TN, patient_FN = patient_metrics['TP'], patient_metrics['FP'], patient_metrics['TN'], patient_metrics['FN']
        patient_TOTAL = patient_FP + patient_TP + patient_FN + patient_TN

        patient_gt_IB, patient_gt_OOB, patient_predict_IB, patient_predict_OOB = patient_metrics['gt_IB'], patient_metrics['gt_OOB'], patient_metrics['predict_IB'], patient_metrics['predict_OOB']

        patient_jaccard = patient_metrics['Jaccard']

        print('\t\t => patient_no: {}'.format(patient_no))
        print('\t\t    patient_CR: {:.3f} | patient_OR: {:.3f}'.format(patient_CR, patient_OR))
        print('\t\t    patient_TP: {} | patient_FP: {} | patient_TN: {} | patient_FN: {}'.format(patient_TP, patient_FP, patient_TN, patient_FN))

        # save patient metrics        
        patient_results_dict = report.add_patients_report(patient_no=patient_no, FP=patient_FP, TP=patient_TP, FN=patient_FN, TN=patient_TN, TOTAL=patient_TOTAL, CR=patient_CR, OR=patient_OR, gt_IB=patient_gt_IB, gt_OOB=patient_gt_OOB, predict_IB=patient_predict_IB, predict_OOB=patient_predict_OOB, jaccard=patient_jaccard)
        save_dict_to_csv(patient_results_dict, patients_results_path)
    
        # for calc total patients CR, OR
        patients_metrics_list.append(patient_metrics)

        # visualization per patients
        patient_predict_visual_path = os.path.join(each_patients_save_dir, 'predict-{}.png'.format(patient_no))

        visual_tool = VisualTool(patient_gt_list, patient_no, patient_predict_visual_path)
        visual_tool.visual_predict(patient_predict_list, args.model, args.inference_interval)

        # CLEAR PAGING CACHE
        clean_paging_chache()

    # for calc total patients CR, OR + (mCR, mOR)
    total_metrics = MetricHelper().aggregate_calc_metric(patients_metrics_list)
    total_mCR, total_mOR, total_CR, total_OR = total_metrics['mCR'], total_metrics['mOR'], total_metrics['CR'], total_metrics['OR']

    report.set_experiment(model=args.model, methods=args.hem_extract_mode, inference_fold=args.inference_fold, mCR=total_mCR, mOR=total_mOR, CR=total_CR, OR=total_OR, details_path=details_results_path, model_path=model_path)
    report.save_report() # save report

    # SUMMARY
    patients_CR = report.get_patients_CR()
    patients_OR = report.get_patients_CR()

    experiment_summary = {
        'model':args.model,
        'methods':args.hem_extract_mode,
        'stage': args.stage,
        'IB_ratio': args.IB_ratio,
        'inference_fold':args.inference_fold,
        'mCR':total_mCR,
        'mOR':total_mOR,
        'CR':total_CR,
        'OR':total_OR,
        'details_path':details_results_path,
        'model_path': model_path,
    }
    
    # return mCR, mOR, OR, CR of experiment
    return args, experiment_summary, patients_CR, patients_OR


def main():    
    # 0. set each experiment args 
    args = get_experiment_args()
    # 3. inference
    args, experiment_summary, patients_CR, patients_OR = inference_main(args)

    # 4. save experiments summary
    experiments_sheet_path = os.path.join(args.experiments_sheet_dir, 'experiments_summary-fold_{}.csv'.format(args.inference_fold))
    os.makedirs(args.experiments_sheet_dir, exist_ok=True)

    save_dict_to_csv({**experiment_summary, **patients_CR}, experiments_sheet_path)

if __name__ == '__main__':
    
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.abspath(__file__)))
        sys.path.append(base_path)
        print(base_path)
        
        from core.utils.misc import save_dict_to_csv, prepare_inference_aseets, get_inference_model_path, \
    set_args_per_stage, check_hem_online_mode, clean_paging_chache

    main()