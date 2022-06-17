def util_test():
    print('evaluation util test')
    
    ### test inference module
    from core.utils.parser import AnnotationParser # json util
    from core.utils.parser import FileLoader # file util
    from core.utils.parser import InfoParser # parser of file nameing rule util

    import os
    import pandas as pd

    ### utils/parser.py test ###
    # FileLoader
    json_path = '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2/01_G_01_R_100_ch1_01_OOB_27.json'
    fileloader = FileLoader(json_path)

    print(fileloader.get_full_path())
    print(fileloader.get_file_name())
    print(fileloader.get_file_ext())
    print(fileloader.get_basename())
    print(fileloader.get_dirname())

    fileloader.set_file_path('/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2/01_G_01_R_100_ch1_03_OOB_27.json')
    print(fileloader.get_full_path())
    print(fileloader.get_file_name())
    print(fileloader.get_file_ext())
    print(fileloader.get_basename())
    print(fileloader.get_dirname())

    # AnnotationParser
    annoparser=AnnotationParser(json_path)
    print(annoparser.get_fps())
    print(annoparser.get_totalFrame())
    print(annoparser.get_annotations_info())
    print(len(annoparser.get_event_sequence()))

    annoparser.set_annotation_path('/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2/01_G_01_R_100_ch1_03_OOB_27.json')
    print(annoparser.get_fps())
    print(annoparser.get_totalFrame())
    print(annoparser.get_annotations_info())
    print(len(annoparser.get_event_sequence(extract_interval=30)))

    # InfoParser
    info_parser = InfoParser(parser_type='ROBOT_ANNOTATION')
    info_parser.write_file_name(json_path) # Annotation path
    info = info_parser.get_info()
    print(info)

    info_parser = InfoParser(parser_type='ROBOT_VIDEO_1')
    info_parser.write_file_name('/data3/OOB_Recog/img_db/ROBOT/R_100/01_G_01_R_100_ch1_03') # DB path or Video path 
    video_name = info_parser.get_video_name()
    info_parser.write_file_name('/data3/OOB_Recog/img_db/ROBOT/R_100/01_G_01_R_100_ch1_03') # DB path or Video path 
    patient_no = info_parser.get_patient_no()

    print(video_name)
    print(patient_no)


def main():
    print('main')
    
    ### test inference module
    from core.api.trainer import CAMIO
    from core.config.base_opts import parse_opts
    from core.api.inference import InferenceDB # inference module
    from core.api.evaluation import Evaluator # evaluation module

    import os
    import pandas as pd


    parser = parse_opts()

    # -------------- Inference Methods --------------------
    parser.add_argument('--inference_save_dir', type=str, 
                        default='../../restuls',
                        help='root directory for infernce saving')
        

    parser.add_argument('--inference_interval', type=int, 
                        default=1000,
                        help='Inference Interval of frame')

    parser.add_argument('--inference_fold',
                    default='3',
                    type=str,
                    choices=['1', '2', '3', '4', '5', 'free'],
                    help='valset 1, 2, 3, free=for setting train_videos, val_vidoes')


    args = parser.parse_args()

    # from pretrained model
    model = CAMIO(args)
    model = model.cuda()

    # from finetuning model
    '''
    model_path = '/OOB_RECOG/model_ckpt/ckpoint_0816-test-mobilenet_v3_large-model=mobilenet_v3_large-batch=32-lr=0.001-fold=1-ratio=3-epoch=24-last.ckpt'
    model = CAMIO.load_from_checkpoint(model_path, args=args)
    '''
    
    # use case 1 - init Inference
    db_path = '/data3/OOB_Recog/img_db/ROBOT/R_100/01_G_01_R_100_ch1_01' # R_100_ch1_01
    gt_json_path = '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2/01_G_01_R_100_ch1_01_OOB_27.json'
    predict_csv_path = os.path.join(args.inference_save_dir, 'R_100_ch1_01.csv')
    metric_path = os.path.join(args.inference_save_dir, 'R_100_ch1_01-metric.json')
    
    # Inference module
    inference = InferenceDB(model, db_path, args.inference_interval) # Inference object
    predict_list, target_img_list, target_frame_idx_list = Inference.start() # call start

    # save predict list to csv
    predict_df = pd.DataFrame({
                    'predict': predict_list
                })
    predict_df.to_csv(predict_csv_path)

    # Evaluation module
    evaluator = Evaluator(predict_csv_path, gt_json_path, args.inference_interval)
    metrics = evaluator.calc() # same return with metricHelper
    CR, OR = metrics['OOB_metric'], metrics['Over_estimation']
    TP, FP, TN, FN = metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']
    
    print(CR, OR)
    print(TP, FP, TN, FN)

    evaluator.set_gt_json_path('/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2/01_G_01_R_100_ch1_03_OOB_27.json') # you can also set member path
    metrics = evaluator.calc()
    CR, OR = metrics['OOB_metric'], metrics['Over_estimation']
    TP, FP, TN, FN = metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN']
    
    print(CR, OR)
    print(TP, FP, TN, FN)
    

if __name__ == '__main__':
    
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    
    if __package__ is None:
        import sys
        from os import path    
        base_path = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
        sys.path.append(base_path)
        print(base_path)

    # test()
    main()