def logging_test():
    print('logging_test')
    
    ### test logging module
    from core.config.base_opts import parse_opts
    from core.utils.logging import LogHelper # Logging module

    import os

    parser = parse_opts()

    # -------------- Inference Methods --------------------
    parser.add_argument('--inference_save_dir', type=str, 
                        default='../../restuls',
                        help='root directory for infernce saving')
        

    parser.add_argument('--inference_interval', type=int, 
                        default=30,
                        help='Interval of inference frame')

    parser.add_argument('--inference_fold',
                    default='3',
                    type=str,
                    choices=['1', '2', '3', '4', '5', 'free'],
                    help='valset 1, 2, 3, free=for setting train_videos, val_vidoes')

    args = parser.parse_args()

    # inference logging
    train_model_info = '{}-{}-{}-{}-{}'.format(args.model, args.dataset, args.task, args.fold, args.train_method)
    inference_info = 'inference_fold-{}'.format(args.inference_fold)

    inference_save_path = os.path.join(args.inference_save_dir, train_model_info, inference_info)
    os.makedirs(inference_save_path, exist_ok=True)

    log_helper = LogHelper(os.path.join(inference_save_path, 'log.txt')) # logging
    log_helper.writeln('\t === START INFERENCE === \t')
    log_helper.writeln()


    # INFERENCEING ... 


    log_helper.writeln('\t === END INFERENCE === \t')
    log_helper.writeln()

def report_helper_test():
    print('report_helper_test')

    ### test report_helper module
    from core.utils.logging import ReportHelper # report helper (for experiments reuslts and inference results)

    ### ReportHelper Test
    report_helper = ReportHelper(report_save_path='../results/report.csv', report_type='inference')
    report_col = report_helper.get_report_form()

    report_col['Patient'] = 'R_10'
    report_col['FP'] = 120
    report_col['TP'] = 130
    report_col['FN'] = 140
    report_col['TN'] = 150
    report_col['TOTAL'] = 160
    report_col['GT_OOB'] = 170
    report_col['GT_IB'] = 180
    report_col['PREDICT_OOB'] = 190
    report_col['PREDICT_IB'] = 200
    report_col['CR'] = 21.10
    report_col['OR'] = 22.20

    report_helper.save_report(report_col)


if __name__ == '__main__':
    
    if __package__ is None:
        import sys
        from os import path
        sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
    
    # logging_test()
    report_helper_test()