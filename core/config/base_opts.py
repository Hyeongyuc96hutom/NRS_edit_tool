import argparse

def parse_opts():
    """
        Base arguments parser
    """
    parser = argparse.ArgumentParser()

    # --------------- Model basic info --------------------
    parser.add_argument('--model',
            default='mobilenetv3_large_100',
            type=str,
            choices=['vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn', 
                        'resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'resnext50_32x4d',
                        'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'squeezenet1_0', 'squeezenet1_1',
                        'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 
                        'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                        'ig_resnext101_32x48d', 'swin_large_patch4_window7_224', 'mobilenetv3_large_100_miil',
                        'mobilenetv3_large_100', 'tf_efficientnetv2_b0', 'tf_efficientnet_b0_ns',
                        'repvgg_b0', 'repvgg-a0', 'multi-model'],
            help='Select model to train/test')

    parser.add_argument('--pretrained',
            action='store_true', # false
            help='If true, load pretrained backbone')

    parser.add_argument('--loss_fn',
            default='ce',
            type=str,
            choices=['ce', 'focal'],
            help='Select loss_fn to train/test')

    parser.add_argument('--batch_size',
            default=256,
            type=int,
            help='Training/Testing batch size')

    parser.add_argument('--min_epoch',
            default=0,
            type=int,
            help='Minimum training epoch')

    parser.add_argument('--max_epoch',
            default=2,
            type=int,
            help='Maximum training epoch')

    parser.add_argument('--num_gpus',
            default=1,
            type=int,
            help='How many GPUs to use for training')

    parser.add_argument('--device',
            default='cuda',
            type=str,
            help='What device to use for training or validation model')

    parser.add_argument('--cuda_list',
            default='7',
            type=str,
            help='Name list of gpus that are used to train')

    parser.add_argument('--use_early_stop',
            action='store_true',
            help='If true, Ealry stopping function on')

    parser.add_argument('--early_stop_monitor',
            default='val_loss',
            type=str,
            help='select monitor value')

    parser.add_argument('--ealry_stop_mode',
            default='min',
            type=str,
            help='select monitor mode')

    parser.add_argument('--ealry_stop_patience',
            default=10,
            type=int,
            help='If monitor value is not updated until this value, stop training')

    # /OOB_RECOG/logs/project-1/TB_log/version_0
    parser.add_argument('--save_path', type=str, 
                        default='/OOB_RECOG/logs/2111119', help='')

    parser.add_argument('--test_mode',
            action='store_true',
            help='If true, Only testing')

    parser.add_argument('--resume',
            action='store_true',
            help='If true, keep training from the checkpoint')

    parser.add_argument('--restore_path', 
            type=str, 
            default=None, 
            help='Resume or test to train the model loaded from the path')

    # --------------- Optimizer and Scheduler info ----------------------
    parser.add_argument('--init_lr',
            default=1e-3,
            type=float,
            help='Initial learning rate')

    parser.add_argument('--weight_decay',
            default=1e-5,
            type=float,
            help='Weight decay value')

    parser.add_argument('--optimizer',
            default='adam',
            type=str,
            choices=['sgd', 'adam'],
            help=('What optimizer to use for training'
                '[Types : sgd, adam]'))

    parser.add_argument('--lr_scheduler',
            default='step_lr',
            type=str,
            choices=['step_lr', 'mul_lr', 'mul_step_lr', 'reduced', 'cosine'],
            help='Learning scheduler selection \n[Types : step_lr, mul_lr]')

    parser.add_argument('--lr_scheduler_step', 
            type=int, 
            default=5, 
            help='Use for Step LR Scheduler')

    parser.add_argument('--lr_scheduler_factor',
            default=0.9,
            type=float,
            help='Multiplicative factor for decreasing learning rate')

    parser.add_argument('--lr_milestones',
            default=[9, 14],
            type=list,
            help='Multi-step milestones for decreasing learning rate')
    
    parser.add_argument('--t_max_iter', 
            type=int, 
            default=200000, 
            help='Use for Step LR Scheduler')

    # -------------- Dataset --------------------
    parser.add_argument('--dataset', 
            default='ROBOT', 
            type=str, 
            choices=['ROBOT', 'LAPA'], 
            help='[robot, lapa] choice on dataset')

    parser.add_argument('--task', 
            default='OOB', 
            type=str,
            choices=['OOB', 'NIR'], 
            help='[OOB, NIR] choice on task')

    parser.add_argument('--data_base_path',
            default='/raid/img_db',
            type=str,
            help='Data location')

    parser.add_argument('--fold',
            default='1',
            type=str,
            choices=['1', '2', '3', '4', '5', 'free'],
            help='valset 1, 2, 3, 4, 5, free=for setting train_videos, val_vidoes')

    parser.add_argument('--mini_fold',
            default='1',
            type=str,
            choices=['general', '1', '2', '3', '4'],
            help='valset 1, 2, 3, 4, 5, free=for setting train_videos, val_vidoes')

    parser.add_argument('--data_version',
            default='v3',
            type=str,
            choices=['v1', 'v2', 'v3'],
            help='Annotation dataset version')

    parser.add_argument('--IB_ratio',
            default=3,
            type=float,
            help='')

    parser.add_argument('--WS_ratio',
            default=4,
            type=int,
            help='')

    parser.add_argument('--num_workers',
            default=6,
            type=int,
            help='How many CPUs to use for data loading')
    
    parser.add_argument('--use_wise_sample',
            action='store_true',
            help='If true, Only testing')

    parser.add_argument('--use_all_sample',
            action='store_true',
            help='If true, Only testing')
    
    parser.add_argument('--use_meta',
            action='store_true',
            help='If true, Only testing')
    
    parser.add_argument('--meta_sampling',
            default=6,
            help='If true, Only testing')

    parser.add_argument('--hem_per_patient',
            action='store_true',
            help='If true, Only testing')


    parser.add_argument('--theator_stage_flag',
            type=int,
            choices=[100,200,300],
            help='for theator experiments for offline methods // 100 == first stage, 200 == second stage')


    parser.add_argument('--stage_hem_path',
            default='',
            help='If true, Only testing')

    # -------------- Train Methods --------------------
    parser.add_argument('--experiment_type', 
            default='ours', 
            type=str,
            choices=['ours', 'theator'], )

    parser.add_argument('--stage', 
            default='mini_fold_stage_0', 
            type=str,
            choices=['mini_fold_stage_0', 'mini_fold_stage_1', 'mini_fold_stage_2', 'mini_fold_stage_3', 'hem_train', 'general_train'])

    parser.add_argument('--hem_extract_mode', type=str,
            default='all-offline', 
            choices=['hem-softmax-offline', 'hem-voting-offline', 'hem-vi-offline',
                     'all-offline',
                     'hem-emb-online', 'hem-focus-online',],
            help='Select train method, normal or hem method')
            
    parser.add_argument('--top_ratio', type=float,
            default=10/100,
            help='Select HEM top ratio')
    
    parser.add_argument('--sampling_type', type=int,
            default=1,
            help='?')
    
    parser.add_argument('--emb_type', type=int,
            default=1,
            help='?')
    
    parser.add_argument('--use_online_mcd',
            action='store_true',
            help='?')
    
    parser.add_argument('--use_proxy_all',
            action='store_true',
            help='?')
    
    parser.add_argument('--use_half_neg',
            action='store_true',
            help='?')
    
    parser.add_argument('--use_neg_proxy',
            action='store_true',
            help='?')
    
    parser.add_argument('--dropout_prob',
            default=0.3,
            type=float,
            help='?')
    
    parser.add_argument('--n_dropout',
            default=1,
            type=int,
            help='?')
    
    parser.add_argument('--use_loss_weight',
            action='store_true',
            help='?')
    
    parser.add_argument('--alpha',
            default=0.1,
            type=float,
            help='?')
    
    parser.add_argument('--use_multi_model',
            action='store_true',
            help='?')
    
    parser.add_argument('--multi_stage',
            action='store_true',
            help='?')
    
    parser.add_argument('--n_stage',
            default=5,
            type=int,
            help='?')
    
    parser.add_argument('--use_emb_only',
            action='store_true',
            help='?')

    # -------------- etc --------------------
    parser.add_argument('--random_seed', type=int, default=10, help='dataset random seed')

    parser.add_argument('--use_lightning_style_save', action='store_true', help='If true, use lightning save module')

    parser.add_argument('--save_top_n', type=int, default=1, help='dataset random seed')
    
    parser.add_argument('--inference_interval', type=int, 
                        default=300,
                        help='Inference Interval of frame')

    parser.add_argument('--inference_fold',
                    default='1',
                    type=str,
                    choices=['1', '2', '3', '4', '5', 'free'],
                    help='valset 1, 2, 3, 4, 5, free')

    parser.add_argument('--experiments_sheet_dir', type=str, 
                        default='/code/OOB_Recog/template/results',
                        help='root directory for experimets results')
    
    parser.add_argument('--trial', type=int,
            default=1,
            help='?')
    
    
    parser.add_argument('--use_test_batch',
            action='store_true',
            help='?')

    parser.add_argument('--apply_mode',
            action='store_true',
            help='?')

    return parser
