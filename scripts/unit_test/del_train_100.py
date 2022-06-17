import os
import shutil

def del_frames(del_dir_path):
    if os.path.exists(del_dir_path):
        shutil.rmtree(del_dir_path)

base_path = '/data3/Public/ViHUB-pro/input_video/train_100'

fold_1_train_set = ['R_1', 'R_3', 'R_4', 'R_5', 'R_7', 'R_10', 'R_14', 'R_15', 'R_17', 'R_18', 
          'R_19', 'R_22', 'R_48', 'R_56', 'R_76', 'R_84', 'R_94', 'R_116', 'R_117', 'R_201', 
          'R_203', 'R_204', 'R_205', 'R_206', 'R_207', 'R_208', 'R_209', 'R_210', 'R_303', 'R_304', 
          'R_305', 'R_310', 'R_320', 'R_321', 'R_324', 'R_329', 'R_334', 'R_338', 'R_339', 'R_340', 
          'R_342', 'R_345', 'R_346', 'R_347', 'R_348', 'R_349', 'R_355', 'R_357', 'R_358', 'R_369', 
          'R_372', 'R_376', 'R_378', 'R_379', 'R_391', 'R_393', 'R_399', 'R_400', 'R_402', 'R_403', 
          'R_406', 'R_409', 'R_412', 'R_413', 'R_415', 'R_419', 'R_420', 'R_427', 'R_436', 'R_445', 
          'R_449', 'R_455', 'R_480', 'R_493', 'R_501', 'R_510', 'R_522', 'R_523', 'R_532', 'R_533']


for train_set in fold_1_train_set:
    target_path = os.path.join(base_path, train_set)
    print(target_path)
    del_frames(target_path)

