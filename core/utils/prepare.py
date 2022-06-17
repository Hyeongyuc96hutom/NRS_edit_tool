import core.config.patients_info as patients_info
import core.config.assets_info as assets_info
from core.utils.parser import FileLoader, InfoParser

from core.utils.misc import prepare_inference_aseets
from core.utils.parser import AnnotationParser

import os
import glob
import yaml
import re
import natsort

class InferenceAssets():
    """
        for prepare inference assets
        case; ['ROBOT', 'LAPA']
        anno_ver; ['V1', 'V2', 'V3']
        fold; ['1', '2', '3', '4', '5', 'all', 'free']

        # inference_assets
        patients: 
            - patient_no: 'R_210'
            patient_video: ['R_210_ch1_03', 'R_210_ch2_04']
            path_info:
                - video_name : 'R_210_ch1_03'
                    video_path : '/VIDEO_PATH'
                    annotation_path : '/ANNOTATION_PATH'
                    db_path : '/DB_PATH'
                - video_name : 'R_210_ch2_04'
                    video_path : 'VIDEO_PATH'
                    annotation_path : '/ANNOTATION_PATH'
                    db_path : '/DB_PATH'

        # use example
        inference_assets_helper = InferenceAssets(case='ROBOT', anno_ver='V3', fold='1')
        inference_assets = inference_assets_helper.get_inference_assets() # dict (yaml)
        patients_count = inference_assets_helper.get_patients_cnt()
    """
    def __init__(self, case, anno_ver, fold):
        self.case = case
        self.anno_ver = anno_ver
        self.fold = fold

    def _support_sanity_check(self):
        sanity = False
        if self.case in ['ROBOT']:
            if self.anno_ver in ['3']:
                if self.fold in ['1','2','3','4','5', 'all', 'free']: 
                    sanity = True
        
        return sanity        

    # return elements of target_list which has parser_str, if not return []
    def _return_elements_is_str_in_list(self, parser_str, target_list) :
        find_elements = [] # Non
        
        for idx, target in enumerate(target_list) :
            if parser_str in target :
                find_elements.append(target)

        return find_elements

    
    def _pateint_video_sort(self, patient_video_list) :
        # sort [R_999_ch1_03, R_999_ch2_01, R_999_ch2_02] => [R_999_ch2_01, R_999_ch2_02, R_999_ch1_03]
        # sort [01_G_01_R_999_ch1_03.mp4, 01_G_01_R_999_ch2_01.mp4, 01_G_01_R_999_ch2_02.mp4] => [01_G_01_R_999_ch2_01.mp4, 01_G_01_R_999_ch2_02.mp4, 01_G_01_R_999_ch1_03.mp4]
        sorted_list = natsort.natsorted(patient_video_list, key=lambda x : os.path.splitext(x)[0].split('_')[-1], alg=natsort.ns.INT)        
        return sorted_list
    
    def _parsing_patient_video(self, patient_list, use_video_list):
        print('\n{}'.format('=====' * 10))
        print('\t ===== GET PATIENT VIDEO ===== \t')
        print('{}\n'.format('=====' * 10))

        # set USE VIDEO LIST
        use_videos = use_video_list.copy()

        patient_video_dict = {}

        # make patients video dict
        for patient in patient_list :
            pateint_parser = patient + '_' # R_1 => R_1_ | R_10 => R_10_
            patient_video = self._return_elements_is_str_in_list(pateint_parser, use_videos)

            patient_video_dict[patient] = patient_video

        # print('\n----- RESULTS [PATIENT CNT : {} ] ------\n'.format(len(patient_video_dict)))
        # print('----- PATIENT : {} ------\n'.format(patient_list))
        # for keys, value in patient_video_dict.items() : 
        #     print('{} | {}'.format(keys, value))

        return patient_video_dict

    def _get_patient_video(self, case, fold):
        patient_video_dict = {}

        if case == 'ROBOT':
            patient_video_dict = self._parsing_patient_video(patients_info.val_videos[fold], patients_info.video_details['robot'])
        elif case == 'LAPA':
            pass

        return patient_video_dict        
    
    def _get_assets_sheet(self, case, anno_ver):
        assets_sheet_dir = 'dummy'
        oob_assets = OOBAssets(assets_sheet_dir, case, anno_ver)
        video_sheet, annotation_sheet, img_db_sheet = oob_assets.get_assets_sheet()

        return video_sheet, annotation_sheet, img_db_sheet

    def save_dict_to_yaml(self, save_dict, save_path): # dictonary, ~.yaml
        with open(save_path, 'w') as f :
            yaml.dump(save_dict, f)

    def print_inference_assets(self, inference_assets_yaml_path): 
        print('\n\n\n\t\t\t ### [INFERENCE ASSETS INFO] ### \n')

        f_loader = FileLoader()

        f_loader.set_file_path(inference_assets_yaml_path)
        inference_assets = f_loader.load()

        patients = inference_assets['patients']
        patient_count = len(patients)

        # print('PATIENT COUNT : {}'.format(patient_count))
        # print('PATIENT LIST')
        # for idx in range(patient_count) : 
        #     print('-', patients[idx]['patient_no'])
        # print('\n\n\t\t\t ### ### ### ### ### ### \n\n')
        
        for idx in range(patient_count) : 
            patient = patients[idx]

            # print('PATIENT_NO : \t\t{}'.format(patient['patient_no']))
            # print('PATIENT_VIDEO : \t{}\n'.format(patient['patient_video']))
            
            # for video_path_info in patient['path_info'] :
            #     print('VIDEO_NAME : \t\t{}'.format(video_path_info['video_name']))
            #     print('VIDEO_PATH : \t\t{}'.format(video_path_info['video_path']))
            #     print('ANNOTATION_PATH : \t{}'.format(video_path_info['annotation_path']))
            #     print('DB_PATH : \t\t{}'.format(video_path_info['db_path']))
            #     print('\n', '-----'*10, '\n')

            print('\n', '=== === === === ==='*5, '\n')
        
    def get_inference_assets(self):
        assert self._support_sanity_check(), 'NOT SUPPORTED INFERENCE ASSETS'
        # 0. prepare assets
        video_sheet, annotation_sheet, img_db_sheet = self._get_assets_sheet(self.case, self.anno_ver)

        # 1. set patients video
        patient_video_dict = self._get_patient_video(self.case, self.fold)

        # 2. patient video sorting
        for patient, video_name_list in patient_video_dict.items() : 
            patient_video_dict[patient] = self._pateint_video_sort(video_name_list)

        # 3. aggregtation dict for yaml assets
        patients = {'patients': []}

        # add 'patinet' obj
        for patient, video_name_list in patient_video_dict.items() : 

            path_info = []

            # add 'info' obj
            for video_name in video_name_list: 
                path_info.append({
                    'video_name': video_name,
                    'video_path': video_sheet[video_name],
                    'annotation_path': annotation_sheet[video_name],
                    'db_path': img_db_sheet[video_name]
                })

            patients['patients'].append({
                'patient_no': patient,
                'patient_video': video_name_list,
                'path_info': path_info
            })
        '''
        # 4. serialization from python object to YAML stream and save
        with open(save_path, 'w') as f :
            yaml.dump(patients, f)
        '''

        return patients

class OOBAssets():
    """
        aggreagte total assets in using OOB project
    """
    def __init__(self, assets_sheet_dir, case='ROBOT', anno_ver='3'):
        self.assets_sheet_dir = assets_sheet_dir
        self.case = case
        self.anno_ver = anno_ver

        self.sheet_name_rule = {
            'video_sheet': 'VIDEO_PATH_SHEET.yaml',
            'annotation_sheet': 'ANNOTATION_PATH_SHEET.yaml',
            'img_db_sheet': 'DB_PATH_SHEET.yaml',
        }

    def _return_idx_is_str_in_list(self, parser_str, target_list) :
        # return index of target_list which has parser_str, if not return -1
        find_idx = -1 # EXCEPTION NUM
        
        for idx, target in enumerate(target_list) :
            if parser_str in target :
                find_idx = idx
                break

        return find_idx

    def _save_dict_to_yaml(self, save_dict, save_path): # dictonary, ~.yaml
        with open(save_path, 'w') as f :
            yaml.dump(save_dict, f)

    def _get_video_details(self):
        if self.case == 'ROBOT':
            return patients_info.video_details['robot_40_case'], patients_info.video_details['robot_60_case']
        elif self.case == 'LAPA':
            pass

    def _get_annotation_root_path(self):
        if self.anno_ver == '1':  
            return assets_info.annotation_path['annotation_v1_base_path']
        elif self.anno_ver == '2':
            return assets_info.annotation_path['annotation_v2_base_path']
        elif self.anno_ver == '3':
            return assets_info.annotation_path['annotation_v3_base_path']

    def _get_img_db_root_path(self):
        if self.case == 'ROBOT':
            return assets_info.img_db_path['robot']
        elif self.case == 'LAPA':
            pass

    def _get_video_root_path(self):
        if self.case == 'ROBOT':
            return assets_info.video_path['robot']
        if self.case == 'LAPA':
            pass

    def _get_video_sheet_for_robot(self, video_root_path, use_video_list):

        print('\n{}'.format('=====' * 10))
        print('\t ===== GET VIDEO PATH FOR ROBOT =====')
        print('{}\n'.format('=====' * 10))
        

        # DATA_PATH_DICT
        video_path_dict = {}

        # set for video use list, should copy because of list remove
        use_videos = use_video_list.copy()
        
        # video_root_path/R000001/ch2_video_02.mp4 ==> only parsing this rule | RXXXXXX/\w{3}_video_\d+[.]MP4, mp4
        base_video_path = glob.glob(os.path.join(video_root_path, '*', '*')) # total path
        
        parser_regex = re.compile(r'[R]\d+/\w{3}_video_\d+[.]mp4$|[R]\d+/\w{3}_video_\d+[.]MP4$') # R000001/ch2_video_02.mp4
        pattern = re.compile(parser_regex)
        
        cnt = 0 # PROCESSED CNT
        
        # parsering with regex
        for path in base_video_path : 
            match = pattern.search(path)
            if match : # MATCHED
                # print('Matching \t', match.group())
                info_parser = InfoParser(parser_type='ROBOT_VIDEO_1')
                info_parser.write_file_name(path)
                video_name = info_parser.get_video_name()

                if video_name in use_videos : # Only Processing in USE VIDEO LIST
                    # print('USE VIDEO : {} \n'.format(video_name))
                    video_path_dict[video_name] = path # add dict | video_name : video_path
                    use_videos.remove(video_name)
                    cnt+=1

            else : # UNMATCHED
                pass
                # print('NON Matching \t', path)


        # print('\n----- PROCESSED DATA CNT : {} | FAILED PROCESSED DATA CNT : {} ------\n'.format(cnt, len(use_videos)))
        # print('FALIED DATA LIST : {}'.format(use_videos))

        ### ADD EXCEPTION FILE
        EXCEPTION_RULE = {'R_76_ch1_01' : os.path.join(video_root_path, 'R000076', 'ch1_video_01_6915320_RDG.mp4'),
                        'R_84_ch1_01' : os.path.join(video_root_path, 'R000084', 'ch1_video_01_8459178_robotic\ subtotal.mp4'), 
                        'R_391_ch2_06' : os.path.join(video_root_path, 'R000391', '01_G_01_R_391_ch2_06.mp4')}

        # print('\n--REGISTERD EXCEPTION DATA RULE --\n')
        # for keys, value in EXCEPTION_RULE.items() : 
        #     print('{} | {}'.format(keys, value))

        print('\n ==> APPLY EXEPTION RULE')
        
        # apply EXCEPTION RULE
        for video_name in use_videos :
            video_path_dict[video_name] = EXCEPTION_RULE.get(video_name, '') # if Non key, return ''
            # print('{} | {}'.format(video_name, video_path_dict[video_name]))

        # print('\n----- RESULTS [VIDEO_PATH_DICT CNT : {} ] ------\n'.format(len(video_path_dict)))
        # for keys, value in video_path_dict.items() : 
            # print('{} | {}'.format(keys, value))

        return video_path_dict

    def _get_annotation_sheet(self, annotation_root_path, use_video_list):

        print('\n{}'.format('=====' * 10))
        print('\t ===== GET ANNOTATION PATH =====')
        print('{}\n'.format('=====' * 10))

        # DATA_PATH_DICT
        anno_path_dict = {}

        # set USE VIDEO LIST
        use_videos = use_video_list.copy()
        
        # parsing all annotation path
        all_anno_path = []
        anno_ext_list = ['json']

        for ext in anno_ext_list :
            all_anno_path.extend(glob.glob(os.path.join(annotation_root_path, '*.{}'.format(ext))))

        cnt = 0 # PROCESSED CNT

        # check which idx is included parser_str in all_anno_path
        for video_name in use_videos :
            idx = self._return_idx_is_str_in_list(video_name, all_anno_path)
            
            if idx == -1 :
                anno_path_dict[video_name] = ''
            else : 
                cnt += 1
                anno_path_dict[video_name] = all_anno_path[idx]

        print('\n----- PROCESSED DATA CNT : {} | FAILED PROCESSED DATA CNT : {} ------\n'.format(cnt, len(use_videos)-cnt))
        # print('FALIED DATA LIST : {}'.format(use_videos))
        
        # print('\n----- RESULTS [ANNO_PATH_DICT CNT : {} ] ------\n'.format(len(anno_path_dict)))
        # for keys, value in anno_path_dict.items() : 
        #     print('{} | {}'.format(keys, value))

        return anno_path_dict

    def _get_img_db_sheet(self, img_db_root_path, use_video_list):

        print('\n{}'.format('=====' * 10))
        print('\t ===== GET DB PATH =====')
        print('{}\n'.format('=====' * 10))

        # DATA_PATH_DICT
        db_path_dict = {}

        # set USE VIDEO LIST
        use_videos = use_video_list.copy()

        # parsing only DIRECTORY in DB root path
        all_DB_path = [path for path in glob.glob(os.path.join(img_db_root_path, '*', '*'), recursive=False) if os.path.isdir(path)] # img_db_root_path / Patient / Video_name / Video_name-000000001.jpg

        cnt = 0 # PROCESSED CNT

        # check which idx is included parser_str in all_anno_path
        for video_name in use_videos :
            idx = self._return_idx_is_str_in_list(video_name, all_DB_path)
            
            if idx == -1 :
                db_path_dict[video_name] = ''
            else : 
                cnt += 1
                db_path_dict[video_name] = all_DB_path[idx]

        print('\n----- PROCESSED DATA CNT : {} | FAILED PROCESSED DATA CNT : {} ------\n'.format(cnt, len(use_videos)-cnt))
        
        # print('\n----- RESULTS [ANNO_PATH_DICT CNT : {} ] ------\n'.format(len(db_path_dict)))
        # for keys, value in db_path_dict.items() : 
        #     print('{} | {}'.format(keys, value))

        return db_path_dict

    def set_assets_sheet_dir(self, assets_sheet_dir):
        self.assets_sheet_dir = assets_sheet_dir

    def get_assets_sheet(self): # only get assets sheet, if you want to save, using make_assets_sheet()
        video_sheet = {}
        annotation_sheet = {}
        img_db_sheet = {}

        if self.case == 'ROBOT':
            robot_40_case_root_path, robot_60_case_root_path = self._get_video_root_path()
            annotation_root_path = self._get_annotation_root_path()
            img_db_root_path = self._get_img_db_root_path()
            robot_40_case_video_details, robot_60_case_video_details  = self._get_video_details()
            video_details = robot_40_case_video_details + robot_60_case_video_details

            video_sheet = {**self._get_video_sheet_for_robot(robot_40_case_root_path, robot_40_case_video_details), **self._get_video_sheet_for_robot(robot_60_case_root_path, robot_60_case_video_details)}
            annotation_sheet = self._get_annotation_sheet(annotation_root_path, video_details)
            img_db_sheet = self._get_img_db_sheet(img_db_root_path, video_details)


        elif self.case == 'LAPA':
            pass

        return video_sheet, annotation_sheet, img_db_sheet

    def save_assets_sheet(self): # get assets sheet and save to assets_sheet_dir
        video_sheet, annotation_sheet, img_db_sheet = self.get_assets_sheet()

        # save assets
        os.makedirs(self.assets_sheet_dir, exist_ok=True) 

        video_sheet_save_path = os.path.join(self.assets_sheet_dir, self.sheet_name_rule['video_sheet'])
        annotation_sheet_save_path = os.path.join(self.assets_sheet_dir, self.sheet_name_rule['annotation_sheet'])
        img_db_sheet_save_path = os.path.join(self.assets_sheet_dir, self.sheet_name_rule['img_db_sheet'])

        self._save_dict_to_yaml(video_sheet, video_sheet_save_path)
        self._save_dict_to_yaml(annotation_sheet, annotation_sheet_save_path)
        self._save_dict_to_yaml(img_db_sheet, img_db_sheet_save_path)
        
    def load_assets_sheet(self, assets_sheet_dir): # when you use custimized datasheet
        # 1. set load path from data_sheet_dir
        video_sheet_yaml_path = os.path.join(assets_sheet_dir, self.sheet_name_rule['video_sheet'])
        annotation_sheet_yaml_path = os.path.join(assets_sheet_dir, self.sheet_name_rule['annotation_sheet'])
        img_db_sheet_yaml_path = os.path.join(assets_sheet_dir, self.sheet_name_rule['img_db_sheet'])

        # 2. load from yaml to dict
        f_loader = FileLoader()

        f_loader.set_file_path(video_sheet_yaml_path)
        video_sheet = f_loader.load()

        f_loader.set_file_path(annotation_sheet_yaml_path)
        annotation_sheet = f_loader.load()

        f_loader.set_file_path(img_db_sheet_yaml_path)
        img_db_sheet = f_loader.load()

        return video_sheet, annotation_sheet, img_db_sheet

class PatientsGT():
    def __init__(self):
        
        self.patients_assets = {}
        '''
            self.patients_assets = {'R_100': [R_100_1.json, R_100_2.json ..], 'R_10':[R_10_1.json, ...]}
        '''

        # 1. load inference dataset
        inference_assets = prepare_inference_aseets(case='ROBOT' , anno_ver='3', inference_fold='all', save_path='/OOB_RECOG/results')
        patients = inference_assets['patients']
        patients_count = len(patients)

        # 2. set self.patints_assets
        for idx in range(patients_count): # per patients
            patient = patients[idx]
            patient_no = patient['patient_no']
            
            self.patients_assets[patient_no] = []

            for path_info in patient['path_info']: # get total annotation path
                self.patients_assets[patient_no].append(path_info['annotation_path'])

    def get_gt(self, patient_no):
        patient_gt = []
        annotation_path_list = self.patients_assets[patient_no]
    
        anno_parser = AnnotationParser(annotation_path_list[0]) # init annoParser

        for annotation_path in annotation_path_list:
            anno_parser.set_annotation_path(annotation_path)
            patient_gt += anno_parser.get_event_sequence(extract_interval=1)
        
        return patient_gt
    
    def get_video_no(self, patient_no):
        video_no = []
        annotation_path_list = self.patients_assets[patient_no]

        info_parser = InfoParser(parser_type='ROBOT_ANNOTATION')

        for annotation_path in annotation_path_list:    
            info_parser.write_file_name(annotation_path)
            video_no.append(info_parser.get_video_name())

        return video_no # already sorted

    def get_start_idx(self, patient_no):
        video_start_idx = []
        annotation_path_list = self.patients_assets[patient_no]

        anno_parser = AnnotationParser(annotation_path_list[0]) # init annoParser

        patinet_len = 0
        for annotation_path in annotation_path_list:
            anno_parser.set_annotation_path(annotation_path)
            
            video_start_idx.append(patinet_len)

            # [no.1 video] totalFrame:3000 => frame index:0~2999 // [no.2 video] start idx:3000
            patinet_len += anno_parser.get_totalFrame()
            
        return video_start_idx



        
        
