import glob
import os
import pandas as pd
from itertools import groupby
from tqdm import tqdm

from PIL import Image, ImageOps
from IPython.display import Image as Img
from IPython.display import display


def decode_list(run_length): # run_length -> [0,1,1,1,1,0 ...]
    decode_list = []

    for length, group in run_length : 
        decode_list += [group] * length

    return decode_list

def encode_list(s_list): # run-length encoding from list
    return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]

def generate_gif(pre_pad_img, section_img, pro_pad_img, output_path, disp_color=['black', 'yellow', 'black']):

    pre_color, section_color, pro_color = disp_color

    pre_pad_images = [ImageOps.expand(Image.open(x), border=0, fill=pre_color) for x in pre_pad_img]
    section_images = [ImageOps.expand(Image.open(x), border=0, fill=section_color) for x in section_img]
    pro_pad_images = [ImageOps.expand(Image.open(x), border=0, fill=pro_color) for x in pro_pad_img]

    images = pre_pad_images + section_images + pro_pad_images

    os.makedirs(output_path, exist_ok=True)

    for i in range(len(images)):
        images[i].save(os.path.join(output_path, '{}.jpg').format(i))

    # iamges = ImageOps.expand(Image.open('original-image.png'),border=300,fill='black').save('imaged-with-border.png')
    
    # im = images[0]
    # im.save(output_path, save_all=True, append_images=images[1:],loop=1, duration=500)
    # loop 반복 횟수
    # duration 프레임 전환 속도 (500 = 0.5초)

def extract_fpfn():

    ## --- robot ---
    fold1 = ['R_2', 'R_6', 'R_13', 'R_74', 'R_100', 'R_202', 'R_301', 'R_302', 'R_311', 'R_312', 
          'R_313', 'R_336', 'R_362', 'R_363', 'R_386', 'R_405', 'R_418', 'R_423', 'R_424', 'R_526']
    # fold1 = ['R_301']
    img_path = '/data3/NRSdata/NRS/robot/mola/img'
    json_path = '/data3/NRSdata/NRS/robot/mola/anno/v3'
    


    ## --- robot_etc ---
    '''
    fold1 = ['R_6', 'R_46', 'R_154', 'R_155', 'R_156', 'R_157', 'R_158','R_160', 'R_161', 'R_162', 'R_163', 'R_164', 'R_165','R_166' , 'R_167', 'R_168', 'R_169', 'R_170', 'R_171', 'R_172']
    img_path = '/data3/NRSdata/NRS/robot/etc/img'
    json_path = '/data3/NRSdata/NRS/robot/etc/anno/v3'
    '''


    ## --- lapa ---
    '''
    fold1 = ['01_VIHUB1.2_A9_L_5', '01_VIHUB1.2_A9_L_6', '01_VIHUB1.2_A9_L_18', '01_VIHUB1.2_A9_L_19', '01_VIHUB1.2_A9_L_20', \
            '01_VIHUB1.2_A9_L_21', '01_VIHUB1.2_A9_L_24', '01_VIHUB1.2_A9_L_30', '01_VIHUB1.2_A9_L_35', \
            '01_VIHUB1.2_A9_L_38', '01_VIHUB1.2_A9_L_40', '01_VIHUB1.2_A9_L_47', '01_VIHUB1.2_A9_L_48', '01_VIHUB1.2_A9_L_51', \
            '01_VIHUB1.2_B4_L_1', '01_VIHUB1.2_B4_L_2', '01_VIHUB1.2_B4_L_4', '01_VIHUB1.2_B4_L_6', '01_VIHUB1.2_B4_L_7', \
            '01_VIHUB1.2_B4_L_8', '01_VIHUB1.2_B4_L_9', '01_VIHUB1.2_B4_L_12', '01_VIHUB1.2_B4_L_16', '01_VIHUB1.2_B4_L_17', \
            '01_VIHUB1.2_B4_L_20', '01_VIHUB1.2_B4_L_24', '01_VIHUB1.2_B4_L_26', '01_VIHUB1.2_B4_L_28', \
            '01_VIHUB1.2_B4_L_29', '01_VIHUB1.2_B4_L_75', '01_VIHUB1.2_B4_L_82', '01_VIHUB1.2_B4_L_84', \
            '01_VIHUB1.2_B4_L_86', '01_VIHUB1.2_B4_L_87', '01_VIHUB1.2_B4_L_90', '01_VIHUB1.2_B4_L_91', '01_VIHUB1.2_B4_L_94', \
            '01_VIHUB1.2_B4_L_98', '01_VIHUB1.2_B4_L_100', '01_VIHUB1.2_B4_L_103', '01_VIHUB1.2_B4_L_106', '01_VIHUB1.2_B4_L_107', \
            '01_VIHUB1.2_B4_L_108', '01_VIHUB1.2_B4_L_111', '01_VIHUB1.2_B4_L_113', '01_VIHUB1.2_B4_L_115', '01_VIHUB1.2_B4_L_120', \
            '01_VIHUB1.2_B4_L_121', '01_VIHUB1.2_B4_L_123', '01_VIHUB1.2_B4_L_127', '01_VIHUB1.2_B4_L_130', '01_VIHUB1.2_B4_L_131', \
            '01_VIHUB1.2_B4_L_134', '01_VIHUB1.2_B4_L_139', '01_VIHUB1.2_B4_L_143', \
            '01_VIHUB1.2_B4_L_144', '01_VIHUB1.2_B4_L_146', '01_VIHUB1.2_B4_L_149', '01_VIHUB1.2_B4_L_150', '01_VIHUB1.2_B4_L_151', \
            '01_VIHUB1.2_B4_L_152', '01_VIHUB1.2_B4_L_153', '01_VIHUB1.2_B5_L_9', '04_GS4_99_L_4', '04_GS4_99_L_7', \
            '04_GS4_99_L_11', '04_GS4_99_L_12', '04_GS4_99_L_16', '04_GS4_99_L_17', '04_GS4_99_L_26', '04_GS4_99_L_28', '04_GS4_99_L_29', \
            '04_GS4_99_L_37', '04_GS4_99_L_38', '04_GS4_99_L_39', '04_GS4_99_L_40', '04_GS4_99_L_42', '04_GS4_99_L_44', '04_GS4_99_L_46', \
            '04_GS4_99_L_48', '04_GS4_99_L_49', '04_GS4_99_L_50', '04_GS4_99_L_58', '04_GS4_99_L_59', '04_GS4_99_L_60', '04_GS4_99_L_61', \
            '04_GS4_99_L_64', '04_GS4_99_L_65', '04_GS4_99_L_71', '04_GS4_99_L_75', '04_GS4_99_L_79', \
            '04_GS4_99_L_84', '04_GS4_99_L_86', '04_GS4_99_L_87', '04_GS4_99_L_88', '04_GS4_99_L_89', '04_GS4_99_L_92', \
            '04_GS4_99_L_94', '04_GS4_99_L_95', '04_GS4_99_L_96', '04_GS4_99_L_99', '04_GS4_99_L_102', '04_GS4_99_L_103', \
            '04_GS4_99_L_104', '04_GS4_99_L_106', '04_GS4_99_L_107', '04_GS4_99_L_108', '04_GS4_99_L_114', '04_GS4_99_L_116',\
            '04_GS4_99_L_120', '04_GS4_99_L_126']

    img_path = '/data3/NRSdata/NRS/lapa/vihub/img'
    json_path = '/data3/NRSdata/NRS/lapa/vihub/anno/v3'
    '''

    assets_dir = '/NRS_EDIT/2022_miccai_rebuttal_assets/robot_20/inference_result_5fps'

    # '/raid/NRS/robot/mola/img/R_13/01_G_01_R_13_ch1_01/0000132690.jpg' => /data3/NRSdata/NRS/robot/mola/~

    base_dir = '/data3/NRSdata/NRS'
    save_root_dir = '/NRS_EDIT/resultsFNFP_5fps_new_new'
    

    target_csv_path = glob.glob(os.path.join(assets_dir, '*', '*', '*-gt.csv')) # // 220511-RS / R_100 / ~.csv

    RS_CLS, NRS_CLS = (0,1)
    FN_CLS, FP_CLS = (1, -1)
    PAD_COLOR, FN_COLOR, FP_COLOR = ('black', 'blue', 'red')
    FN_save_path, FP_save_path = ('FN', 'FP')

    for ids, t_csv in enumerate(target_csv_path):
        csv_df = pd.read_csv(t_csv)
        save_dir = os.path.splitext(os.path.join(save_root_dir, '/'.join(t_csv.split('/')[3:])))[0]
        
        
        print('[+] MAKE GIF (FP, FN) - {}'.format(ids))
        print('t_csv: \t\t', t_csv)
        print('save_dir: \t', save_dir)
        
        # change img dir
        new_path = []
        for old_path in csv_df['target_img'].values.tolist():
            # new_path.append(os.path.join(base_dir, '/'.join(old_path.split('/')[3:]))) # robot, lapa
            # new_path.append(os.path.join(base_dir, '/'.join(old_path.split('/')[5:]))) # etc

            new_path.append(os.path.join(img_path, '/'.join(old_path.split('/')[6:]))) # robot, lapa
            # new_path.append(os.path.join(img_path, '/'.join(old_path.split('/')[5:]))) # etc
            
        csv_df['target_img'] = new_path

        # find FP, FN
        csv_df['gt-pred'] = csv_df['gt'] - csv_df['predict']
        diff_vec = csv_df['gt-pred'].values.tolist()
        encode_list(diff_vec) # [[length, value], [length, value]...]

        # gif
        total_len = len(csv_df)
        pad = 30 # 3ea = 3s?

        s_idx = 0
        for i, (length, diff_cls) in enumerate(encode_list(diff_vec)): # [[length, value], [length, value]...]
            e_idx = s_idx + length - 1
            
            if diff_cls in [FN_CLS, FP_CLS] :
                
                target_s_idx = s_idx - pad
                target_e_idx = s_idx + pad

                target_s_idx = 0 if target_s_idx < 0 else target_s_idx
                target_e_idx = total_len if target_e_idx > total_len else target_e_idx
                    
                pre_pad_df = csv_df.iloc[target_s_idx : s_idx]
                section_df = csv_df.iloc[s_idx : e_idx + 1] # if totlaFrmae=10, iloc[9:11] ==> no exception (cuz dataframe)
                pro_pad_df = csv_df.iloc[e_idx + 1 : target_e_idx]

                pre_pad_img = pre_pad_df['target_img'].values.tolist() 
                target_img = section_df['target_img'].values.tolist()
                pro_pad_img = pro_pad_df['target_img'].values.tolist()

                # for save
                target_s_frame = csv_df.iloc[target_s_idx].frame_idx
                s_frame = csv_df.iloc[s_idx].frame_idx
                e_frame = csv_df.iloc[e_idx].frame_idx
                target_e_frame = csv_df.iloc[target_e_idx - 1].frame_idx

                # print(total_len)
                # print(target_s_idx, s_idx, e_idx, target_e_idx-1)
                # print(target_s_frame, s_frame, e_frame, target_e_frame)

                if diff_cls == FN_CLS:
                    os.makedirs(os.path.join(save_dir, 'FN'), exist_ok=True)
                    output_path = os.path.join(save_dir, 'FN', '{}-{}-{}-{}'.format(target_s_frame, s_frame, e_frame, target_e_frame))
                    disp_color = [PAD_COLOR, FN_COLOR, PAD_COLOR]
                
                if diff_cls == FP_CLS:
                    os.makedirs(os.path.join(save_dir, 'FP'), exist_ok=True)
                    output_path = os.path.join(save_dir, 'FP', '{}-{}-{}-{}'.format(target_s_frame, s_frame, e_frame, target_e_frame))
                    disp_color = [PAD_COLOR, FP_COLOR, PAD_COLOR]

                generate_gif(pre_pad_img, target_img, pro_pad_img, output_path, disp_color)
            
            s_idx += length

        print('[-] MAKE GIF (FP, FN)')
        print('-----'*5)



if __name__ == "__main__":
    extract_fpfn()