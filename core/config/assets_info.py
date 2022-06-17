
"""
    annotation, img_db path
    oob_assets save path
"""

video_path = {
    # dataset 1 (40 case) + dataset 2 (60 case)
    'robot': ['/data1/HuToM/Video_Robot_cordname', '/data2/Video/Robot/Dataset2_60case'],
}

annotation_path = {
    'annotation_v1_base_path': '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V1',
    'annotation_v2_base_path': '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V2',
    'annotation_v3_base_path': '/data2/Public/IDC_21.06.25/ANNOTATION/Gastrectomy/Event/OOB/V3/TBE'
}

img_db_path = {
    'robot': '/raid/img_db/ROBOT', ### cuaton @@@ changed key @@@ 12 -> robot
}

oob_assets_save_path = {
    'oob_assets_v1_robot_save_path':  '/data2/Public/OOB_Recog/oob_assets/V1/ROBOT',
    'oob_assets_v1_lapa_save_path':'/data2/Public/OOB_Recog/oob_assets/V1/LAPA',

    'oob_assets_v2_robot_save_path':'/raid/img_db/oob_assets/V2/ROBOT',
    'oob_assets_v2_lapa_save_path': '/raid/img_db/oob_assets/V2/LAPA',

    'oob_assets_v3_robot_save_path': '/raid/img_db/oob_assets/V3/ROBOT/211207',

    'theator-oob_assets_v3_robot_save_path': '/raid/img_db/oob_assets/V3/theator-ROBOT'
}

mc_assets_save_path = {
    'robot': '/data2/Public/OOB_Recog/offline',
}