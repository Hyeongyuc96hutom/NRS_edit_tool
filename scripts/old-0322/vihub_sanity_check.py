
import os, glob, natsort

anno_path = '/NRS_editing/mobilenet-set1-apply1/TB_log/version_4/inference_results'
result_path1 = '/nas1/ViHUB-pro/results/edited_video/ViHUB_0314/gangbuksamsung'
result_path2 = '/nas1/ViHUB-pro/results/edited_video/ViHUB_0314/severance_1st'
result_path3 = '/nas1/ViHUB-pro/results/edited_video/ViHUB_0314/severance_2nd'

anno_list = glob.glob(os.path.join(anno_path, '*', '*', '*.csv'))
anno_list2 = glob.glob(os.path.join(anno_path, '*', '*', '*.csv'))

result_list1 = glob.glob(os.path.join(result_path1, '*', '*', '*.mp4'))
result_list2 = glob.glob(os.path.join(result_path2, '*', '*' ,'*', '*.mp4'))
result_list3 = glob.glob(os.path.join(result_path3, '*', '*', '*', '*.mp4'))

result_list = result_list1+result_list2+result_list3

for anno in anno_list:
    csv_f = anno.split('/')[-1]
    csv_f = os.path.splitext(csv_f)[0]

    for result_video in result_list:
        video_f = result_video.split('/')[-1]
        video_f = os.path.splitext(video_f)[0].split('-')[0]

        if csv_f == video_f:
            anno_list2.remove(anno)

anno_list2 = natsort.natsorted(anno_list2)
for i in anno_list2:
    print(i)

print(len(anno_list2))

# for result_video in result_list:
#     print(result_video)
