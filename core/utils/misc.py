import time
import datetime
from itertools import groupby

def encode_list(s_list): # run-length encoding from list
        return [[len(list(group)), key] for key, group in groupby(s_list)] # [[length, value], [length, value]...]

def decode_list(run_length): # run_length -> [0,1,1,1,1,0 ...]
        decode_list = []

        for length, group in run_length : 
            decode_list += [group] * length

        return decode_list

def idx_to_time(idx, fps) :
        time_s = idx // fps
        frame = int(idx % fps)
        ms = round(1/fps * frame, 2)

        # converted_time = converted_time + '.' + str(frame) # h:m:s.frame
        converted_time = str(datetime.timedelta(seconds=time_s + ms)) # h:m:s:.ms

        # exception 
        if ms == 0:
            converted_time = converted_time + '.00'

        # formatting
        hms, ms = converted_time.split('.')
        converted_time = '.'.join([hms, ms[:2]])

        return converted_time

def time_to_sec(time_str):
        sec = 0
        
        dateFormatter = "%H:%M:%S.%f"
        datetime_obj = datetime.datetime.strptime(time_str, dateFormatter)
        sec = (datetime_obj-datetime.datetime(1900, 1, 1)).total_seconds()
        
        return sec

def sec_to_time(sec):
        converted_time = str(datetime.timedelta(seconds=sec)) # h:m:s.ms

        # exception
        if sec % 1 == 0:
            converted_time = converted_time + '.00'

        # formatting
        hms, ms = converted_time.split('.')
        converted_time = '.'.join([hms, ms[:2]])

        return converted_time

def get_current_time():
    startTime = time.time()
    s_tm = time.localtime(startTime)
    
    return time.strftime('%Y-%m-%d-%H:%M:%S', s_tm), time.strftime('%Y-%m-%d %I:%M:%S %p', s_tm)


def get_rs_time_chunk(event_sequence, video_fps, inference_interval):

    RS_CLASS, NRS_CLASS = (0,1)
    rs_chunk = []

    # event_sequence = [0,0,1,0,1,0,0,1,1,1,1,1,0,0]
    encoded_list = encode_list(event_sequence)
     # [[2, 0], [1, 1], [1, 0], [1, 1], [2, 0], [5, 1], [2, 0]] // total:14
     
    sequence_idx = 0
    
    for sequence_cnt, label in encoded_list:
        
        if label == RS_CLASS:
            start_time = idx_to_time((sequence_idx * inference_interval), video_fps)
            duration_time = idx_to_time((sequence_cnt * inference_interval), video_fps)
            rs_chunk.append([start_time, duration_time])

        sequence_idx += sequence_cnt # start frame idx
    
    if rs_chunk: # event
        pass
        # print(rs_chunk)
        # TO-DO check video last?
        # last_start_time, last_duration_time = rs_chunk[-1]

    return rs_chunk

def get_nrs_frame_chunk(event_sequence, inference_interval):

    RS_CLASS, NRS_CLASS = (0,1)
    nrs_chunk = []

    # event_sequence = [0,0,1,0,1,0,0,1,1,1,1,1,0,0]
    encoded_list = encode_list(event_sequence)
     # [[2, 0], [1, 1], [1, 0], [1, 1], [2, 0], [5, 1], [2, 0]] // total:14
     
    sequence_idx = 0
    
    for sequence_cnt, label in encoded_list:
        
        if label == NRS_CLASS:
            start_frame = sequence_idx * inference_interval
            duration_frame = sequence_cnt * inference_interval
            end_frame = (start_frame + duration_frame) - 1
            nrs_chunk.append([start_frame, end_frame])

        sequence_idx += sequence_cnt # start frame idx
    
    if nrs_chunk: # event
        pass
        # print(nrs_chunk)
        # TO-DO check video last?

    return nrs_chunk

# inference_flow.py에서 가져옴
def clean_paging_chache():
    import subprocess # for CLEAR PAGING CACHE

    # clear Paging Cache because of I/O CACHE [docker run -it --name cam_io_hyeongyu -v /proc:/writable_proc -v /home/hyeongyuc/code/OOB_Recog:/OOB_RECOG -v /nas/OOB_Project:/data -p 6006:6006  --gpus all --ipc=host oob:1.0]
    print('\n\n\t ====> CLEAN PAGINGCACHE, DENTRIES, INODES "echo 1 > /writable_proc/sys/vm/drop_caches"\n\n')
    subprocess.run('sync', shell=True)
    subprocess.run('echo 1 > /writable_proc/sys/vm/drop_caches', shell=True) ### For use this Command you should make writable proc file when you run docker