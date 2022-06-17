import os
import json
import pandas as pd
import numpy as np
import yaml
import re

class AnnotationParser():
    def __init__(self, annotation_path:str):
        self.IB_CLASS, self.OOB_CLASS = (0,1)
        self.annotation_path = annotation_path
        self.json_data = FileLoader(annotation_path).load()
    
    def set_annotation_path(self, annotation_path):
        self.annotation_path = annotation_path
        self.json_data = FileLoader(annotation_path).load()

    def get_totalFrame(self):
        return self.json_data['totalFrame'] # str

    def get_fps(self):
        return self.json_data['frameRate'] # float
    
    def get_annotations_info(self):
        # annotation frame
        annotation_idx_list = []
        
        for anno_data in self.json_data['annotations'] :
            start = anno_data['start'] # start frame number
            end = anno_data['end'] # end frame number
            code = anno_data.get('code', -100)

            annotation_idx_list.append([start, end, code]) # annotation_idx_list = [[start, end, code], [start, end, code]..]

        return annotation_idx_list

    def get_event_sequence(self, extract_interval=1):

        event_sequence = np.array([self.IB_CLASS]*self.get_totalFrame())
        annotation_idx_list = self.get_annotations_info()
        
        for start, end, code in annotation_idx_list:
            event_sequence[start: end+1] = self.OOB_CLASS
        
        return event_sequence.tolist()[::extract_interval]

class FileLoader():
    def __init__(self, file_path=''):
        self.file_path = file_path

    def set_file_path(self, file_path):
        self.file_path = file_path

    def get_full_path(self):
        return os.path.abspath(self.file_path)

    def get_file_name(self):
        return os.path.splitext(self.get_basename())[0]

    def get_file_ext(self):
        return os.path.splitext(self.get_basename())[1]
    
    def get_basename(self):
        return os.path.basename(self.file_path)
    
    def get_dirname(self):
        return os.path.dirname(self.file_path)
    
    def load(self):
        # https://stackoverflow.com/questions/9168340/using-a-dictionary-to-select-function-to-execute
        support_loader = {
            '.json':(lambda x: self.load_json()), # json object
            '.csv':(lambda x: self.load_csv()), # Dataframe
            '.yaml':(lambda x: self.load_yaml()), # dict
            '.png':-1 # PIL
        }

        data = support_loader.get(self.get_file_ext(), -1)('dummy')

        # assert data_loader != -1, 'NOT SUPPOERT FILE EXTENSION ON FileLoader'

        return data

    def load_json(self): # to json object
        with open(self.file_path) as self.json_file :
            return json.load(self.json_file)

    def load_csv(self): # to Dataframe
        df = pd.read_csv(self.file_path)
        return df
    
    def load_yaml(self): # to dict
        load_dict = {}

        with open(self.file_path, 'r') as f :
            load_dict = yaml.load(f, Loader=yaml.FullLoader)
    
        return load_dict