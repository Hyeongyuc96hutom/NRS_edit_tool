import time
import os
import pandas as pd
import json
import numpy as np

from core.utils.parser import FileLoader # file load helper


# use in Report class (def save_report) : save json file for numpy floats (should casting)
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)



class ReportAnnotation():
    def __init__(self, report_save_path):
        self.report_save_path = report_save_path # .json

        self.total_report = self._init_total_report_form()
        self.annotations = self.total_report['annotations']

    def _init_total_report_form(self):
        init_total_report_form = {
            'totalFrame':'', 
            'frameRate':'',
            'width':'',
            'height':'',
            '_id':'',
            'annotations': [],
            'annotationType': '',
            'createdAt': '',
            'updatedAt': '',
            'annotator': '',
            'name': '',
            'label':'',
        }

        return init_total_report_form
    
    def _get_report_form(self, report_type):
        init_report_form = { # one-columne of annotations report
            'annotation': { # one-columne of inference report (each patients)
                'start':'',
                'end':'',
                'code':'',
            }
        }

        return init_report_form[report_type]
    
    def set_report_save_path(self, report_save_path):
        self.report_save_path = report_save_path

    def set_total_report(self, totalFrame, frameRate, width, height, _id, annotationType, createdAt, updatedAt, annotator, name, label):
        self.total_report['totalFrame'] = totalFrame
        self.total_report['frameRate'] = frameRate
        self.total_report['width'] = width
        self.total_report['height'] = height
        self.total_report['_id'] = _id
        self.total_report['annotationType'] = annotationType
        self.total_report['createdAt'] = createdAt
        self.total_report['updatedAt'] = updatedAt
        self.total_report['annotator'] = annotator
        self.total_report['name'] = name
        self.total_report['label'] = label
    

    
    def add_annotation_report(self, start, end, code):
        annotation = self._get_report_form('annotation')

        annotation['start'] = start
        annotation['end'] = end
        annotation['code'] = code
        
        self.annotations.append(annotation)

        return annotation

    def clean_report(self):
        self.total_report = self._init_total_report_form()
        self.annotations = self.total_report['annotations']
    
    def load_report(self):
        if os.path.isfile(self.report_save_path):
            f_loader = FileLoader()
            f_loader.set_file_path(self.report_save_path)
            saved_report_dict = f_loader.load()        
        
        self.total_report = saved_report_dict
        self.annotations = self.total_report['annotations']

    def get_annotations(self):
        return self.annotations


    def save_report(self):
        os.makedirs(os.path.dirname(self.report_save_path), exist_ok=True)

        json_string = json.dumps(self.total_report, indent=4, cls=MyEncoder)
        print(json_string)

        with open(self.report_save_path, "w") as json_file:
            json.dump(self.total_report, json_file, indent=4, cls=MyEncoder)