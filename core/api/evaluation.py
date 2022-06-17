# evalation module
# 1. gt to list
# 2. calc metric

import os
import csv
import pandas as pd
import numpy as np

import json
import argparse

from core.utils.metric import MetricHelper # metric util
from core.utils.parser import AnnotationParser, FileLoader # json util, file util
from core.utils.parser import FileLoader 

class Evaluator():
    def __init__(self, model_output_csv_path:str, gt_json_path:str, inference_interval:int):
        self.model_output_csv_path = model_output_csv_path
        self.gt_json_path = gt_json_path
        self.inference_interval = inference_interval

    def _fit_to_min_length(self, gt_list, predict_list):
        gt_len, predict_len = len(gt_list), len(predict_list)
        min_len = min(gt_len, predict_len)

        gt_list, predict_list = gt_list[:min_len], predict_list[:min_len]

        return gt_list, predict_list
        
    def set_model_output_csv_path(self, model_output_csv_path):
        self.model_output_csv_path = model_output_csv_path

    def set_gt_json_path(self, gt_json_path):
        self.gt_json_path = gt_json_path

    def set_path(self, model_output_csv_path, gt_json_path):
        self.model_output_csv_path = model_output_csv_path
        self.gt_json_path = gt_json_path

    def get_assets(self): # return gt_list(from .json), predict_list(from .csv)
        file_loader= FileLoader(self.model_output_csv_path) # csv
        predict_df = file_loader.load() # DataFrame
        predict_list = list(predict_df['predict'].tolist())

        anno_parser = AnnotationParser(self.gt_json_path)
        gt_list = anno_parser.get_event_sequence(self.inference_interval)
        gt_list, predict_list = self._fit_to_min_length(gt_list, predict_list)

        return gt_list, predict_list

    def calc(self):
        """
        Calculate Over_Estimation_Ratio(OR) and Confidence_Ratio(CR).
        {
            "01_G_01_R_100_ch1_03": {
                "over_estimation_ratio": 0.01046610685164902,
                "confidence_ratio": 0.9785809906291834
        }

        Returns:
            `turple`, of Over_Estimation_Ratio and Confidence_Ratio.
            [self.over_estimation_ratio, self.confidence_ratio]

        Example:
            >>> calc()
                metrics = {
                    'TP': 
                    'TN': 
                    'FP': 
                    'FN': 
                    'Accuracy': 
                    'Precision': 
                    'Recall': 
                    'F1-Score': 

                    'OOB_metric':
                    'Over_estimation':
                    'Under_estimation':
                    'Correspondence_estimation':
                    'UNCorrespondence_estimation':
                }
        """

        # 1. prepare data(assets)
        gt_list, predict_list = self.get_assets()

        # 2. calc metric
        metric_helper = MetricHelper()
        metric_helper.write_preds(np.array(predict_list), np.array(gt_list))
        metrics = metric_helper.calc_metric()

        return metrics

