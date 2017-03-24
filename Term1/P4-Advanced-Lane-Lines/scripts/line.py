# coding: UTF-8
import numpy as np


class Line():
    def __init__(self):
        self.is_detected = False
        self.left_fit = [np.array([False])]
        self.right_fit = [np.array([False])]

    def update(self, left_fit, right_fit):
        self.left_fit = left_fit
        self.right_fit = right_fit
        # TODO: Confirm that the lines are detected
        self.is_detected = True
