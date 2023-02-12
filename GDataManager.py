import numpy as np

class GDataManager:
    def __init__(self):
        self.images:np.array = None
        self.displayImages:np.array = None
        self.mask:np.array = None
        self.cur_img: np.array = None
        self.cur_mask: np.array = None
