# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 11:23:26 2021

@author: chenj
"""

from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import WeibullAFTFitter

if __name__ == '__main__':
    data=pd.read_excel('./SelectedFeatures(VT)Excel/SelectedFeaturesLUNG1.xlsx') #reading file
    #data2=pd.read_excel('./SelectedFeaturesExcel/SelectedFeaturesH&N1.xlsx')
    CI_result=np.zeros([100,1])
    for i in range(100):
    #data = load_rossi()
        newDf =data.iloc[:,:i+3] 
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(newDf, duration_col='duration_col', event_col='event_col')
        
        CI_result[i,0]=cph.concordance_index_
        print(i)

