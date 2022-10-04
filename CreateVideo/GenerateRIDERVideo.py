# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 15:54:58 2021

@author: chenj
"""

import pylidc as pl
from pylidc.utils import volume_viewer
import matplotlib.pyplot as plt
import numpy as np
import xlwt
import os
import pydicom
#from pydicom.dataset import Dataset, FileDataset
import SimpleITK as sitk
import pydicom_seg
from pydicom import dcmread
import cv2


root_source_path='A:/Dataset/RIDER/RIDER Lung CT/'

video_store_path='A:/Dataset/Comparison Study of Radiomics and Deep Radiomics/Videos_RIDER/'

patient_ID=os.listdir(root_source_path)

for i in range(len(patient_ID)):
    image_series1_path=[]
    image_series2_path=[]
    mask1_path=[]
    mask2_path=[]
    
    patient_path=root_source_path+patient_ID[i]
    
    second_path=patient_path+'/'+os.listdir(patient_path)[0]
    
    second_patient_ID=os.listdir(second_path)
    image_series_path=[]
    for j in range(len(second_patient_ID)):
        if ('RIDER' in second_patient_ID[j]) and ('RE' not in second_patient_ID[j]):
            mask1_path=second_path+'/'+second_patient_ID[j]
            testsize=os.path.getsize(mask1_path+'/'+os.listdir(mask1_path)[0])
        if ('RIDER' in second_patient_ID[j]) and ('RE' in second_patient_ID[j]):
            mask2_path=second_path+'/'+second_patient_ID[j]
            retestsize=os.path.getsize(mask2_path+'/'+os.listdir(mask2_path)[0])
        if ('TEST' not in second_patient_ID[j]):
            image_series_path.append(second_patient_ID[j])
        
    if (testsize>retestsize):
        print("Hello")
        if len(os.listdir(second_path+'/'+image_series_path[0]))>len(os.listdir(second_path+'/'+image_series_path[1])):
            image_series1_path=second_path+'/'+image_series_path[0]
            image_series2_path=second_path+'/'+image_series_path[1]
        else:
            image_series1_path=second_path+'/'+image_series_path[1]
            image_series2_path=second_path+'/'+image_series_path[0]
    else:
        if len(os.listdir(second_path+'/'+image_series_path[0]))>len(os.listdir(second_path+'/'+image_series_path[1])):
            image_series1_path=second_path+'/'+image_series_path[1]
            image_series2_path=second_path+'/'+image_series_path[0]
        else:
            image_series1_path=second_path+'/'+image_series_path[0]
            image_series2_path=second_path+'/'+image_series_path[1]
            


    if len(image_series1_path)>0 and len(image_series2_path)>0:           
        reader1 = sitk.ImageSeriesReader()
        img_names1 = reader1.GetGDCMSeriesFileNames(image_series2_path)
        reader1.SetFileNames(img_names1)
        image1 = reader1.Execute()
        image_array1 = sitk.GetArrayFromImage(image1) # z, y, x
        image_array1 = image_array1.astype(np.float64)

        if image_array1[0,:,:].min().min()<-700:
            image_array1 = image_array1+1000
            print("yes")
        
        
#     #maxCTvalue=image_array[:,:,:].max().max().max()
#     #minCTvalue=image_array[:,:,:].min().min().min()
    
#     # CTvalueRange=2000#int(int((maxCTvalue-minCTvalue)/1000)/2)*2000
#     # #image_array[image_array<0]=0
#     # #image_array[image_array>CTvalueRange]=CTvalueRange
    
    #image_array=image_array+1000
        image_array1[image_array1<0]=0
        image_array1[image_array1>2000]=2000
        image_array1=image_array1*256/2000
        mask1_name=mask2_path+'/'+os.listdir(mask2_path)[0]
        


        dcm = pydicom.dcmread(mask1_name)

        #for segment_number in result.available_segments:
            #if (segmentinfos[segment_number]['SegmentDescription'].value) =='GTV-1':
        mask1= dcm.pixel_array
        mask1=mask1[0:image_array1.shape[0],:,:]
    
    
        sum_mask1_dim1=np.zeros([np.shape(mask1)[0],1])
        sum_mask1_dim2=np.zeros([np.shape(mask1)[1],1])
        sum_mask1_dim3=np.zeros([np.shape(mask1)[2],1])
        for dim1 in range(np.shape(mask1)[0]):
            sum_mask1_dim1[dim1,0]=sum(sum(mask1[dim1,:,:]))
        for dim2 in range(np.shape(mask1)[1]):
            sum_mask1_dim2[dim2,0]=sum(sum(mask1[:,dim2,:]))
            sum_mask1_dim3[dim2,0]=sum(sum(mask1[:,:,dim2]))
        
        
        available_index_z=np.where(sum_mask1_dim1>0)[0]
        available_index_x=np.where(sum_mask1_dim2>0)[0]
        available_index_y=np.where(sum_mask1_dim3>0)[0]
        if available_index_x[0]<32:
            available_x_min=0
        elif available_index_x[0]>416:
            available_x_min=384
        else:
            available_x_min=available_index_x[0]-32
            
        if available_index_y[0]<32:
            available_y_min=0
        elif available_index_y[0]>416:
            available_y_min=384
        else:
            available_y_min=available_index_y[0]-32
        
        print(available_x_min)
        print(available_y_min)
        im_array=np.zeros([128,128,3])
    
    #available_x_min=0
    #available_y_min=0
        
        video_name=video_store_path+patient_ID[i]+'-ReTest.mp4'
        size=(128,128)
        videowrite = cv2.VideoWriter(video_name,-1,25,size)#20是帧数，size是图片尺寸
        for available_image in range(available_index_z[0],available_index_z[-1]+1):
            im_raw=image_array1[available_image,:,:]
            im_array[:,:,0]=im_raw[available_x_min:available_x_min+128,available_y_min:available_y_min+128]
            im_array[:,:,1]=im_raw[available_x_min:available_x_min+128,available_y_min:available_y_min+128]
            im_array[:,:,2]=im_raw[available_x_min:available_x_min+128,available_y_min:available_y_min+128]
            im_array=im_array.astype(np.uint8) 
            for i in range(25):
                videowrite.write(im_array)
        videowrite.release()
