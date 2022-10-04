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
from rt_utils import RTStructBuilder

root_source_path='A:/Dataset/OPC-Radiomics/OPC-Radiomics/'

video_store_path='A:/Dataset/Comparison Study of Radiomics and Deep Radiomics/Videos_OPC/'


patient_ID=os.listdir(root_source_path)

for i in range(581,582):#len(patient_ID)):

    image_series_path=[]
    mask_path=[]
    
    patient_path=root_source_path+patient_ID[i]

    
    second_patient_path=patient_path+'/'+os.listdir(patient_path)[0]
    

            
    Imageitem=os.listdir(second_patient_path)
    for t in range(len(Imageitem)):
        if(len(os.listdir(second_patient_path+'/'+Imageitem[t]))==1):
            mask_path=second_patient_path+'/'+Imageitem[t]
            mask_name_list=os.listdir(mask_path)
            mask_name=mask_path+'/'+mask_name_list[0]
        if(len(os.listdir(second_patient_path+'/'+Imageitem[t]))>1):
            image_series_path=second_patient_path+'/'+Imageitem[t]
                
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(image_series_path)
    reader.SetFileNames(img_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image) # z, y, x
    image_array = image_array.astype(np.float64)

    if image_array[0,:,:].min().min()<-700:
        image_array = image_array+1000
        print("yes")
    
    #maxCTvalue=image_array[:,:,:].max().max().max()
    #minCTvalue=image_array[:,:,:].min().min().min()
    
    # CTvalueRange=2000#int(int((maxCTvalue-minCTvalue)/1000)/2)*2000
    # #image_array[image_array<0]=0
    # #image_array[image_array>CTvalueRange]=CTvalueRange
    
    #image_array=image_array+1000
    image_array[image_array<0]=0
    image_array[image_array>2000]=2000
    image_array=np.uint8(image_array*255/2000)
    


    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=image_series_path, 
        rt_struct_path=mask_name
        )
    ROIlist=rtstruct.get_roi_names()
    if 'IGTV' in ROIlist:
        mask= rtstruct.get_roi_mask_by_name("IGTV")
        print("GTV")
    if 'HTV' in ROIlist:
        mask= rtstruct.get_roi_mask_by_name("HTV")
        print("HTV")

    #mask= rtstruct.get_roi_mask_by_name("GTV")
    mask=mask.transpose((2,0,1))
    
    sum_mask_dim1=np.zeros([np.shape(mask)[0],1])
    sum_mask_dim2=np.zeros([np.shape(mask)[1],1])
    sum_mask_dim3=np.zeros([np.shape(mask)[2],1])
    for dim1 in range(np.shape(mask)[0]):
        sum_mask_dim1[dim1,0]=sum(sum(mask[dim1,:,:]))
    for dim2 in range(np.shape(mask)[1]):
        sum_mask_dim2[dim2,0]=sum(sum(mask[:,dim2,:]))
        sum_mask_dim3[dim2,0]=sum(sum(mask[:,:,dim2]))
        
        
    available_index_z=np.where(sum_mask_dim1>0)[0]
    available_index_x=np.where(sum_mask_dim2>0)[0]
    available_index_y=np.where(sum_mask_dim3>0)[0]
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
        
    video_name=video_store_path+patient_ID[i]+'.mp4'
    size=(128,128)
    videowrite = cv2.VideoWriter(video_name,-1,25,size)#20是帧数，size是图片尺寸
    for available_image in range(available_index_z[0],available_index_z[-1]+1):
        im_raw=image_array[available_image,:,:]
        im_array[:,:,0]=im_raw[available_x_min:available_x_min+128,available_y_min:available_y_min+128]
        im_array[:,:,1]=im_raw[available_x_min:available_x_min+128,available_y_min:available_y_min+128]
        im_array[:,:,2]=im_raw[available_x_min:available_x_min+128,available_y_min:available_y_min+128]
        im_array=im_array.astype(np.uint8) 
        for i in range(25):
            videowrite.write(im_array)
    videowrite.release()
