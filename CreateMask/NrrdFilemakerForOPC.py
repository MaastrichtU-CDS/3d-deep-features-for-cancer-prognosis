# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:00:15 2020

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

nrrd_store_path='A:/Dataset/Comparison Study of Radiomics and Deep Radiomics/NrrdFilesForRadiomics/'


patient_ID=os.listdir(root_source_path)

for i in range(581,605):#len(patient_ID)):

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
    
    
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=image_series_path, 
        rt_struct_path=mask_name
        )
    ROIlist=rtstruct.get_roi_names()
    if 'GTV' in ROIlist:
        mask= rtstruct.get_roi_mask_by_name("GTV")
        print("GTV")
    if 'HTV' in ROIlist:
        mask= rtstruct.get_roi_mask_by_name("HTV")
        print("HTV")
    
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
    available_z_min=available_index_z[0]
    availabel_z_max=available_index_z[-1]+1
    data_write=image_array[available_z_min:availabel_z_max,available_x_min:available_x_min+128,available_y_min:available_y_min+128]
    mask_write=mask[available_z_min:availabel_z_max,available_x_min:available_x_min+128,available_y_min:available_y_min+128]
    mask_write= np.array(mask_write,dtype=np.float64)
    
    Img=sitk.GetImageFromArray(data_write) # convert image_array to image

    Mask=sitk.GetImageFromArray(mask_write)
    
    Img.SetSpacing([np.float64(image.GetSpacing()[2]),np.float64(image.GetSpacing()[0]),np.float64(image.GetSpacing()[1])])
    Mask.SetSpacing([np.float64(image.GetSpacing()[2]),np.float64(image.GetSpacing()[0]),np.float64(image.GetSpacing()[1])])
    sitk.WriteImage(Img,nrrd_store_path+patient_ID[i]+'.nrrd',True)
    sitk.WriteImage(Mask,nrrd_store_path+patient_ID[i]+'_label.nrrd',True)