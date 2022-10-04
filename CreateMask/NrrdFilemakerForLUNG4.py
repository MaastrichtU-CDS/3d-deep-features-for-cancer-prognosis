# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:13:30 2022

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
from DicomRTTool.ReaderWriter import DicomReaderWriter
from rt_utils import RTStructBuilder
from dicompylercore import dicomparser, dvh, dvhcalc

root_source_path='A:/Dataset/Lung 4/'

nrrd_store_path='A:/Dataset/Comparison Study of Radiomics and Deep Radiomics/NrrdFilesForRadiomicsLung4/'


patient_ID=os.listdir(root_source_path)

for i in range(321,322):
    image_series_path=[]
    mask_path=[]
    print(patient_ID[i])
    patient_path=root_source_path+patient_ID[i]
    
    second_level_pathlist=os.listdir(patient_path)
    
    for j in range(len(second_level_pathlist)):
        second_level_patientpath=patient_path+'/'+second_level_pathlist[j]
        third_level_pathlist=os.listdir(second_level_patientpath)
        if 'DICOM' in third_level_pathlist:
            image_series_path=second_level_patientpath+'/DICOM/'
        else:
            mask_path=second_level_patientpath+'/'+third_level_pathlist[0]+'/'
            mask_name=mask_path+os.listdir(mask_path)[0]
    
    
    
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(image_series_path)
    reader.SetFileNames(img_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image) # z, y, x
    image_array = image_array.astype(np.float64)

    if image_array[0,:,:].min().min()<-700:
        image_array = image_array+1000
        print("yes")
    

    image_array[image_array<0]=0
    image_array[image_array>2000]=2000
    image_array=image_array*255/2000
    Dicom_reader = DicomReaderWriter(description='Examples', arg_max=True)

    path = patient_path
    Dicom_reader.walk_through_folders(path)
    Cont_Names = ['gtv-1']
    Dicom_reader.set_contour_names_and_associations(Contour_Names=Cont_Names)
    indexes = Dicom_reader.which_indexes_have_all_rois()
    pt_indx = indexes[-1]
    Dicom_reader.set_index(pt_indx)
    Dicom_reader.get_images_and_mask()
    #image = Dicom_reader.ArrayDicom # [Images, rows, cols]
    mask = Dicom_reader.mask # [Images, rows, cols, # classes + 1]
    
    # rtstruct = RTStructBuilder.create_from(
    #     dicom_series_path=image_series_path, 
    #     rt_struct_path=mask_name
    #     )
    # ROIlist=rtstruct.get_roi_names()
    # if 'GTV-1' in ROIlist:        
    #     mask= rtstruct.get_roi_mask_by_name("GTV-1")
        
    # mask=mask.transpose((2,0,1))
    
    
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