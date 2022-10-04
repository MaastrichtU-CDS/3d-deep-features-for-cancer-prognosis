# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 16:48:05 2022

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
from dicompylercore import dicomparser, dvh, dvhcalc
from DicomRTTool import DicomReaderWriter
from RegisterImages.WithDicomReg import register_images_with_dicom_reg, pydicom, sitk



patient_root_path='A:/Dataset/ABCs/GLIS-RT/'

patient_ID=os.listdir(patient_root_path)

nrrd_store_path='A:/Dataset/Comparison Study of Radiomics and Deep Radiomics/NrrdFilesForRadiomicsABCs/'
for i in range(240,260):#len(patient_ID)):
    
#------------------------------------------------------------------------------------
#extract REG, RTSTRUCT, CT Series, MR T1 Series path in ABCs dataset
#------------------------------------------------------------------------------------
    patient_path_2=patient_root_path+patient_ID[i]
    image_modality_path=os.listdir(patient_path_2)
    for j in range(len(image_modality_path)):
        if "MRI" in image_modality_path[j]:
            MRI_Image_path=patient_path_2+'/'+image_modality_path[j]
        else:
            CT_Image_path=patient_path_2+'/'+image_modality_path[j]
    
    CT_Image_file=os.listdir(CT_Image_path)

    for j in range(len(CT_Image_file)):
        #RT_file="string"
        if "CTT1" in CT_Image_file[j]:
            Reg_file_path=CT_Image_path+'/'+CT_Image_file[j]+'/'+os.listdir(CT_Image_path+'/'+CT_Image_file[j])[0]
        if "Targets" in CT_Image_file[j]:
            RT_file=CT_Image_path+'/'+CT_Image_file[j]+'/'+os.listdir(CT_Image_path+'/'+CT_Image_file[j])[0]
        else:
            potential_path=CT_Image_path+'/'+CT_Image_file[j]
            # if "Targets" in CT_Image_file[j]:
            #     RT_file=potential_path+'/'+os.listdir(potential_path)[0]
            potential_file=potential_path+'/'+os.listdir(potential_path)[0]
            file_modality=pydicom.read_file(potential_file)
            if file_modality.Modality=="CT":
                CT_series_path=potential_path
        
    MRI_Image_file=os.listdir(MRI_Image_path)
    
    for j in range(len(MRI_Image_file)):
        potential_path=MRI_Image_path+'/'+MRI_Image_file[j]
        if ('T2' not in MRI_Image_file[j]) and ('FLAIR' not in MRI_Image_file[j]):
            MRI_series_path=potential_path
        # Potential_MRI_file=potential_path+'/'+os.listdir(potential_path)[0]
        # MRI_modality=pydicom.read_file(Potential_MRI_file)
        # if MRI_modality.Modality=="MR" and ("3d1" in MRI_modality.SequenceName):
            
#-------------------------------------------------------------------------------------
#registration of CT　and MRI T1 images based on REG Files and resample MR into CT size
#-------------------------------------------------------------------------------------

    fixed_path = CT_series_path#'A:/Dataset/ABCs/GLIS-RT/GLI_002_GBM/07-09-2008-NA-IMRT BRAIN-01704/3.000000-BRAIN IMRT-59252/'
    moving_path = MRI_series_path#'A:/Dataset/ABCs/GLIS-RT/GLI_002_GBM/07-09-2008-NA-MRI BRAIN INTRAOP WITH AND WITHOUT CONTRAST-13340/106.000000-AX POST MPRAGE MPR-52888/'

    fixed_reader = sitk.ImageSeriesReader()
    fixed_img_names = fixed_reader.GetGDCMSeriesFileNames(fixed_path)
    fixed_reader.SetFileNames(fixed_img_names)
    fixed_reader.MetaDataDictionaryArrayUpdateOn()
    fixed_reader.LoadPrivateTagsOn()
    fixed_image = fixed_reader.Execute()
    fixed_image_array=sitk.GetArrayFromImage(fixed_image)
    #fixed_image_array[np.where(fixed_image_array<-1024)]=-1024
    #fixed_image_array[np.where(fixed_image_array>1024)]=1024
    #fixed_image_2 = sitk.GetImageFromArray(fixed_image_array)
    # windows_center_str2=fixed_reader.GetMetaData(0,'0028|1050')
    # windows_center_num2=int(windows_center_str2.split('\\')[0])-25
    # windows_width_str2=fixed_reader.GetMetaData(0,'0028|1051')
    # windows_width_num2=int(windows_width_str2.split('\\')[0])


    moving_reader = sitk.ImageSeriesReader()
    moving_img_names = moving_reader.GetGDCMSeriesFileNames(moving_path)
    moving_reader.SetFileNames(moving_img_names)
    moving_reader.MetaDataDictionaryArrayUpdateOn()
    moving_reader.LoadPrivateTagsOn()
    moving_image = moving_reader.Execute()
    moving_image_array=sitk.GetArrayFromImage(moving_image)
    # windows_center_str=moving_reader.GetMetaData(0,'0028|1050')
    # windows_center_num=int(windows_center_str.split('\\')[0])
    # windows_width_str=moving_reader.GetMetaData(0,'0028|1051')
    # windows_width_num=int(windows_width_str.split('\\')[0])

    registration_file = Reg_file_path
    dicom_registration = pydicom.read_file(registration_file)

    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    resampled_moving = register_images_with_dicom_reg(fixed_image=fixed_image, moving_image=moving_image, dicom_registration=dicom_registration)
    resampled_moving_array = sitk.GetArrayFromImage(resampled_moving)
    # resampled_moving_array[np.where(resampled_moving_array<(windows_center_num-windows_width_num/2))]=windows_center_num-windows_width_num/2
    # resampled_moving_array[np.where(resampled_moving_array>(windows_center_num+windows_width_num/2))]=windows_center_num+windows_width_num/2
    # fixed_image_array[np.where(fixed_image_array<(windows_center_num2-windows_width_num2/2))]=windows_center_num2-windows_width_num2/2
    # fixed_image_array[np.where(fixed_image_array>(windows_center_num2+windows_width_num2/2))]=windows_center_num2+windows_width_num2/2
#-------------------------------------------------------------------------------------
#Read RTSTRUCT File and Reconstuct Mask of ROI
#-------------------------------------------------------------------------------------

    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=CT_series_path, 
        rt_struct_path=RT_file
        )
    ROIlist=rtstruct.get_roi_names()
    if 'GTV' in ROIlist:
        mask= rtstruct.get_roi_mask_by_name("GTV")
    
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
#-----------------------------------------------------------------------------------
#Store Nrrd Files
#-----------------------------------------------------------------------------------        
    print(available_x_min)
    print(available_y_min)
    print(patient_ID[i])
    available_z_min=available_index_z[0]
    availabel_z_max=available_index_z[-1]+1
    MRI_data_write=resampled_moving_array[available_z_min:availabel_z_max,available_x_min:available_x_min+128,available_y_min:available_y_min+128]
    CT_data_write=fixed_image_array[available_z_min:availabel_z_max,available_x_min:available_x_min+128,available_y_min:available_y_min+128]
    mask_write=mask[available_z_min:availabel_z_max,available_x_min:available_x_min+128,available_y_min:available_y_min+128]
    mask_write= np.array(mask_write,dtype=np.float64)
    
    CT_Img=sitk.GetImageFromArray(CT_data_write) # convert image_array to image
    MRI_Img=sitk.GetImageFromArray(MRI_data_write)
    Mask=sitk.GetImageFromArray(mask_write)
    
    CT_Img.SetSpacing([np.float64(fixed_image.GetSpacing()[2]),np.float64(fixed_image.GetSpacing()[0]),np.float64(fixed_image.GetSpacing()[1])])
    MRI_Img.SetSpacing([np.float64(fixed_image.GetSpacing()[2]),np.float64(fixed_image.GetSpacing()[0]),np.float64(fixed_image.GetSpacing()[1])])
    Mask.SetSpacing([np.float64(fixed_image.GetSpacing()[2]),np.float64(fixed_image.GetSpacing()[0]),np.float64(fixed_image.GetSpacing()[1])])
    sitk.WriteImage(CT_Img,nrrd_store_path+patient_ID[i]+'_CT.nrrd',True)
    sitk.WriteImage(MRI_Img,nrrd_store_path+patient_ID[i]+'_MRI.nrrd',True)
    sitk.WriteImage(Mask,nrrd_store_path+patient_ID[i]+'_label.nrrd',True)
    
# # outputCT_root_path='A:/Dataset/Comparison Study of Radiomics and Deep Radiomics/CTimageSeries/'
# # outputMRI_root_path='A:/Dataset/Comparison Study of Radiomics and Deep Radiomics/MRIimageSeries/'

# # for i in range(np.shape(fixed_image_array)[0]):
# #     cv2.imwrite(outputMRI_root_path+str(i)+".png",np.array((resampled_moving_array[i,:,:]-windows_center_num)*255/windows_width_num,np.int8))
# #     cv2.imwrite(outputCT_root_path+str(i)+".png",np.array((fixed_image_array[i,:,:]-windows_center_num2)*255/windows_width_num2,np.int8))

