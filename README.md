# 3D Deep Features for Cancer Prognosis

The source codes for manuscript 'Using 3D Deep Features from CT scans for Cancer Prognosis using a Video Classification Model'

There are three main steps in our studies --(1) deep and radiomics features extraction, (2)feature selection, and (3)Cox proportional hazards models.

![image text](https://github.com/MaastrichtU-CDS/3d-deep-features-for-cancer-prognosis/blob/main/Image/Figure%201.jpg)

### More details for step (1)

More details about radiomics feature extraction can be found at the tutorial the pyradiomics(https://pyradiomics.readthedocs.io/en/latest/)
Tools for generating mask of region of interest (ROI) in different datasets are provided in 'CreateMask' folder. For example, 'NrrdFilemakerForLUNG4.py' can be used to
generate mask of ROI in LUNG4 dataset.

For extracting 3D deep features from CT, we need to transfer ROI of CT to videos. You can find video generator at 'CreateVideo' folder to finish this part of job.
Due to the differences of mask storage in datasets, we provided different video generators for each dataset. For example, GenerateLUNG1Video.py can be used to generate 
video of ROI in LUNG1 dataset. An example of generated video -LUNG1-002.mp4- is added in 'CreateVideo' folder too.

Deep features extraction based on deep learning platform Gluon(https://mxnet.apache.org/versions/1.5.0/gluon/index.html). Therefore, before deep feature extraction, we 
need to establish this platform at your device at first(Both GPU or CPU version are ok). Then you can use FeatureExtractor.py (at 'DeepFeaturesExtraction' folder) to 
extract video's deep features. Deep features of video 'LUNG1-002.mp4' is added in 'DeepFeaturesExtraction' folder as an example. It is need some time to download 
pretrained action recognition model from when you run FeatureExtractor.py at first time.

### More details for step (2)

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.
