UAVid dataset (image only)

*Introduction
UAVid dataset is a high-resolution UAV semantic segmentation dataset focusing on street scenes.
The dataset consists of 30 sequences (seq1 to seq30), which are captured with 4K high-resolution in oblique views.
In total, 300 images have been densely labeled with 8 classes for the semantic labeling task.
The 8 classes and corresponding label color (R,G,B) are as follows:
Background clutter       	(0,0,0)
Building			(128,0,0)
Road				(128,64,128)
Tree				(0,128,0)
Low vegetation		        (128,128,0)
Moving car			(64,0,128)
Static car			(192,0,192)
Human				(64,64,0)
The dataset brings new challenges, including large scale variation, moving object recognition and temporal consistency preservation.

*Task Description
The task for UAVid dataset is to predict per-pixel semantic labelling for the UAV imagery or the UAV video sequences, and results will be evaluated with meanIoU metric.
The original video files for each sequence are provided together with the images and labels.

*Data Description
Training/validation data: each sequence is provided with images and labels. Images are named according to the frame index (0-based) in the video sequence.
Test data: only test images are provided.

*Data expansion
We have collected another 12 sequences for the dataset. They are from seq31 to seq42, and are distributed to train, val and test sets.
We recommend to use all data for training and testing. Otherwise, please only use seq1 to seq30 for training and testing as in the original paper.

*Copyright
UAVid dataset is copyright by us and published under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 License. 
This means that you must attribute the work in the manner specified by the authors, you may not use this work for commercial purposes and if you alter, transform, or build upon this work, you may distribute the resulting work only under the same license.

*Citation
When using this dataset in your research, please cite:
@article{uavid20,
  Author = {Ye Lyu and 
		George Vosselman and 
		Guisong Xia and 
		Alper Yilmaz and 
		Michael Ying Yang},
  Title  = {UAVid: A Semantic Segmentation Dataset for UAV Imagery},
  journal   = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year      = {2020},
}

*Contact
y.lyu@utwente.nl
michael.yang@utwente.nl