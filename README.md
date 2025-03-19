
# Placenta Missing Parts Database and Detection Model

## Overview

	This dataset consists of high-resolution RGB images and corresponding depth maps of human placentas, acquired with an Intel RealSense D435 stereo camera. The dataset is designed to support research into the detection and quantification of missing (deficient) regions of the placenta. In addition to the dataset, we provide a detection model that builds upon the UCNet framework for Uncertainty Inspired RGB-D Saliency Detection (Wang et al., arXiv:2009.03075). We fine-tuned a pre-trained UCNet model on our dataset, incorporating additional loss terms (Dice loss and Focal loss) to focus the network on accurately detecting missing parts.

## Data Acquisition
	•	Imaging Device: Data were acquired using the Intel RealSense D435 stereo camera, which simultaneously captures high-resolution RGB images and corresponding depth maps.
	•	Acquisition Protocol:
		The data collection process is automated using the script data_collection.py. 
		The system was deployed in a maternity ward, where three trained medical students captured images of placentas immediately after delivery. 
		For each placenta, the system automatically captured:
			1.	A full, unaltered image of the placenta.
			2.	An image with one simulated missing region.
			3.	An image with two simulated missing regions.
	•	Data Upload:
		Captured data are automatically uploaded to an online storage repository via the script uploader.py, which interfaces with Dropbox.

	Annotation Protocol
		•	Annotation Tool:
	Missing parts (deficits) on the placenta were manually simulated by medical students using annotate_tool.py. Under the guidance of a specialist in obstetrics, the students created realistic defect patterns immediately following image acquisition.
		•	Annotation Details:
	Each placenta is represented by three versions:
		•	Full image (no defects)
		•	Single-deficit image
		•	Double-deficit image

## Data Format and Structure
	•	Images:
	•	RGB images are provided in JPEG format with a resolution of 640×480 pixels (or as determined by the camera settings).
	•	Depth Maps:
	•	Depth maps are stored in CSV format, representing a 2D array of depth values.
	•	The camera intrinsic parameters used during acquisition are:
	•	Focal Length X (fx): 383.41
	•	Focal Length Y (fy): 383.41
	•	Principal Point X (ppx): 320.99
	•	Principal Point Y (ppy): 242.16
	•	Annotations:
	•	Ground truth (GT) annotations are provided as binary masks (stored as JPEG images), where a value of 1 indicates a missing region.
	•	Directory Structure:



## Automated Full Placenta Extraction
 
   	In some instances, doctors captured placenta images with the missing region physically removed, which could lead to inconsistent data for downstream analysis.
	To address this, we developed an automated full placenta extraction pipeline. 
	Using a YOLO model in conjunction with SAM2, our system rapidly annotates the full placenta, removes the background, and produces cleaner images that represent the complete organ.

## Depth Map Preprocessing:
	Due to the inherent noise in the raw depth data, all depth maps were carefully preprocessed to improve their quality for model training.
	In the prepare_placenta_dataset.py script, 
	we applied normalization and noise reduction techniques to the depth maps. 
	This step involved filtering out extreme outliers and normalizing the depth values to a consistent scale, 
	ensuring that the training process receives cleaner and more reliable depth information. 
	This preprocessing helps improve the accuracy of depth-based features during model training.

## Dataset Splitting:
	The dataset is organized into subfolders corresponding to training, validation, and test sets. Within each set,
	data are separated into directories for images, ground truth (GT) masks, and depth maps. 
	The split was performed chronologically so that the same placenta does not appear in both training and test sets, thereby preventing any data leakage.
## Provided Scripts
	•	data_collection.py:
		This script is used to capture RGB and depth data from the Intel RealSense D435 camera in a clinical setting.
	•	annotate_tool.py:
		A tool for real-time annotation of missing parts on placenta images by medical students, following guidelines provided by a specialist.
    •	uploader.py:
		Handles the automatic upload of collected data to an online storage system (Dropbox).
	•	remove_background.py:
		removes the background of the placenta 
    •	calc_volume.py:
		A script for computing the volumes of missing regions. The script calculates the volume of each missing region by approximating it as a half 3D ellipsoid. It uses robust statistics (median and IQR) to remove outliers in the x, y, and z dimensions from the depth maps before estimating the volume. This approach helps mitigate the effects of noisy depth measurements.
    •	Analysis Scripts:
		Additional scripts are provided for data analysis, including generating bar charts of missing region volumes (in cubic centimeters) and performing clustering/binned analysis.
    •	prepare_placenta_dataset.py:
		This script handles the preprocessing of the raw data. 
		It splits the dataset into training, validation, and test sets based on the acquisition time to ensure that the same placenta does not appear
		in multiple splits (preventing data leakage). In addition, it cleans the depth data by normalizing and reducing noise, and groups matching files (RGB images and corresponding depth maps) into cohesive units for downstream training and analysis.
		
# Model: UCNet-Based Detection of Missing Placental Parts

	Base Architecture
	
	Our model builds on the UCNet framework for Uncertainty Inspired RGB-D Saliency Detection (Wang et al., arXiv:2009.03075). UCNet leverages both RGB and depth information to predict salient regions, originally incorporating uncertainty estimation.
	
	Fine-Tuning and Modifications
		•	Pre-trained Weights:
	We started with the pre-trained UCNet model published by the authors and fine-tuned it on our placenta dataset.
		•	Loss Augmentation:
	To shift the model’s focus from pure saliency to the detection of missing parts, we augmented the original loss with:
		•	Dice Loss: To maximize the overlap between predicted and ground truth masks, which is crucial for detecting small, irregular deficits.
		•	Focal Loss: To address class imbalance by emphasizing hard-to-detect missing regions.
		•	Evaluation and Comparison:
	We compare the performance of our modified UCNet against a state-of-the-art detection model (YOLO-based) applied to depth data. (Further evaluation metrics will be added as more data become available.)
	
	Model Integration
	
	Our final model integrates these modifications into a unified network that:
		•	Processes both RGB and depth inputs.
		•	Produces segmentation outputs for missing regions.
		•	Is fine-tuned with the additional Dice and Focal loss terms to improve detection performance.
	
	Ethical Considerations
		•	Ethical Approval:
	All data collection procedures were approved by the relevant ethics board. Only placental images were collected, with no patient-identifiable information.
		•	Anonymization:
	All images and associated data have been anonymized to protect patient privacy.
	
	Usage and Licensing
		•	Dataset Access:
	The dataset is available for download from [Repository/Link], and users must agree to the provided usage terms.
		•	Licensing:
	[Specify your chosen license, e.g., Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA), etc.]
		•	Model Code:
	The model code, including the fine-tuned UCNet with detection modifications, is available at [GitHub Repository/Link]. We request that users cite our work if they use the dataset or model.
	
	Citation
	
	If you use this dataset or model in your research, please cite:
	
	@article{YourPaper202X,
	  title={Detection of Missing Parts in Placenta Images Using an Augmented UCNet Architecture},
	  author={Your Name and Collaborators},
	  journal={Journal/Conference Name},
	  year={202X}
	}
	
