# Phantom Deleter

The Phantom Deleter is a program for deleting imaging phantoms from DICOM images. 
Imaging Phantoms are objects with known attenuation, that have been placed externally to the patient to provide reference Hounsfield Unit (HU) values. 
While useful for calibrating attenuation, they can lead to false segmentation results in algorithms not trained on data containing phantoms.



## Installation

### Python
The tool was developed using `Python 3.8`, and this version is recommended.

### Installing packages
Required packages are defined in `setup.py`.


In the terminal, type:
1. `cd Your/Directory`
2. `pip install .`

## Usage

1. Open `phantom_detector_deleter_main.py` in your code-editor or IDE of choice.
2. Find `SRC_DIR = r""` at the bottom of the script.
3. Insert the filepath of the folder holding all DICOM files with imaging phantoms to be removed into `""`
4. Find `TARGET_DIR = r""` at the bottom of the script.
5. Insert the filepath of the folder that should hold all files with removed phantom `""`
6. Run the script

It will now read in all dicoms and try to identify and delete phantoms, saving the results into the target directory.
Errors will be collected and saved in a csv file to the same directory.

### Customization
In the call to `detect_and_delete_phantoms()`, various settings may be adjusted. 
For the publication, the current default settings were left unchanged.
1. `show_preview` - False by default to enhance performance but when set to True will show live preview of deletion.
2. `msg_every_n_files` specifies after the handling of how many files textual updates should be given to the console.
3. `backup_after_n_errors` specifies after the handling of how many files the error CSV should be updated. Keeping this high will increase performance, but also mean that progress might be loss if crashes or interruptions occur.
4. `max_thread_number` specifies how many threads should run at a single timepoint. Experiment with this number to find optimal performance.

## Functionality

### OpenCV algorithm
DICOM images are first thresholded. Then, all contours are detected. 
Various endpoints, like aspect ratio, size, coordinates, contour area, and attenuation metrics are extracted from each contour identified.

#### Identification of the contour corresponding to the phantom.

To identify the contour corresponding to the phantom, various scoring systems are in place and may be selected.

##### Manual scoring system
The default which was found to work best was based on a manual scoring system, assigning scores to different features.
That way, a metric of uncertainty is included, and phantoms will only be removed if scores reach a certain threshold.
This threshold can be adjusted by adjusting the `score_threshold` option in `opencv_phantom_detector.py`.
The default option is 9, staying rather conservative.

##### Machine Learning classifiers
Other Machine Learning based approaches were explored and classifiers (Binary Decision Tree, Random Forest, logistic regression) were trained using a manually curated dataset of contour endpoints.
They may be selected within the source code, but were not finally deployed. The training code can be found under 
`opencv_phantom_detector/contour_classifier/final_version_logistic_regression_trainer.ipynb`. 
ML models may still be deployed for classification by adjusting `mode` to "ML_score" and specifying the `prediction_model_path` in `opencv_phantom_detector.py`.
Models are available in `opencv_phantom_detection` as `.sav` files.

## Contact:
The code published here was written and deployed by P. Erik Tonnesen in collaboration with Wael Amayri.
Please refer to the corresponding author of the associated publication "Muscle Reference Values from Thoracic and Abdominal CT for Sarcopenia Assessment: The Framingham Heart Study".



### Manual phantom detector

Another manual phantom detector was developed and tested by Wael Amayri. 
This is not included in this repository, since it was not deployed on the data presented in the publication "Muscle Reference Values from Thoracic and Abdominal CT for Sarcopenia Assessment: The Framingham Heart Study".
If the functionality of said detector is of interest, Wael Amayri may be contacted.