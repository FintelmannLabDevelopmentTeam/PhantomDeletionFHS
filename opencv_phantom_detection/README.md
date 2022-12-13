# Phantom detector using OpenCV

### Setup
Requires the following modules:

* numpy
* pandas
* scikit-learn
* cv2 (openCV)
* matplotlib
* pickle

They can all be installed via `pip` package manager using `pip install packagename`, where `packagename` has to be replaced with the name of the package

## Usage

Import via   
> `from opencv_phantom_detector import Phantom_detector as pdetec`  

Then we can use   

> `detector = pdetec()`  
> `try:`  
>   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`left_x, right_x, upper_y, lower_y = detector.find_phantom(dcm)`    
>  `except:`  
>   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `print("Could not read file")`
> 
> 
> 
where `dcm` is a dicom FileArray, as imported via `dicom_reader.py`.

