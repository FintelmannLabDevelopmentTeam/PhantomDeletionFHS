import os
import cv2
import numpy as np
import pandas as pd
from pydicom import FileDataset




# This file holds helpful methods to keep the opencv_phantom_detector clean.

def create_thresholded_image(img: np.ndarray, thresholding_value=-100):
    '''
    Creates a binarily thresholded image.
    :param img: The image to which the filter should be applied. It will not be altered.
    :type img: array
    :param thresholding_value: Everything > this value will become white (255 on grey scale), everything else black.
    :type thresholding_value: int
    :return: The thresholded image as uint8 array.
    '''

    retval, thresholded_image = cv2.threshold(img, thresholding_value, 255, cv2.THRESH_BINARY)
    return thresholded_image


def find_contours(img: np.ndarray, hierarchy=cv2.RETR_LIST, approximation=cv2.CHAIN_APPROX_SIMPLE):
    '''
    Calculates contour objects on a binary image.
    :param img: The binary image array.
    :param hierarchy: The method for calculating the hierarchy. see https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    :param approximation: The method for approximating the shape, if any. See https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
    :return: The calculated contours list, as well as the hierarchy list.
    '''
    img_8bit = np.uint8(img)
    contours, hierarchy = cv2.findContours(img_8bit, hierarchy, approximation)
    return contours, hierarchy


def get_endpoints_from_contour(cnt: list, original_image: np.ndarray, thresholded_image: np.ndarray, mode: str="manual_score", hierarchy=None):
    '''
    Calculates endpoints from contour data.
    :param cnt: The contour data.
    :param original_image: The original image array, not turned to uint8 yet. This is important, so that the algorithm can estimate mean values, max values etc.
    :param mode: Decides, which values are being calculated. Can be "manual_score", or "ML_score".
    :param hierarchy: The hierarchy object corresponding to the contour. Might be interesting to look at in the future as another decision value.
    :return: our endpoints in a dictionary.
    '''


    area = cv2.contourArea(cnt)

    if(mode != "manual_score"):
        moments = cv2.moments(cnt)

        if (moments['m00'] == 0):
            cx, cy = (-100, -100)  # give it unrealistic values, if division cannot be performed
        else:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            centroid = (cx, cy)
    else:
        moments=None
        cx = None
        cy = None
        centroid = None


    try:
        (x, y), (MA, ma), orientation_angle = cv2.fitEllipse(cnt)
    except:
        orientation_angle = 0  # 5 points are required to fit ellipse. Otherwise just save this to dictionary.

    left_x, upper_y, w, h = cv2.boundingRect(cnt)
    right_x = left_x + w
    lower_y = upper_y + h
    bounding_box_area = w * h

    if h == 0:
        aspect_ratio = 0
    else:
        aspect_ratio = float(w) / h  # width through height of bounding box

    if bounding_box_area == 0:
        extent = 0
    else:
        extent = float(area) / bounding_box_area  # Aspect of bounding box to real size

    mask = np.zeros(np.uint8(original_image).shape, np.uint8)
    cv2.rectangle(mask, (left_x, upper_y), (right_x, lower_y), (255, 255, 255), -1)
    # Check if these work correctly
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(original_image, mask=mask)


    first_third_x = int(left_x + (right_x - left_x) * 0.33)
    second_third_x = int(left_x + (right_x - left_x) * 0.66)

    mean_val_th = np.average(thresholded_image[upper_y:lower_y, left_x:right_x])

    if ((first_third_x == left_x) | (upper_y == lower_y)):
        first_third_mean = -10000
    else:
        first_third_mean = np.average(original_image[upper_y:lower_y, left_x:first_third_x])

    if ((first_third_x == second_third_x) | (upper_y == lower_y)):
        second_third_mean = -10000
    else:
        second_third_mean = np.average(original_image[upper_y:lower_y, first_third_x:second_third_x])

    if ((second_third_x == right_x) | (upper_y == lower_y)):
        third_third_mean = -10000
    else:
        third_third_mean = np.average(original_image[upper_y:lower_y, second_third_x:right_x])



    mean_val = np.average(original_image[upper_y:lower_y, left_x:right_x])


    return_dict = {
        "upper_y": upper_y,
        "lower_y": lower_y,
        "left_x": left_x,
        "right_x": right_x,
        "mean_val": mean_val,
        "mean_val_thresholded": mean_val_th,
        "first_third_mean":first_third_mean,
        "second_third_mean":second_third_mean,
        "third_third_mean":third_third_mean,
        "centroid": centroid,
        "centroid_x": cx,
        "centroid_y": cy,
        "area": area,
        "bounding_box_area": bounding_box_area,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "min_val": min_val,
        "max_val": max_val,
        "min_loc": min_loc,
        "max_loc": max_loc,
        "orientation_angle": orientation_angle,
        "moments": moments,
        "hierarchy": hierarchy
    }

    return return_dict


def crop_image(img: np.ndarray, top_cut_off=256, side_cut_off=100):
    '''
    Will slice an image array to become smaller.
    :param img: The image array to be sliced.
    :param top_cut_off: The pixels to be cut off at the top.
    :param side_cut_off: The pixels to be cut off at either side.
    :return: the sliced image array.
    '''
    image_size = img.shape[1]
    return img[top_cut_off:, side_cut_off:image_size - side_cut_off]


def find_phantom_in_contours_using_ML_model(contours: list, original_img: np.ndarray, thresholded_img: np.ndarray,
                                            loaded_model, finish_early_acceptance_threshold=0.95):
    '''
    Finds the most likely phantom contour out of a list of contours.
    :param contours: The list of contours.
    :param original_img: The image pixelarray.
    :param thresholded_img: The thresholded binary pixelarray.
    :param loaded_model: The ML classifier model.
    :param finish_early_acceptance_threshold: A value that will, when surpassed by the predictive value, allow early cancelling of loop.
    :return: The endpoints of the phantom contour, its prediction value, and its index.
    '''

    # Test this first, but in the future have model loaded once. Could make this a class and set it as class variable.

    i = 0
    highest_pred = 0
    highest_ep = None
    highest_index = i

    for cnt in contours:
        ep = get_endpoints_from_contour(cnt, original_img, thresholded_img, mode="ML_score", hierarchy=None)
        ep["contour no"] = i

        # create array in right shape to be read in by logistic regression model:
        # upper_y	lower_y	left_x	right_x	mean_val	mean_val_thresholded	centroid_x	centroid_y	area	bounding_box_area	aspect_ratio	extent	min_val	max_val	orientation_angle	contour no
        property_list = [ep["upper_y"], ep["lower_y"], ep["left_x"], ep["right_x"], ep["mean_val"],
                         ep["mean_val_thresholded"], ep["centroid_x"], ep["centroid_y"], ep["area"],
                         ep["bounding_box_area"], ep["aspect_ratio"], ep["extent"], ep["min_val"], ep["max_val"],
                         ep["orientation_angle"], ep["contour no"], ep["first_third_mean"], ep["second_third_mean"], ep["third_third_mean"]]
        X = np.array(property_list)
        predictions = loaded_model.predict_proba(X.reshape(1, -1))
        prediction = predictions[0, 1]  # get positive prediction chance

        if (prediction > finish_early_acceptance_threshold):  # finish early
            return ep, prediction, i

        if (prediction > highest_pred):
            highest_pred = prediction
            highest_ep = ep
            highest_index = i

        i += 1
    return highest_ep, highest_pred, highest_index


def create_adjusted_image(dcm: FileDataset, top_cut_off=256, side_cut_off=100):
    '''
    This calculates the factor by which to multiply our image data in order to have the attenuation levels correct for all images. It then applies that to the pixel_array and return sit.
    :param dcm: The dicom file to be checked.
    :return: The adjusted image array.
    '''

    # Important are: Round1CT: 35cm thoracic2, lumbar5_50
    # Round2CT: ABDOMEN 5_50, CORONARY 25_50

    # https://blog.kitware.com/dicom-rescale-intercept-rescale-slope-and-itk/
    rescale_intercept = dcm.RescaleIntercept
    rescale_slope = dcm.RescaleSlope
    img = dcm.pixel_array
    img = crop_image(img, top_cut_off, side_cut_off)

    img = img * rescale_slope + rescale_intercept

    return img

def find_phantom_in_contours(contours: list, original_img: np.ndarray, thresholded_img: np.ndarray, stop_early_threshold = 10,):
    '''
    Will find the contour that is most likely the phantom in a list of contours. Does that based upon a score that is calculated for that contour.
    :param contours: A list of the contours to be searched for a phantom-like one.
    :param original_img: The image on which the contours lie.
    :param thresholded_img: The thresholded image array.
    :param stop_early_threshold: When this score is reached, the other contours will not be checked anymore.
    :return: The endpoints of the best contour, as well as the score it achieved, the distribution of scores, and the index of the "winning" contour.
    '''

    highest_score = 0
    cnt_endpoints = None

    score_distribution = {}
    for i in range(11):
        score_distribution[i] = 0

    i = 0
    best_index = 0

    for cnt in contours:
        ep = get_endpoints_from_contour(cnt, original_img, thresholded_img, mode="manual_score", hierarchy=None)
        # Would also like mean value
        # could here do either manual decisions, or deploy DL network
        # will use manual selection for now, based on csv data extracted from 5 patients, and the corresponding binary decision tree.
        # of course, this can and has to be improved.

        # with smaller images, area might also not be the best measurement
        score = 0

        # if(ep["mean_val_thresholded"] > 240):
        #    score+=1
        if ( abs(ep["mean_val"]) <100 ): #the problem with the mean calculations is, that when turned, the bounding box will also contain a lot of air, eg negative values
            score += 1
        if ((ep["area"] > 1600) and (
                ep["area"] < 4000)):
            score += 1
        if ((ep["aspect_ratio"] > 2) and (ep["aspect_ratio"]<4.15)):
            score += 1
        if ((ep[
            "extent"] >= 0.6)):  # bounding box to real size ratio. According to binary tree learning the strongest predictor
            score += 1
        if (ep[
            "min_val"] < -400):  # also used as predictor by decision tree. Rationally not the best choice though, since there might be holes in the phantom data.
            score += 1
        if ((ep["max_val"] > 150) and (ep["max_val"]<1000)):
            score += 1
        if (ep["orientation_angle"] > 85):
            score += 1
        if(abs(ep["first_third_mean"]) < 200):
            score += 1
        if (abs(ep["second_third_mean"]) < 120):
            score += 1
        if (abs(ep["third_third_mean"]) < 200):
            score += 1

        if (ep["area"] < 100):
            score = 0

        score_distribution[score] += 1

        if (score > highest_score):
            highest_score = score
            cnt_endpoints = ep
            best_index = i

        if(score>stop_early_threshold):
            break

        i += 1

    return cnt_endpoints, highest_score, score_distribution, best_index


def find_phantom_in_contours_using_ML_model(contours: list, original_img: np.ndarray, thresholded_img: np.ndarray,
                                            loaded_model, finish_early_acceptance_threshold=0.95):
    '''
    Finds the most likely phantom contour out of a list of contours.
    :param contours: The list of contours.
    :param original_img: The image pixelarray.
    :param thresholded_img: The thresholded binary pixelarray.
    :param loaded_model: The ML classifier model.
    :param finish_early_acceptance_threshold: A value that will, when surpassed by the predictive value, allow early cancelling of loop.
    :return: The endpoints of the phantom contour, its prediction value, and its index.
    '''

    i = 0
    highest_pred = 0
    highest_ep = None
    highest_index = i

    #Get score for each contour on whether it is the phantom and return the contour with the highest score.
    for cnt in contours:
        ep = get_endpoints_from_contour(cnt, original_img, thresholded_img, hierarchy=None)
        ep["contour no"] = i

        # create array in right shape to be read in by the ML model:
        # upper_y	lower_y	left_x	right_x	mean_val	mean_val_thresholded	centroid_x	centroid_y	area	bounding_box_area	aspect_ratio	extent	min_val	max_val	orientation_angle	contour no
        property_list = [ep["upper_y"], ep["lower_y"], ep["left_x"], ep["right_x"], ep["mean_val"],
                         ep["mean_val_thresholded"], ep["centroid_x"], ep["centroid_y"], ep["area"],
                         ep["bounding_box_area"], ep["aspect_ratio"], ep["extent"], ep["min_val"], ep["max_val"],
                         ep["orientation_angle"], ep["contour no"], ep["first_third_mean"], ep["second_third_mean"], ep["third_third_mean"]]
        X = np.array(property_list)
        predictions = loaded_model.predict_proba(X.reshape(1, -1))
        prediction = predictions[0, 1]  # get positive prediction chance

        if (prediction > finish_early_acceptance_threshold):  # finish early
            return ep, prediction, i

        if (prediction > highest_pred):
            highest_pred = prediction
            highest_ep = ep
            highest_index = i

        i += 1
    return highest_ep, highest_pred, highest_index


##############################################################################
# BELOW IS LEGACY CODE USED FOR TESTING ETC.
##############################################################################
def show_image(img: np.ndarray, time=8000):
    '''
    Shows an image representation of an array.
    :param img: The array to be displayed as image.
    :type img: array
    :param time: The time in ms the window should appear.
    :return:
    '''
    window1 = cv2.namedWindow("w1")
    cv2.imshow(window1, img)
    cv2.waitKey(time)
    cv2.destroyWindow(window1)


def save_image_to_csv(img: np.ndarray, name: str = "savedImage"):
    '''
    Saves an image array to a csv file.
    :param img: The image array.
    :param name: Name of the csv file to be created.
    :return:
    '''
    df = pd.DataFrame(img)
    df.to_csv(name + ".csv")


def find_all_dicom_files(folder_directory: str):
    '''
    Creates a list of all dicom file paths within a directory.
    :param folder_directory: The directory which is to be searched.
    :return: A list of all dicom file paths found.
    '''
    list_of_file_paths = []
    for root, dirs, files in os.walk(folder_directory):
        for filename in files:
            if ".dcm" in filename or ".DCM" in filename:  # exclude non-dicoms, good for messy folders
                source_path = os.path.join(root, filename)
                list_of_file_paths.append(source_path)
    return list_of_file_paths


def create_database_from_dicoms(folderpath: str):
    '''
    Loop over all dicoms in a folder and generate csv with contour data. This is, so we can find thresholds that we can apply to identify our phantom amongst other contour objects.
    :param folderpath: The folder holding all dicom related directories.
    :return: A dataframe holding contour data.
    '''
    # create list of dicom pathnames

    import dicom_reader #this is only legacy, and importing this only for testing in pycharm will prevent crashes when calling from console.

    paths = find_all_dicom_files(folderpath)
    if len(paths) < 1:
        return pd.DataFrame()

    contour_endpoints = []

    fileno=1
    for path in paths:
        print("File {} out of {}".format(fileno, len(paths)))
        try:
            img = dicom_reader.read_and_decompress_dicom_file(path)
            if img.get('pixel_array') is not None:
                img = create_adjusted_image(img)
                img_th = create_thresholded_image(img)
                contours, hierarchy = find_contours(img_th)

                i = 0

                for contour in contours:
                    img_contour_endpoints = get_endpoints_from_contour(contour, img, img_th, hierarchy)
                    img_contour_endpoints["file"] = path
                    img_contour_endpoints["contour no"] = i
                    contour_endpoints.append(img_contour_endpoints)
                    i += 1
            else:
                print(path, " does not hold pixelarray values. Cannot read.")
        except Exception as e:
            print("An error occured with file ", path)
            print("Error output :", e)

    df = pd.DataFrame(contour_endpoints)
    return df



