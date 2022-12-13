# nimmt image und gibt als endpoint obere linke Koordinate und untere rechte von Bounding box
from pydicom import FileDataset

from opencv_phantom_detection.opencv_phantom_detector_util import create_adjusted_image, create_thresholded_image, find_contours, \
    find_phantom_in_contours, find_phantom_in_contours_using_ML_model
import pickle


class Phantom_detector:

    def __init__(self, top_cutoff=256, side_cutoff=100, threshold=-100, mode: str="manual_score", score_threshold = 9,
                 prediction_model_path: str ="logistic_regression_model_for_contour_classification_if_phantom.sav",
                 probability_threshold=0.9):
        '''
        Creates a new phantom detector object, that can find phantoms in dicom images.
        :param top_cutoff: Top cropping of image in pixels. Careful when playing around with this, as x and y values go into the ML model without correction.
        :param side_cutoff: Image will be cropped on both sides by side_cutoff pixels. Careful when playing around with this, as x and y values go into the ML model without correction.
        :param threshold: The thresholding value to be used when creating binary images.
        :param mode: Which contour classifier should be used. Can be "manual_score" for better functioning score approach, or "ML_score" for Machine Learning approach.
        :param score_threshold: The score, which a given contour needs to achieve in the manual model, in order to be classified "phantom".
        :param prediction_model_path: The path to the ML model used to classify our contours.
        :param probability_threshold: The threshold that will need to be outperformed by a given phantom contour in our ML model. (most useful in logistic regression)
        '''
        self.top_cutoff = top_cutoff
        self.side_cutoff = side_cutoff
        self.threshold = threshold
        self.mode = mode
        self.score_threshold = score_threshold
        self.probability_threshold = probability_threshold
        # Load in ML model
        if mode == "ML_score":
            self.loaded_model = pickle.load(open(prediction_model_path, "rb"))
        pass

    def find_phantom(self, dcm: FileDataset):
        '''
        Will find the correct phantom contours in an image and return upper left and lower right coordinates of the bounding box.
        :param dcm: The dicom image as read in by pydicom
        :param score_threshold: The threshold to be exceeded. Maximum score is 9.
        :return: Contour list, index of best scoring contour, and coordinates of the phantom bounding box in order left_x, right_x, upper_y, lower_y. Returns None if score is below a given score threshold.
        '''

        # create images #######

        img = create_adjusted_image(dcm, self.top_cutoff, self.side_cutoff)

        thresholded_img = create_thresholded_image(img, self.threshold)

        # find contours ######
        contours, hierarchy = find_contours(thresholded_img)

        if len(contours) < 1:
            print("No contours found.")
            return None

        # Decide on correct one for phantom ####

        if self.mode == "manual_score":
            ep, score, score_distribution, best_index = find_phantom_in_contours(contours, img, thresholded_img)#, filename, save_contour_data_to_csv)
        #print(score_distribution, " Score: ", score)
        elif self.mode == "ML_score":
            ep, prediction, best_index = find_phantom_in_contours_using_ML_model(contours, img, thresholded_img,self.loaded_model)
        else:
            print("Need to select either mode= 'manual_score' or mode= 'ML_score'")

        # adjust for cropping ####
        left_x, right_x, upper_y, lower_y = (
        ep["left_x"] + self.side_cutoff, ep["right_x"] + self.side_cutoff, ep["upper_y"] + self.top_cutoff,
        ep["lower_y"] + self.top_cutoff)


        if self.mode == "manual_score":
            if score < self.score_threshold:
                return None
        elif self.mode == "ML_score":
            if prediction < self.probability_threshold:
                return None

        return contours, best_index, left_x, right_x, upper_y, lower_y  # , best_index

    def set_threshold(self, threshold):
        '''
        adjusts threshold of phantom detector object.
        :param threshold: The threshold to be set.
        :return:
        '''
        self.threshold = threshold
