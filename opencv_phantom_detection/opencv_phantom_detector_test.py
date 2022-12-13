import cv2
import numpy as np
import pandas as pd

import opencv_phantom_detector_util as util
from dicom_reader import read_and_decompress_dicom_file
from opencv_phantom_detector import Phantom_detector as pdetec


#CAVE: THIS FILE WAS ONLY CREATED TO TEST AND PLAY AROUND WITH THE ALGORITHM, AS WELL AS TO CREATE PHANTOM CONTOUR GROUND TRUTH DATASETS. IT IS PRONE TO CRASHES AND NOT PROPERLY REFACTORED. ASK Paul Erik Tonnesen IF YOU HAVE QUESTIONS ABOUT IT.


def create_contour_database_for_training_and_find_correct_phantom_contours(filepath: str):
    '''
    Will read in all dicom images from a filepath, create contours, and loop over each of them asking the user whether it contains the phantom or not.
    That way, a database is created containing contour data and classification whether it is holding a phantom or not. That can later be used to train decision algorithms (see ./contour_classifier)
    :param filepath: The folder to search for dicoms
    :return: 
    '''


    df = util.create_database_from_dicoms(filepath)
    df_only_phantoms = pd.DataFrame([])

    df.to_csv("no_annotation_Contours_all_threshold-100_topcutoff300_sidecutoff100.csv")

    files = df.file.unique()

    for file in files:
        dcm = read_and_decompress_dicom_file(file)
        img = dcm.pixel_array
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        contour = -1
        found = False

        detector = pdetec()
        window1 = cv2.namedWindow("w1")

        faulty_files = ["no file"]

        while not found:
            phantom = False
            if contour == -1:

                try:
                    x_left, x_right, y_upper, y_lower, best_index = detector.find_phantom(dcm)

                    row = df.loc[(df["file"] == file) & (df["contour no"] == best_index)]
                    phantom = True

                except:
                    print("Error detecting phantom automatically in file ", file)
                    x_left, x_right, y_upper, y_lower = (0, 0, 1, 1)

            else:
                row = df.loc[(df["file"] == file) & (df["contour no"] == contour)]

                try:
                    x_left = row["left_x"].values[0] + 100
                    y_upper = row["upper_y"].values[0] + 256
                    y_lower = row["lower_y"].values[0] + 256
                    x_right = row["right_x"].values[0] + 100
                except:
                    print("No more contours. Type skipFile")
                    x_left, x_right, y_upper, y_lower = (0, 0, 1, 1)

            plot_img = img.copy()
            cv2.rectangle(plot_img, (x_left, y_upper), (x_right, y_lower), (0, 0, 255), 1)

            cv2.imshow(window1, plot_img)
            cv2.waitKey(1)


            if phantom:
                df.loc[row.index, "is_phantom"] = 1
                df_only_phantoms.append(row)
                found = True

            else:
                usr = input("Phantom? y/n/skipFile")
                if usr == "y":
                    df.loc[row.index, "is_phantom"] = 1
                    df_only_phantoms.append(row)
                    found = True
                if usr == "skipFile":
                    faulty_files.append(file)
                    df.drop(df[df["file"] == file].index, inplace=True)
                    found = True
            contour += 1
    try:
        df.to_csv("new_annotated_Contours_all_threshold-100_topcutoff300_sidecutoff100.csv")
        cv2.destroyWindow(window1)
    except:
        pass





def test_algorithm(filepath, img_show_mode=True):
    faulty_files = []
    detector = pdetec()

    window1 = cv2.namedWindow("w1")

    files = util.find_all_dicom_files(filepath)


    for file in files:
        print("Reading in file: ", file)
        dcm = read_and_decompress_dicom_file(file)
        try:
            contours, best_index, left_x, right_x, upper_y, lower_y = detector.find_phantom(dcm)
            if (img_show_mode):
                img = util.create_adjusted_image(dcm, 0, 0)
                img = np.uint8(img)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                cv2.drawContours(img, contours, best_index, (0,255,0), thickness=1, offset=(100, 256))

                cv2.imshow(window1, img)
                cv2.waitKey(1)

        except:
            print("error finding phantom in file ", file)
            faulty_files.append(file)

            img = np.uint8(dcm.pixel_array)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            cv2.putText(img, "Failed:" + file, (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255))

            cv2.imshow(window1, img)
            cv2.waitKey(1)
            #input("press key to continue")

    cv2.destroyWindow(window1)
    pd.DataFrame(faulty_files).to_csv("Files where algorithm fails.csv")


if __name__ == "__main__":
    folderpath = r''
    test_algorithm(folderpath, True)
    #create_contour_database_for_training_and_find_correct_phantom_contours(folderpath)

