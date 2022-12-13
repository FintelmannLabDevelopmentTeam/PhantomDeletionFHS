import os

import cv2
import pandas as pd
import numpy as np
from dicom_reader import read_and_decompress_dicom_file
from opencv_phantom_detection import opencv_phantom_detector as opd
import time

import threading

#When writing to or reading from an encrypted drive, make sure that it is not autolocked by the system at some point. Otherwise, the program might continue running, but cannot save the files, or will just encounter strange errors.


def detect_and_delete_phantoms(src_dir: str, target_dir: str, deviation_threshold=10, show_preview=False, msg_every_n_files=500, backup_after_n_errors=1000, max_thread_number=1):
    '''
    Loops over all files in a directory, and, for each one, opens a new thread that tries to delete the phantom, and saves them to another directory. Errors that may occur are categorized and stored in a csv file.
    Args:
        src_dir: The source directory of all the files that are to be edited
        target_dir: The target directory, to which all the files should be saved.
        deviation_threshold: If more than one algorithm is applied, this determines how far away their corresponding coordinates are allowed to lie without interrupting and saving a file as error.
        show_preview: Whether a window should show, what the algorithm is doing, simultaneously.
        msg_every_n_files: After the handling of how many files textual updates should be given to the console.
        backup_after_n_errors: After the handling of how many files the error CSV should be updated.
        max_thread_number: How many threads are allowed to run at a single timepoint.

    Returns:

    '''
    print("Starting run.")
    handled_files = find_all_dicoms_in_target_dir(TARGET_DIR)
    n_previously_handled_files = len(handled_files)
    files_to_handle = find_all_dicoms_in_target_dir(SRC_DIR)
    n_files_to_handle = len(files_to_handle)
    failed_files = []

    #already_handled = pd.read_csv("already_handled_files.csv")

    th2 = None

    if show_preview:
        window1 = cv2.namedWindow("Phantom Deleter")
    else:
        window1 = None

    tic = time.perf_counter()
    n = 0
    thread_list = []
    previously_handled = 0

    detector_opencv = opd.Phantom_detector()

    #loop over all dicom files in directory
    for source_path in files_to_handle:


        if source_path.replace(src_dir, target_dir) in handled_files:
            #print("already deleted phantom from ", source_path)
            previously_handled +=1

        else:

            #This should make sure that only n threads are running at once
            if(len(thread_list)==max_thread_number):
                for thread in thread_list:
                    thread.join()
                    del(thread)
                thread_list.clear()

            #Run calculation for the loaded dicom in separate thread.
            th = threading.Thread(target=get_position_and_handle_errors_or_delete, args=(source_path, detector_opencv,
                                                                                         src_dir, target_dir, failed_files, show_preview, window1, deviation_threshold))
            th.start()
            thread_list.append(th)






        #for every n files send message file list, so when algorithm fails for some reason, a backup is there to give us an advanced starting point
        if n % msg_every_n_files == 0:
            toc = time.perf_counter()
            print(f"Started handling threads for {n} out of {n_files_to_handle} files in {toc - tic:0.4f} seconds. {n-previously_handled} files have actually been opened for editing (incl. failed files). {previously_handled} Files had already been edited before by previous runs of this algorithm.")
            print(f"Encountered {len(failed_files)} Errors")

            n_actually_handled_files = n-previously_handled
            if n_actually_handled_files == 0:
                time_per_file = 0
            else:
                time_per_file = (toc - tic)/n_actually_handled_files
            files_left = n_files_to_handle-(n_previously_handled_files-previously_handled+n)
            time_left = time_per_file * files_left
            print(f"At current speed, the algorithm will need {(time_left/60/60):0.3f} hours to finish. There are {files_left} files left to actually calculate, at a past average speed of approx. {time_per_file:0.4f} seconds per file editing.\n")

            # store file as failed together with the reasons for failing.

        if ((len(failed_files)+1) % backup_after_n_errors == 0):
            try:
                print(f"Reached {len(failed_files)} errors. Saving.")
                pd.DataFrame(failed_files).to_csv("failed_files.csv")
            except:
                print("Errors could not be saved. Failure in accessing csv file.")


        n+=1


    #wrap up in the end
    toc = time.perf_counter()
    for thread in thread_list:
        thread.join()
        del (thread)
    print(f"Handled {n} files in {toc - tic:0.4f} seconds.")
    print(f"Encountered {len(failed_files)} Errors")
    print("Done with all files.")

    try:
        print(f"Reached {len(failed_files)} errors. Saving.")
        pd.DataFrame(failed_files).to_csv("failed_files.csv")
    except:
        print("Errors could not be saved. Failure in accessing csv file.")
    if show_preview:
        cv2.destroyWindow(window1)



def save_file(new_file_path: str, dcm):
    '''
    Saves a dicom file.
    Args:
        new_file_path: the path to the location of the newly stored file.
        dcm: The dicom file.

    Returns:

    '''
    os.makedirs(new_file_path[0:new_file_path.rfind('\\')], exist_ok=True)
    dcm.save_as(new_file_path)

def load_file(source_path: str, error, errormsg):
    '''
    Loads in a dicom file. If no slope and intercept, returns 0 and 1, and an error message.
    Args:
        source_path: The path from which the file will be loaded.
        error: A boolean. True, if error happened. False, if not.
        errormsg: a dictionary holding the errormsg for a given file.

    Returns: The dicom file, its rescale intercept and its slope, if available. Otherwise returns 0 and 1 for the two ladder. Also, returns the error, and errormsg.

    '''
    dcm = read_and_decompress_dicom_file(source_path)
    try:
        rescale_intercept = dcm.RescaleIntercept
        rescale_slope = dcm.RescaleSlope
    except:
        print("No rescale parameters found for file ", source_path)
        rescale_intercept, rescale_slope = (0, 1)
        error = True
        errormsg["file"] = source_path
        errormsg["no_rescale_params"] = True
    return dcm, rescale_intercept, rescale_slope, error, errormsg

def get_position_and_handle_errors_or_delete(source_path, detector_opencv,
                                             src_dir, target_dir, failed_files, show_preview, window1,
                                              deviation_threshold):
    '''
    Loads in a dicom, tries to find the phantom, and to delete it. Performs error handling, if any of the steps fail. In that case, an errormessage is generated and added to an ongoing csv file.
    Args:
        source_path: The path from which to load the dicom file.
        detector_opencv: The openCV algorithm detector object.
        src_dir: The folder holding all the dicoms. Needed to determine final path.
        target_dir: The folder where all generated dicoms with deleted phantom should be stored. Needed to determine final path.
        failed_files: A list of all failed files with corresponding error messages.
        show_preview: True, if a little window should show how the phantom is being detected.
        window1: The opencv window, which should show the images.
        deviation_threshold: If more than one algorithm is applied, this determines how far away their corresponding coordinates are allowed to lie without interrupting and saving a file as error.

    Returns:

    '''
    error = False
    errormsg = {}

    # load file
    try:
        dcm, rescale_intercept, rescale_slope, error, errormsg = load_file(source_path, error, errormsg)
    except:
        error = True
        errormsg["file"] = source_path
        errormsg["dcm_load_failed"] = True
        print("Could not load dicom file ", source_path)

    if not error:

        contours, best_index, error, errormsg = get_phantom_pos(dcm, source_path, detector_opencv,
                                                                         errormsg, error, rescale_intercept, rescale_slope, deviation_threshold)


    if not error:
        if show_preview:
            img = np.uint8(dcm.pixel_array)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            cv2.drawContours(img, contours, best_index, color=(0,255, 0), thickness=2, offset=(100,256))

            cv2.imshow(window1, img)
            cv2.waitKey(1)

        # delete Phantom
        try:

            cv2.drawContours(dcm.pixel_array, contours, best_index, color=dcm.SmallestImagePixelValue, thickness=-1,
                             offset=(100, 256))


            # Save file in new thread
            new_file_path = source_path.replace(src_dir, target_dir)
            save_file(new_file_path, dcm)



        except:
            print("Error deleting phantom from file ", source_path)
            error = True
            errormsg["file"] = source_path
            errormsg["deletion_failed"] = True

    if error:
        failed_files.append(errormsg)  # this is where an error might occur, when two threads try to access the failed_files list at once. Hopefully, that will just not happen.





def get_phantom_pos(dcm, source_path, detector_opencv, errormsg, error, rescale_intercept, rescale_slope, deviation_threshold):
    '''
    Calculates the most probable location of the phantom bounding box.
    Args:
        dcm: The dicom file to be edited.
        source_path: The path to that dicom file.
        detector_opencv: The openCV algorithm detector object.
        errormsg: An ongoing python dictionary holding a classification of errors, that occured.
        error:  True, if an error occured.
        rescale_intercept: The rescale intercept. Needed to calculate true HU values.
        rescale_slope: The rescale slope. Needed to calculate true HU values.
        deviation_threshold: If more than one algorithm is applied, this determines how far away their corresponding coordinates are allowed to lie without interrupting and saving a file as error.

    Returns: (OpenCV only in algorithms: returns contours and best_index, and error and errormsg). Otherwise returns the bounding box coordinates of the phantom in the form upper_y, lower_y, right_x, left_x. Returns none for these 4, if an error occurs during calculation. Also returns the (altered) error boolean and errormsg.

    '''
    # Apply opencv detection algorithm

    try:
        contours, best_index, opencv_left_x, opencv_right_x, opencv_upper_y, opencv_lower_y = detector_opencv.find_phantom(dcm)
    except:
        #print("OpenCV algorithm encountered problem with file: ", source_path)
        error = True
        errormsg["file"] = source_path
        errormsg["opencv_algorithm_failed"] = True


    # remove phantom
    if not error:

        return contours, best_index, error, errormsg

    else:
        return None, None, error, errormsg


def find_all_dicoms_in_target_dir(target_dir):
    '''
    Creates a list of all dicom files within a directory.
    Args:
        target_dir: The directory to be searched for dcm files.

    Returns: A list of all the files.

    '''
    all_files = []

    for root, dirs, files in os.walk(target_dir):
        for filename in files:
            if ".dcm" in filename or ".DCM" in filename:  # exclude non-dicoms, good for messy folders
                source_path = os.path.join(root, filename)
                all_files.append(source_path)
    return all_files

if __name__ == '__main__':
    SRC_DIR = r""
    TARGET_DIR = r""
    detect_and_delete_phantoms(SRC_DIR, TARGET_DIR, show_preview=False)
