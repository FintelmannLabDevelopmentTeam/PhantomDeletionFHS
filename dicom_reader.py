import pydicom
from pydicom.errors import InvalidDicomError


def read_and_decompress_dicom_file(source_path):
    try:
        ds = pydicom.read_file(source_path)
    except IOError:
        print ('IOError reading %s' % source_path)
        return None
    except InvalidDicomError:  # DICOM formatting error
        print ('InvalidDicomError reading %s' % source_path)
        return None
    except:
        print ('Unkown Error reading %s' % source_path)
        return None

    try:
        # Decompresses Pixel Data and modifies the Dataset in-place.
        ds.decompress()
    except NotImplementedError:
        print('NotImplementedError reading %s' % source_path)
        return None
    except:  # DICOM formatting error
        print ('Unkown Error reading %s' % source_path)
        return None

    return ds