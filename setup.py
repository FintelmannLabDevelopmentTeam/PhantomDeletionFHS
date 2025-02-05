from setuptools import setup

setup(name='phantom_deleter',
      version='0.0.1',
      description='A script to identify and remove imaging phantoms in DICOM images.',
      author='Paul Erik Tonnesen & Wael Amayri',
      author_email='erik@tonnesen.de & wamayri@mgh.harvard.edu',
      packages=['opencv_phantom_detection'],
      install_requires=['numpy',
                        'pandas<=1.4.0',
                        'opencv-python',
                        'pysimplegui',
                        'scikit-image',
                        'scikit-learn',
                        'pydicom',
                        'cython',
                        'python-gdcm',
                        'pillow',
                        'pylibjpeg>=1.2',
                        'pylibjpeg-libjpeg',
                        'pylibjpeg-openjpeg',
                        'pylibjpeg-rle'
                        ])
