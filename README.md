# Master Thesis - Handwritten text cell detection in historical documents
<br>

#### Notebook author: Olle Dahlstedt
_Latest public version: 2021-03-10_
<br>

##### Tip for first time users:
Each "part" of the pipeline is contained within the scripts provided. If you would like to examine the code for each part, look at the import statements and follow along.

<br>
In case of non-familiarity with Python, find the documentation to all libraries used here:

1. OpenCV: https://docs.opencv.org/4.5.1/
2. Scikit-image: https://scikit-image.org/docs/stable/api/api.html
3. Numpy: https://numpy.org/doc/1.20/
4. Scipy: https://docs.scipy.org/doc/scipy/reference/
5. Matplotlib: https://matplotlib.org/stable/api/index.html

<br>
If problems occur with running the main.py script, please consult this README.md file.
Only contact the author if your problem is not solved by following the recommended installation instructions.

A simple summary of a recommended installation:

1. Set up a [virtual environment](https://docs.python.org/3/tutorial/venv.html) and make sure the libraries listed above are installed. For users of Anaconda, most packages, but not all, are included in the conda-base package.
#### IMPORTANT NOTICE
In order to use this program, the user must install the opencv-python-contrib package. <br>
There are conflicts with installing both opencv-python and opencv-python-contrib. <br>
Therefore, make sure to install the opencv-python-contrib package in a new virtual environment.


2. Clone this repository to your file system. <br>
   Run `git clone https://github.com/xqses/text-detection.git`
3. Set up your preferred image directory. <br>
   For the sake of simplicity, you should put the directory of images in the same directory as the main.py script. An example folder structure is the following:
    1. dev/text-detection/main.py
    2. dev/text-detection/test_img/IMG_{1, 2, ..., etc.}.JPG
    3. If you want any other folder structure then you must change the code manually to reflect this. Please see the ImageHelper, main and open_file functions.

4. A sample of 20 images are included in the Git repository.

5. Run the main.py script with the arguments os="win" or os="linux", and the optional arguments multi=True and n_img = your preferred # of images.
   <br>Run it from your command line or from your Python IDE.