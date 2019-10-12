# Copyright (c) 2019 Team MODAP
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import types
from PIL import Image

class Corpus(object):
    """
    General manager and handler to manipulate image datasets to train
    """
    
    def __init__(self):
        self.inputs = []
        self.groundTruths = []
        self.meta = []

    def load_dir(self, images_dir:str, groundtruth_dir:str) -> None:
        """load_dir
        Loads directory of images to a ground truth document 
        
        :param images_dir:
        :type images_dir: str
        :param groundtruth_dir:
        :type groundtruth_dir: str
        :rtype: None
        """

        with open(groundtruth_dir, "r") as df:
            for line in df:
                truthArray_raw = line.strip().split(" ")
                truthArray = []
                for item in truthArray_raw:
                    try:
                        item_cast = int(item)
                    except ValueError:
                        item_cast = item.strip('"')
                    truthArray.append(item_cast)   
                self.groundTruths.append(truthArray[1:5])
                self.meta.append(truthArray[6:])
        breakpoint() 
