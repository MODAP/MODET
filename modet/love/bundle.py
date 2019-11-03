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


import os
import json
import math
import uuid
import types
import pickle
import natsort
import numpy as np
from PIL import Image

from .. import utils 


class Corpus(object):
    """
    General manager and handler to manipulate image datasets to train
    """
    
    def __init__(self, savedir="", name=str(uuid.uuid4())[-8:]):
        self.groundTruths = [] 
        self.meta = []
        self.name = name
        self.savedir = os.path.join(savedir, self.name+".mocorp")
        os.makedirs(os.path.join(self.savedir, "modet_love_data"), exist_ok=True)
        os.makedirs(os.path.join(self.savedir, "modet_love_data", "inputs"), exist_ok=True)
        with open(os.path.join(self.savedir, "corpus.conf"), "w") as df:
            df.write(json.dumps({"name":self.name, "savedir":savedir}))

    @classmethod
    def open(cls, savedir:str)->object:
        """open
        Opens a saved corpus
        :param savedir: the save directory
        :type savedir: str
        :rtype: Corpus
        """
        with open(os.path.join(savedir, "corpus.conf"), "r") as df:
            db = json.loads(df.read())
            name = db["name"]
            sd = db["savedir"]
        with open(os.path.join(savedir, "modet_love_data", "meta"), "rb") as df:
            meta = pickle.load(df)
        with open(os.path.join(savedir, "modet_love_data", "truths"), "rb") as df:
            truths = pickle.load(df)
        c = cls(sd, name) 
        c.groundTruths = truths
        c.meta = meta
        with open(os.path.join(savedir, "corpus.conf"), "w") as df:
            df.write(json.dumps(db))

        return c

    @property
    def __truths_dir(self):
        return os.path.join(self.savedir, "modet_love_data", "truths")

    @property
    def __meta_dir(self):
        return os.path.join(self.savedir, "modet_love_data", "meta")

    @property
    def __input_files_dir(self):
        return os.path.join(self.savedir, "modet_love_data", "inputs")

    @property
    def images(self):
        imgs = os.listdir(self.__input_files_dir)
        for i in imgs:
            with open(os.path.join(self.__input_files_dir, i), "rb") as df:
                img = pickle.load(df)
                yield img
    
#     def load_dirs(self, imagedirs_dir:str, groundtruths_dir:str, ext:str=".txt", truth_downsampling_correction:float=1) -> None:
        # """load_dirs
        # Gets images in folders in images_dir and matches them with ground truths, then loads them
        
        # :param imagedirs_dir: directory where directories of frames are located
        # :type imagedirs_dir: str
        # :param groundtruths_dir: directory where files of ground truths are located
        # :type groundtruths_dir: str
        # :param ext: groundtruth file extension
        # :type ext: str
        # :param truth_downsampling_correction: corrects the downsampling (if any) with a factor of... (default is 1 — no correction.)
        # :type truth_downsampling_correction: float
        # :rtype: None
        # """
        
        # imgDirs = natsort.natsorted(os.listdir(imagedirs_dir)) 
        # for i, imgDir in enumerate(imgDirs):
            # print("Loading frame set", i, "of", len(imgDirs))
            # fullImgDir = os.path.join(imagedirs_dir, imgDir)
            # fullTruthDir = os.path.join(groundtruths_dir, imgDir+ext)
            # self.load_dir(fullImgDir, fullTruthDir, truth_downsampling_correction)

    def load_dir(self, images_dir:str, groundtruth_dir:str, truth_downsampling_correction:float=1) -> None:
        """load_dir
        Loads directory of images to a ground truth document 
        
        :param images_dir: directory where frames are located
        :type images_dir: str
        :param groundtruth_dir: file containing ground truths
        :type groundtruth_dir: str
        :param truth_downsampling_correction: corrects the downsampling (if any) with a factor of... (default is 1 — no correction.)
        :type truth_downsampling_correction: float
        :rtype: None
        """

        # Opens and parses the groundtruth file
        with open(groundtruth_dir, "r") as df:
            for line in utils.progressbar(df.readlines(), "Truths Loading: "):
                truthArray_raw = line.strip().split(" ")
                truthArray = []
                # Loads ground truths and casts the appropriate items to int
                for item in truthArray_raw:
                    try:
                        # Try to cast to int, downsample as needed
                        item_cast = int((float(item)/truth_downsampling_correction))
                    except (ValueError, TypeError) as e:
                        item_cast = item.strip('"')
                    # Appends the cast value to truths
                    truthArray.append(item_cast) 
                # Gets the index of the frame
                frame = truthArray[5]
                # Appends the found truths to the class-wide array containing the ground truths
                while True:
                    try:
                        self.groundTruths[frame].append(truthArray[1:5])
                        break
                    except IndexError:
                        self.groundTruths.append([])
                        self.groundTruths[frame].append(truthArray[1:5])
                        continue
                while True:
                    try:
                        self.meta[frame].append(truthArray[6:])
                        break
                    except IndexError:
                        self.meta.append([])
                        self.meta[frame].append(truthArray[6:])
                        continue
        # Saves the truths and metadata
        with open(self.__meta_dir, "wb") as df:
            pickle.dump(self.meta, df) 
        with open(self.__truths_dir, "wb") as df:
            pickle.dump(self.groundTruths, df) 
        # Sorts the image directory by frame
        imgDirs = natsort.natsorted(os.listdir(images_dir)) 
        imgIndex = 0
        # Opens each frame file
        for url in utils.progressbar(imgDirs, "Images Loading: "):
            # Opens the file as image
            im = Image.open(os.path.join(images_dir, url))
            imageArray = []
            for pixel in im.getdata():
                # Appends each pixel [r, g, b] to imageArray
               imageArray.append(list(pixel)) 
            # Adds the image to the class-wide array
            with open(os.path.join(self.__input_files_dir, str(imgIndex)), "wb") as df:
                pickle.dump(imageArray, df)
            imgIndex += 1

class CorpusManager(object):
    """
    """

    def __init__(self, corpus:Corpus, anchor_factor=40):
        self.corpus = corpus
        with open(os.path.join(corpus.savedir, "corpus.conf"), "r") as df:
            db = json.loads(df.read())
            if db.get("ismanaged"):
                with open(os.path.join(corpus.savedir, "modet_love_data", "anchored_truths"), "rb") as df:
                    data = pickle.load(df)
                    self.__compiled_output_data = data[0]
                    self.__is_compiled = True
                    self.anchor_factor = data[1]
            else:
                self.anchor_factor = anchor_factor
                self.__is_compiled = False

    @property
    def __anchors(self):
        x_anchorpoints = [i*self.anchor_factor for i in list(range(int((1280/self.anchor_factor)+1)))]
        y_anchorpoints = [i*self.anchor_factor for i in list(range(int((720/self.anchor_factor)+1)))]
        anchors = [list((i, j) for j in y_anchorpoints) for i in x_anchorpoints]
        return [i for sublist in anchors for i in sublist]

    def __assign_anchor(self, x, y, a, b):
        # x = x_1, y = y_1, a = x_2, b = y_2
        anchors = self.__anchors
        avg = (((a+x)/2), ((y+b)/2)) 
        distances = {}
        for point in anchors:
            distances[point] = math.sqrt((point[0]-avg[0])**2+(point[1]-avg[1])**2)
        return min(distances, key=distances.get) 

    def __inject_corpus(self):
        with open(os.path.join(self.corpus.savedir, "corpus.conf"), "r+") as df:
            db = json.loads(df.read())
            db["ismanaged"] = True
            df.seek(0)
            df.write(json.dumps(db))
            df.truncate()
        with open(os.path.join(self.corpus.savedir, "modet_love_data", "anchored_truths"), "wb") as df:
            pickle.dump([self.outputs, self.anchor_factor], df)
            
    @property
    def inputs(self):
        return np.reshape(np.array(list(self.corpus.images)), (-1, 1280, 720, 3))

    @property
    def inputs_gen(self):
        for i in self.corpus.images:
            yield np.array(i).reshape(1280, 720, 3)

    @property
    def outputs(self):
        if not self.__is_compiled:
            raise ValueError("Please call CorpusManager.compile() to compile this corpus.")
        return np.array(self.__compiled_output_data)

    @property
    def outputs_gen(self):
        if not self.__is_compiled:
            raise ValueError("Please call CorpusManager.compile() to compile this corpus.")
        for i in self.__compiled_output_data:
            yield np.array(i)

    def compile(self):
        assert not self.__is_compiled, "This corpus has already been compiled!"
        anchors = self.__anchors
        self.__compiled_output_data = []
        for frame in utils.progressbar(self.corpus.groundTruths, "Parsing truths: "):
            anchor_template = [[]]*len(anchors)
            anchor_indxs = [anchors.index(self.__assign_anchor(i[0], i[1], i[2], i[3])) for i in frame]
            for box, index in zip(frame, anchor_indxs):
                anchor_template[index] = [box]
            self.__compiled_output_data.append(anchor_template)
        self.__is_compiled = True
        self.__inject_corpus()

