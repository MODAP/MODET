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

import modet
import pickle
from modet.love import bundle
from modet.brain import squeezedet


c = bundle.Corpus.open("/Users/houliu/Nextcloud/Documents/Projects/MODET/love_corpora/1-1-1.mocorp")
cm = bundle.CorpusManager(c, batch_size=1)

net = squeezedet.SqueezeDet()

net.fit(cm)

# c = bundle.Corpus(name="1-1-1", savedir="love_corpora")
# c.load_dir("./corpusraw.donotsync/Drone1/1.1.1", "./corpusraw.donotsync/Labels/SingleActionLabels/3840x2160/1.1.1.txt")
# c.load_dirs("./corpusraw.donotsync/Drone1/", "./corpusraw.donotsync/Labels/SingleActionLabels/3840x2160/", truth_downsampling_correction=3)

# im = Image.open("./corpusraw.donotsync/Drone1/1.1.1/0.jpg")
# imGen = im.getdata()
# 
# for pixel in imGen:
    # print(pixel)

