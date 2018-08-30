from ctypes import *
import math
import random
import pandas as pd
import time
from os.path import expanduser

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

lib = CDLL(expanduser("~") + "/darknet/libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms)

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                b.w=im.w if b.w > im.w else b.w
                b.h=im.h if b.h > im.h else b.h
                xmin=int(b.x-b.w/2)
                xmin=0 if xmin < 0 else xmin
                ymin=int(b.y-b.h/2)
                ymin=0 if ymin < 0 else ymin
                xmax=int(b.x+b.w/2)
                xmax=im.w if xmax > im.w else xmax
                ymax=int(b.y+b.h/2)
                ymax=im.h if ymax > im.h else ymax
                res.append((meta.names[i], dets[j].prob[i], (xmin, ymin, xmax, ymax)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res

# parameters configuration
names=pd.read_csv(expanduser("~") + "/LOC_synset_mapping.txt", sep='\t', header=None)
names[1]=names[0].str[10:]
names[0]=names[0].str[0:9]
ids=names.loc[:,0]
cfg_path=(expanduser("~") + "/ImageNet/yolov3-ILSVRC.cfg").encode()
weight_path=(expanduser("~") + "/darknet/backup/yolov3-ILSVRC.backup").encode()
meta_path=(expanduser("~") + "/ImageNet/ILSVRC.data").encode()
net = load_net(cfg_path, weight_path, 0)
meta = load_meta(meta_path)

# encapsulate prediction to a single function
def prediction(path):
    # r = detect(net, meta, str(path).encode(), .01)
    r = detect(net, meta, str(path).encode(), .005)
    # format prediction
    res=str()
    pred=[[str(names.loc[names[0]==str(val[0], "utf-8")[0:9]].index[0]+1) 
        + " " + " ".join(map(str, val[2]))] for val in r]
    if len(pred) > 0:
        res=' '.join([i[0] for i in pred[:5]])
    return res

# predict for all test data
sub=pd.read_csv(expanduser("~") + "/LOC_sample_submission.csv", sep=',')
for i in range(100000):
    img=expanduser("~") + "/ILSVRC/Data/CLS-LOC/test/ILSVRC2012_test_"+str("%08d" % (i+1))+'.JPEG'
    res=prediction(img)
    sub.loc[i, 'PredictionString']=res
    print(img,":\n", res)

# generate result file for submission
sub.to_csv(expanduser("~") + "/submissions/sub_"+str(int(time.time()*(10e5)))+".csv", index=False)