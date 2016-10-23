#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2016 Sensetime, CUHK
# Written by Yang Bin, Wang Kun
# Modified by Kai KANG
# --------------------------------------------------------

"""Test a Fast R-CNN network on an image database."""

import argparse
import pprint
import time
import os
import sys
import cPickle
import numpy as np
import _init_paths
import caffe
from fast_rcnn.craft import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--def', dest='prototxt',
                        help='prototxt file defining the network',
                        default=None, type=str)
    parser.add_argument('--net', dest='caffemodel',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--wait', dest='wait',
                        help='wait until net file exists',
                        default=True, type=bool)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to test',
                        default='vid_2015_val_20', type=str)
    parser.add_argument('--start_idx',
                        help='Start index for testing. [0]',
                        default=0, type=int)
    parser.add_argument('--stop_idx',
                        help='Stop index for testing. [Inf]',
                        default=np.inf, type=int)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--vis', dest='vis', help='visualize detections',
                        action='store_true')
    parser.add_argument('--num_dets', dest='max_per_image',
                        help='max number of detections per image',
                        default=100, type=int)
    parser.add_argument('--num_per_batch', dest='boxes_num_per_batch',
                        help='split boxes to batches',
                        default=0, type=int)
    parser.add_argument('--bbox_mean', dest='bbox_mean',
                        help='the mean of bbox',
                        default=None, type=str)
    parser.add_argument('--bbox_std', dest='bbox_std',
                        help='the std of bbox',
                        default=None, type=str)
    parser.set_defaults(vis=False)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print 'Called with args:'
    print args

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print 'Using config:'
    pprint.pprint(cfg)

    while not os.path.exists(args.caffemodel) and args.wait:
        print 'Waiting for {} to exist...'.format(args.caffemodel)
        time.sleep(10)

    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)
    net.name = os.path.splitext(os.path.basename(args.caffemodel))[0]

    # apply bbox regression normalization on the net weights
    with open(args.bbox_mean, 'rb') as f:
        bbox_means = cPickle.load(f)
    with open(args.bbox_std, 'rb') as f:
        bbox_stds = cPickle.load(f)

    net.params['bbox_pred_vid'][0].data[...] = \
        net.params['bbox_pred_vid'][0].data * bbox_stds[:, np.newaxis]

    net.params['bbox_pred_vid'][1].data[...] = \
        net.params['bbox_pred_vid'][1].data * bbox_stds + bbox_means

    imdb = get_imdb(args.imdb_name)
    imdb.competition_mode(True)
    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    test_net(net, imdb, max_per_image=args.max_per_image,
            boxes_num_per_batch=args.boxes_num_per_batch,
            vis=args.vis, st=args.start_idx, ed=args.stop_idx)
