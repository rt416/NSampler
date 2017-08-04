""" Compute the average statistics from the given subjects list """

import csv
import os
import numpy as np
import tensorflow as tf
from numpy import genfromtxt
from train import get_output_radius
from common.utils import name_network, name_patchlib, set_network_config, define_checkpoint, mc_inference, dt_trim, dt_pad, clip_image, save_stats


def compute_stats(opt, subjects_list):
    # --------------------------- Get the network name ------------------------
    if not (os.path.exists(opt["recon_dir"])):
        os.makedirs(opt["stats_dir"])

    csv_dir = define_checkpoint(opt)
    nn_dir = name_network(opt)
    stats_dir = os.path.join(opt['stats_dir'], nn_dir)

    print("Compute statistics based on the following subjects:")
    for subject in subjects_list: print(subject)

    if os.path.exists(os.path.join(stats_dir, 'stats.csv')):
        get_summary_stats(csv_in=os.path.join(stats_dir, 'stats.csv'),
                          csv_out=os.path.join(stats_dir, 'summary_stats.csv'),
                          subjects_list=subjects_list)

    if os.path.exists(os.path.join(stats_dir, 'stats_brain.csv')):
        get_summary_stats(csv_in=os.path.join(stats_dir, 'stats_brain.csv'),
                          csv_out=os.path.join(stats_dir, 'summary_stats_brain.csv'),
                          subjects_list=subjects_list)


def get_summary_stats(csv_in, csv_out, subjects_list):

    # load the csv file as a numpy array:
    tmp = np.genfromtxt(csv_in, delimiter=',', dtype=None)

    # get the rows for specified subjects and stack them up.
    valid_rows = []
    for row in tmp:
        if row[0] in subjects_list:
            print("adding subject: " + row[0])
            valid_rows.append(row[1:])

    if len(valid_rows)!=len(subjects_list):
        print('some subjects are missing from the stats table.')
        return

    tbl = np.stack(valid_rows).astype(float)

    # get all the summary stats
    rows_new = []
    headers = ['summary'] + list(tmp[0, 1:])
    rows_new.append(headers)
    rows_new.append(['mean'] + list(tbl.mean(axis=0)))
    rows_new.append(['std'] + list(tbl.std(axis=0)))
    rows_new.append(['min'] + list(tbl.min(axis=0)))
    rows_new.append(['max'] + list((tbl.max(axis=0))))

    # save it to a csv file:
    print("Creating "+ csv_out)
    f = open(csv_out, 'wb')
    w = csv.writer(f)
    for row in rows_new:
        w.writerow(row)