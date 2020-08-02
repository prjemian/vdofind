#!/usr/bin/env python

"""
vdofind: find the program content in a recorded video

Hope to use machine learning to identify frames as:
  * program
  * other
Then, identify the transitions from other->program and program->other
by examining the ``is_program_frame`` probability as it
goes from low->high and high->low, respectively given a
series of frames.

Eventually, might become more sophisticated at identifying
other frame types such as black.  Few samples for those so
stick to the easy ones first.

* https://www.tensorflow.org/tutorials/keras/classification
* just two categories = ['program', 'other']
* Exclude a few seconds of frames at start, end, and transitions.
* Training & Testing
    * Non excluded frames will be used 
    * https://www.tensorflow.org/tutorials/keras/classification#train_the_model
    * Choose n (1k, 10k, 100k) frames at random
    * excluded frames will be in the validation set
    * Half the non-excluded frames will comprise the training set
    * Other half the non-excluded frames will comprise the test set
"""

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyRestTable
import time

categories = "keep discard validate".split()

recordings = {
    "../Soylent Green 2020-05-30 19-00 32-2.ts": {
        "rate": 29.97,          # frames per second
        "exclude_frames": 100,  # +/- 3.3s from event time
        "events": [
            # events: [hhmmss, category, description]
            # category: label index of frames before this time
            ["00:00:00", 1, "recording starts"],
            ["00:00:23", 1, "black screen"],
            ["00:00:24", 1, "producer credit"],
            ["00:00:31", 0, "black screen"],
            ["00:00:32", 0, "movie starts"],
            ["00:13:41", 0, "other block starts"],
            ["00:18:20", 1, "movie resumes"],
            ["00:33:14", 0, "other block starts"],
            ["00:37:29", 1, "movie resumes"],
            ["00:49:31", 0, "other block starts"],
            ["00:54:07", 1, "movie resumes"],
            ["01:07:49", 0, "other block starts"],
            ["01:12:34", 1, "movie resumes"],
            ["01:25:49", 0, "other block starts"],
            ["01:30:50", 1, "movie resumes"],
            ["01:45:44", 0, "other block starts"],
            ["01:49:14", 1, "movie resumes"],
            ["02:01:50", 0, "movie credits start"],
            ["02:03:52", 0, "other block starts"],
            ["02:04:59", 1, "recording ends"],
        ],
    }
}


def hhmmss2frame_num(hhmmss, rate):
    """convert hh:mm:ss into frame number"""
    h, m, s = map(int, hhmmss.split(":"))
    t = 60*(60*h + m) + s
    return round(t*rate)


def initial_assessment(changes, rate, exclusion, fr_max):
    fr_ref = 0
    blocks = []
    for hhmmss, category, description in changes:
        fr_num = hhmmss2frame_num(hhmmss, rate)
        fr_pre = max(0, fr_num-exclusion)
        fr_post = min(fr_max, fr_num+exclusion)
        if fr_ref != fr_pre:
            # train & test
            if fr_ref < fr_pre:
                blocks.append([fr_ref, fr_pre, categories[category]])

        # validate
        blocks.append([fr_pre, fr_post, "validate"])
        fr_ref = fr_post
    return blocks


def eliminate_overlapping_blocks(raw_blocks):
    """eliminate overlapping frame blocks: validate has highest priority"""
    blocks = []

    # find the validate blocks
    for fr_start, fr_end, category in raw_blocks:
        if category == "validate":
            if len(blocks) == 0:
                blocks.append([fr_start, fr_end, category])
            elif fr_start <= blocks[-1][1]:
                blocks[-1][1] = fr_end
            else:
                blocks.append([fr_start, fr_end, category])
    blocks = sorted(blocks)

    def add_block(block):
        # find the block to insert _after_
        for i, b in enumerate(blocks):
            if block[0] > b[0]:
                continue
            if block[0] < b[0]:
                i -= 1
                break

        e = min(block[1], b[0])
        b = blocks[i]
        s = max(block[0], blocks[i][1])
        blocks.insert(i+1, [s, e, block[2]])

    # add the other blocks
    for raw in raw_blocks:
        if raw[2] != "validate":
            add_block(raw)

    return sorted(blocks)


def categorize_by_blocks(changes, rate=None, exclusion=None):
    """
    Assign frames to one of the categories.
    """
    rate = rate or 29.97
    exclusion = exclusion or 45
    fr_max = max([hhmmss2frame_num(row[0], rate) for row in changes])
    raw_blocks = initial_assessment(changes, rate, exclusion, fr_max)
    blocks = eliminate_overlapping_blocks(raw_blocks)
    return sorted(blocks)


def machine_learning_model(model_size=None, train_frac=None):
    model_size = model_size or 10000    # number of frames to select
    train_frac = train_frac or 0.5      # fraction of selected frames used to train the net
    test_frac = 1.0 - train_frac        # fraction of selected frames used to test the net

    for k, r in recordings.items():
        print(f"Recording: {k}")
        r["blocks"] = categorize_by_blocks(
            r["events"], rate=r["rate"], exclusion=r["exclude_frames"])

        total_frames = sum([
            block[1] - block[0]
            for block in r["blocks"]
        ])

        def identify_category(fr_num):
            for b in r["blocks"]:
                if b[0] <= fr_num < b[1]:
                    return b[2]

        selected_frames = sorted([
            fr_num
            for fr_num in np.random.randint(0, total_frames, model_size)
            if identify_category(fr_num) != "validate"
        ])
        r["selected_frames"] = selected_frames

        # REPORTS

        # table = pyRestTable.Table()
        # table.labels = "fr_start fr_end category".split()
        # for fr_start, fr_end, category in r["blocks"]:
        #     table.addRow((fr_start, fr_end, category))
        # print(table)

        # summary = {k: 0 for k in categories}
        # for block in r["blocks"]:
        #     v = block[1] - block[0]
        #     summary[block[2]] += v

        # table = pyRestTable.Table()
        # table.labels = "category number_frames".split()
        # for k, v in summary.items():
        #     table.addRow((k, v))
        # table.addRow(("TOTAL", total_frames))
        # print(table)

        selected_count = len(selected_frames)
        selected_summary = {k: 0 for k in categories}
        for fr_num in selected_frames:
            selected_summary[identify_category(fr_num)] += 1

        table = pyRestTable.Table()
        table.labels = "category number_frames".split()
        for k, v in selected_summary.items():
            table.addRow((k, v))
        table.addRow(("TOTAL", selected_count))
        print(table)


if __name__ == "__main__":
    machine_learning_model(100)
