#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create a list of all the timelapse videos we have from the existing 
assay dev image databases. This has several values at the start that might
need to be updated.
"""

import os
import numpy as np
import pandas as pd

# Config section
raw_dir = "/allen/aics/assay-dev/MicroscopyOtherData/ImageDatabasesByUser/"

print("FNs likely need to be manually selected from %s" % raw_dir)
fn_usr = [
    ("image_database_CCH_wip.xlsx", "CCH"),
    ("image_database_CLF_20190906.xlsx", "CLF"),
    ("image_database_MH.xlsx", "MH"),
    ("image_database_toedit_IAM_20190730.xlsx", "IAM"),
    ("image_database_toedit_JLG_newer.xlsx", "JLG"),
    ("image_database_toedit_RY_20190827.xlsx", "RY"),
    ("image_database_toedit_SN_20190827.xlsx", "SN"),
    ("image_database_toedit_SUE_20190925.xlsx", "SUE"),
]

print("Might need to update user initials-to-path mapping")
usr_lookup_for_path = {
    "CCH": "Caroline",
    "SN": "Sara",
    "SUE": "Sue",
    "CLF": "Frick",
    "IAM": "Irina",
    "JLG": "Jamie",
    "MH": "Melissa",
    "RY": "Ruian",
}

# Load and process dataframes
full_fns = [(raw_dir + fn, usr) for fn, usr in fn_usr]


def load_df(fn, usr):
    df = pd.read_excel(fn)
    df["User"] = usr
    return df


raw_dfs = [load_df(fn, usr) for fn, usr in full_fns]
time_lapse_values = np.concatenate([df["Time-lapse"].unique() for df in raw_dfs])


# Find values for time negative time lapses
print("Values for 'Time-lapse' are:")
print(time_lapse_values)

print("Might need to update negative values if there are new ones, e.g. 'nope' or 'naw'")
negative_values = set(['no', 'NO', 'No ', ' no', 'No', 'nan', np.nan, 'No and yes', '  '])
positive_values = set(time_lapse_values).difference(negative_values)

# Sort out entries that don't have a negative entry
time_lapse_dfs = [df.loc[df['Time-lapse'].isin(positive_values)] for df in raw_dfs]
time_lapse_df = pd.concat(time_lapse_dfs, join='inner', ignore_index=True, sort=False)

# Figure out the full path

def row_to_path(row):
    if type(row.Date) is not pd._libs.tslibs.timestamps.Timestamp:
        return None
    user = usr_lookup_for_path[row.User]
    year = row.Date.strftime("%Y")
    date = row.Date.strftime("%Y%m%d")
    path = os.path.join('/allen/aics/assay-dev/MicroscopyData/',user,year,date)
    return path

def annotate_user_path(df):
    df['Path'] = df.apply(row_to_path, 1)
    return df

time_lapse_df = annotate_user_path(time_lapse_df)
time_lapse_df = time_lapse_df.query('Path not in [None,]')

# Write out
time_lapse_df.to_csv('./big_raw_image_list.csv')
