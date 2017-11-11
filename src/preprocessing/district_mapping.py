import sys
sys.path.append('../')

import os

import pandas as pd

import utilities.io as io

doc = io.read_csv('../../data/subset/listings_sub.csv')
