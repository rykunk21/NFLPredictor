# DATA
import os
import openpyxl
from bs4 import BeautifulSoup
import urllib.request
import math


## LEARN
import io
import re
import string
import tqdm
import glob

import numpy as np

import tensorflow as tf
from keras import layers

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

### CONSTANTS
SCORES_DIR = './datasets/gameScores'
RECAPS_DIR = './datasets/gameRecaps'
VECS_DIR = './datasets/vecs'
CURRENT_WEEK = 6

NFLTEAMS = ['bears', 'bengals', 'bills', 'broncos', 'browns', 'buccaneers', 'cardinals', 'chargers', 'chiefs', 'colts', 'commanders', 'cowboys', 'dolphins', 'eagles', 'falcons', '49ers', 'giants', 'jaguars', 'jets', 'lions', 'packers', 'panthers', 'patriots', 'raiders', 'rams', 'ravens', 'saints', 'seahawks', 'steelers', 'texans', 'titans', 'vikings']