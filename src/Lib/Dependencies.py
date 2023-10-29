# GENERAL
from typing import Any
import os
import math
import glob
import io
import re
import string
import tqdm

# DATA
import openpyxl
from bs4 import BeautifulSoup
import urllib.request
from lxml import html


## LEARN
import numpy as np

import tensorflow as tf
import keras
import pymc3 as pm

from keras import layers
from keras.layers import Dense

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pickle
from keras.optimizers import Adam



### CONSTANTS
SCORES_DIR = './datasets/gameScores'
RECAPS_DIR = './datasets/gameRecaps'
VECS_DIR = './datasets/vecs'
MODELS_DIR = './models'

NFLTEAMS = ['bears', 'bengals', 'bills', 'broncos', 'browns', 'buccaneers', 'cardinals', 'chargers', 'chiefs', 'colts', 'commanders', 'cowboys', 'dolphins', 'eagles', 'falcons', '49ers', 'giants', 'jaguars', 'jets', 'lions', 'packers', 'panthers', 'patriots', 'raiders', 'rams', 'ravens', 'saints', 'seahawks', 'steelers', 'texans', 'titans', 'vikings']