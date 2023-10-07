# Tools
import pandas as pd
import numpy as np
import scipy as sp

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns


def percentage(part, whole, digits=4):
    return round(100 * float(part)/float(whole), digits)