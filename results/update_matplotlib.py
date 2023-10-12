# Standard library imports
import os

# Third party imports
import matplotlib.pyplot as plt

# Update plot parameters to use latex font as specified
os.environ["PATH"] += os.pathsep + '/Library/TeX/texbin'
params = {"text.usetex"         : True,
          "font.family"         : "serif",
          "font.size"           : "11",
          'text.latex.preamble' : r"\usepackage{bm}"}
plt.rcParams.update(params)