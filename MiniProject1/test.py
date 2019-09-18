import pip
from pip._internal import main
import os
os.system('/bin/bash -c "sudo pip install numpy matplotlib scipy"')
import numpy as np
import math
import scipy
import matplotlib

main(['list'])
main(['show', 'wheel'])
print('scipy Version: '+scipy.__version__)
print('matplotlib Version: '+matplotlib.__version__)
print (np.cbrt(27))


