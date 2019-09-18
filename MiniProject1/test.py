import pip
from pip._internal import main
import os
os.system('/bin/bash -c "pip install numpy"')
import numpy as np
import math

main(['list'])
main(['show', 'wheel'])
print (np.cbrt(27))


