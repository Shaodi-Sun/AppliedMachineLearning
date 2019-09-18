import pip
from pip._internal import main
import os
os.system('/bin/bash -c "pip install numpy"')

main(['list'])
main(['show', 'wheel'])


