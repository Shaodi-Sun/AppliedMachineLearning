#MacOs install Python2 and Python3 Bash Commands

#!/bin/bash
brew install python3 
#create symbolic link
sudo mkdir -p /usr/local/Frameworks
sudo chown -R $(whoami) /usr/local/*
brew link python3

#After above commands, I have both python 2.7.10 and python 3.7.4 
#on my machine
#python --version output 2.7.10 
#python3 --version output 3.7.4 


#create virtual environment
mkdir ~/.virtualenvs
python3 -m venv ~/.virtualenvs/comp551
source ~/.virtualenvs/comp551/bin/activate

#install pip 
sudo curl -O https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
#update pip to the latest version
python3 -m pip install --user --upgrade pip==19.2.3
export PATH=$PATH:/usr/local/Cellar/python/3.7.4_1/Frameworks/Python.framework/Versions/3.7/bin/