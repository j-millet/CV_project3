# !/bin/bash
pip install cityscapesscripts

mkdir -p archives
csDownload -d archives gtFine_trainvaltest.zip camera_trainvaltest.zip leftImg8bit_trainvaltest.zip

unzip -n archives/gtFine_trainvaltest.zip -d ./
unzip -n archives/camera_trainvaltest.zip -d ./ 
unzip -n archives/leftImg8bit_trainvaltest.zip -d ./