# Avatar
Facial recognition with openCV4 controlling monster avatar

## ****  Getting Started ****
This project should run on Windows, Mac, or Linux since it uses openCV and a
cross platform GUI. Install the required packages with pip, but if you you are
on linux... installing the wxpython files is not fun since the wheels are not
well supported. See the following website for help:
https://wxpython.org/pages/downloads/index.html

If you are using linux on MAC, then see below for help installing drivers
for your webcam before you do a test with openCV.

--------------------------------------------------------------------------------
## **** Installation With Pip ****
Run the following command to install all the requirements from the doc.
This sometimes works... you can also just install everything in the requirements
document and you may need different versions based on your setup. Good luck.

pip3 install -r requirements.txt

--------------------------------------------------------------------------------
## **** Instructions for setting up webcam on MAC ****
sudo apt-get install git
sudo apt-get install curl xzcat cpio
git clone https://github.com/patjak/facetimehd-firmware.git
cd facetimehd-firmware
make
sudo make install
cd ..
sudo apt-get install kmod libssl-dev checkinstall
git clone https://github.com/patjak/bcwc_pcie.git
cd bcwc_pcie
make
sudo make install
sudo depmod
sudo modprobe -r bdc_pci
sudo modprobe facetimehd
sudo nano /etc/modules
**add line "facetimehd", write out (ctl+o) & close**

NOTE:
Most of those steps will need to be repeated every time the kernel is upgraded.
--------------------------------------------------------------------------------
