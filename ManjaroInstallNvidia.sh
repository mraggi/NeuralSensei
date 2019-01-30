#! /bin/bash
# Script to install Nvidia Drivers On Manjaro Laptop
RED='\033[0;31m'
BLUE='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

echo -e "${WHITE}--Script to install Nvidia Drivers On Manjaro Laptop--\n--It only works on Geforce 600 and higher--${NC}"
echo -e "${RED}Please make a full update of your system before running this script\n${NC}"

echo -e "${BLUE}Below is your Kernell version, remember the first 3 digits"
echo -e "${RED}$(uname -r)"
echo -e "${BLUE}When Installing Nvidia, it will ask which repository you want"
echo -e "Select ${RED}linuxXXX-nvidia ${BLUE}where XXX are the first 3 digits of your kernel${NC}"
read -p "Read above and press any key..."
echo

echo "Now installing stuff"
sudo pacman -S bumblebee mesa xf86-video-intel lib32-virtualgl

echo -e "Now installing Nvidia Driver ${RED}Please REMEMBER THE CORRECT KERNELL${NC}"
read -p "Read above and press any key..."
echo

sudo pacman -S nvidia
sudo pacman -S lib32-nvidia-utils

echo "Adding your user to bumblebee group"
sudo gpasswd -a $USER bumblebee
echo

echo "Enabling bumblebee service"
sudo systemctl enable bumblebeed.service
echo

echo -e "\n${BLUE}Now reboot your system :3"
echo -e "Anything that you want to run using the gpu must be run right after the 'optirun' instruction"
echo -e "Example: "
echo -e "   optirun ipython"

# To test if it's done type bellow command:
# optirun glxgears -info
