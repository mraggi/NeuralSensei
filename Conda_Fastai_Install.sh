#! /bin/bash
# Script to install Anaconda and Fastai on manjaro
RED='\033[0;31m'
BLUE='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

echo -e "${WHITE}--Script to install Anaconda and Fastai On Manjaro--${NC}"
echo -e "${RED}Please make a full update of your system before running this script\n${NC}"

echo "Installing yay..."
git clone https://aur.archlinux.org/yay.git
cd yay
makepkg -si
echo

echo -e "${BLUE}Installing Anaconda"
echo -e "${RED}Beware this will delay about 30min on the 'Compressing Package' process${NC}"
yay -S anaconda --noconfirm
echo

echo "CondaStart()
{
    source /opt/anaconda/bin/activate root
}" >> ~/.bashrc

echo
echo -e "To start the anaconda shell type ${BLUE}CondaStart${NC}\n"
source ~/.bashrc

CondaStart

echo "Configuring fastai environment..."
conda create -n fastai
conda activate fastai
conda config --prepend channels conda-forge
conda config --prepend channels pytorch
conda config --prepend channels fastai/label/test
conda config --prepend channels fastai
conda install fastai pytorch pillow-simd

echo -e "Done!\n"
echo -e "To activate fastai environment type ${BLUE}conda activate fastai${NC}"