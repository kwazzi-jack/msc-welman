#!/bin/bash

# Colors
YELLOW='\033[1;33m'
RED='\033[0;31m'
GREEN='\033[0;32m'

# Function to wrap a string in color
colorize() {
  local color_code="$1"
  local text="$2"
  local reset='\033[0m' # No Color
  echo -e "${color_code}${text}${reset}"
}


echo "==============================="
echo "   $(colorize $GREEN "Virtual Environment Setup")   "
echo "==============================="

# Colorized words
PYTHON=$(colorize $YELLOW "python3.8")
PIP=$(colorize $YELLOW "pip")
VIRTUALENV=$(colorize $YELLOW "virtualenv")
VENV=$(colorize $YELLOW 'venv')
LOG=$(colorize $GREEN '✓')
ERR=$(colorize $RED '✗')

# Check if python3.8 is installed
echo "[$LOG] Checking for $PYTHON."
if ! command -v python3.8 &>/dev/null; then
    echo "[$ERR] $PYTHON is not installed. 
    
-> Please install $PYTHON: https://www.python.org/downloads/release/python-3113/"
    exit 1
else
    echo "[$LOG] $PYTHON is installed."
fi

# Check if pip is installed
echo "[$LOG] Checking for $PYTHON package manager ($PIP)."
if ! command -v python3.8 -m pip --version &>/dev/null; then
    echo "[$ERR] $PIP not installed for $PYTHON. 
    
-> Please install $PIP for $PYTHON: https://packaging.python.org/en/latest/tutorials/installing-packages/"
    exit 1
else
    echo "[$LOG] $PIP is installed."
fi

# Check if virtualenv is installed
echo "[$LOG] Checking for $VIRTUALENV package."
if ! command -v python3.8 -m virtualenv --version &>/dev/null; then
    echo "[$ERR] $VIRTUALENV not installed for $PYTHON. 
    
-> Please install $VIRTUALENV for $PYTHON: 

        python3.8 -m pip install virtualenv
        
   or, for Debian/Ubuntu, use:

        sudo apt install virtualenv"
    exit 1
else
    echo "[$LOG] $VIRTUALENV is installed."
fi

# Check if virtual environment exists
REPLACE_ENV=true
if [ -d "./venv" ]; then
  read -rp "[$LOG] Virtual environment $VENV already exists. Do you want to replace it? (y/n): " CHOICE
  if [[ $CHOICE != [Yy] ]]; then
    echo "[$LOG] Using existing virtual environment."
    REPLACE_ENV=false
  else
    echo "[$LOG] Replacing virtual environment."
    rm -rf ./venv
  fi
fi

# Create virtual environment
if $REPLACE_ENV; then
    echo "[$LOG] Creating virtual environment called $VENV."
    if output=$(python3.8 -m virtualenv -p 3.8 venv 2>&1); then
        echo "[$LOG] $VENV created successfully."
    else
        echo "[$ERR] Failed to create virtual environment.

    -> Command error output:
            $output"
        exit 1
    fi
fi

# New python interpreter
PYVENV=./venv/bin/python

# Install require packages
echo "[$LOG] Checking for required packages."
$PYVENV -m pip install --no-compile -r "./requirements.txt"

# Done
echo "[$LOG] Setup complete."