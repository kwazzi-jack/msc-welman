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
echo "      $(colorize $GREEN "Simulation Setup")   "
echo "==============================="

# Colorized words
LOG=$(colorize $GREEN '✓')
ERR=$(colorize $RED '✗')

echo "[$LOG] Fetching arguments"
NAME=$1
DIRECTORY=$2
MAIN="$DIRECTORY/$NAME"

echo "[$LOG] Name='$NAME'"
echo "[$LOG] Path='$MAIN'"

mkdir -p "$MAIN"

cp -r source "$MAIN/"
cp simulation.ipynb "$MAIN/$NAME.ipynb"

echo "[$LOG] Directory made and contents copied"

YAML_FILE="$MAIN/$NAME.config"

SEARCH="script_options = script_settings(\"main.config\")"
REPLACE="script_options = script_settings(\"$NAME.config\")"
sed -i "s|$SEARCH|$REPLACE|" "$MAIN/$NAME.ipynb"

cat <<EOF > "$YAML_FILE"
config-dir: config
data-dir: data
log-level: 3-INFO
mpl-dir: ''
n-cpu: 4
name:
plots-dir: plots
seed: 666
EOF

echo "[$LOG] Files renamed and config file made"
echo "[$LOG] Done"