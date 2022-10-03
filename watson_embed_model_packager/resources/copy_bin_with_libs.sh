#!/usr/bin/env bash

# This script is used at build time to copy a given binary file and its dynamic
# libs to output directories that can then be copied over to a minimal release

# Read env config
bin_out=${BIN_OUT_PATH:-"/bin_out"}
lib_out=${LIB_OUT_PATH:-"/lib_out"}

# From here on, fail on unset variables and broken pipes
set -euo pipefail

# Figure out what bin we're targeting
target_bin_name=$1
target_bin=$(realpath $(which $target_bin_name))

# List of libs to ignore and funciton to check an arbitrary lib name
ignored_libs=(
    "libc"
    "libselinux"
    "libpcre2-8"
    "ld-linux-aarch64"
    "ld-linux-x86-64"
)
function ignored_lib {
    lib_name=$(echo "$1" | cut -d'.' -f 1 | rev | cut -d'/' -f 1 | rev)
    for ignored_lib in "${ignored_libs[@]}"
    do
        if [ "$ignored_lib" == "$lib_name" ]
        then
            echo "true"
            return
        fi
    done
    echo "false"
}

# Make sure the output dirs exist
mkdir -p $bin_out
mkdir -p $lib_out

# Copy the bin over
cp $target_bin $bin_out/$target_bin_name

# Copy over all the linked libs
for val in $(ldd $target_bin)
do
    if [ -f $val ] && [ "$(ignored_lib $val)" == "false" ]
    then
        cp $val $lib_out/
    fi
done
