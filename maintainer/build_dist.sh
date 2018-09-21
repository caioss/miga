#!/bin/bash

BUILD_DIR="dist_build"

rm -rf $BUILD_DIR
mkdir $BUILD_DIR

rsync -avc --delete ../package/ $BUILD_DIR/

cd $BUILD_DIR
python3 setup.py sdist
cd ..
