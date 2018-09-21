#!/bin/bash

BUILD_DIR="dist_build"

twine upload $BUILD_DIR/dist/*
