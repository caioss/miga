#!/bin/bash

#
# Version standard
#
# MAJOR.MINOR.BUGFIX
# MAJOR.MINOR.BUGFIX-dev
# MAJOR.MINOR.BUGFIX-rc1
# MAJOR.MINOR.BUGFIX-rc2
#

VERSION="1.0.0"
SHORT="1.0"

sed -i 's/[0-9]\+\.[0-9]\+\.[0-9]\+\(-dev\|-rc[0-9]*\)\?/'$VERSION'/' ../package/miga/__init__.py

sed -i 's/\(version *= *"\)[0-9]\+\.[0-9]\+\.[0-9]\+\(-dev\|-rc[0-9]*\)\?/\1'$VERSION'/' ../package/setup.py

sed -i 's/\(version *= *"\)[0-9]\+\.[0-9]\+\.[0-9]\+\(-dev\|-rc[0-9]*\)\?/\1'$SHORT'/' ../docs/python/source/conf.py
sed -i 's/\(release *= *"\)[0-9]\+\.[0-9]\+\.[0-9]\+\(-dev\|-rc[0-9]*\)\?/\1'$VERSION'/' ../docs/python/source/conf.py
