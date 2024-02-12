#!/bin/bash

# generate pyproject.toml file, Check the pyproject.toml file again when dependencies are added or removed
python3 -m pip install pdm
pdm import setup.py

update_version_in_package_xml() {
    # path to the package.xml file
    PACKAGE_XML_PATH=$1
    # check if the package.xml exists
    if [[ ! -f "$PACKAGE_XML_PATH" ]]; then
        echo "package.xml not found at path: $PACKAGE_XML_PATH"
        exit 1
    fi
    # replace the version in package.xml using sed
    sed -i.bak -E "s|<version>[^<]+</version>|<version>$VERSION</version>|" "$PACKAGE_XML_PATH"
    # check if the sed command was successful
    if [[ $? -ne 0 ]]; then
        echo "Failed to update version in $PACKAGE_XML_PATH"
        exit 1
    fi
    echo "Version updated to $VERSION in $PACKAGE_XML_PATH"
}

# UPDATE ROS & ROS 2 INTERFACE VERSIONS (ADDED TO SEPARATE REPOSITORIES, THEREFORE, THIS IS NOT NEEDED)
#######################################################################################################################

# # get the version from pyproject.toml
# VERSION=$(grep -oP '(?<=version = ")[^"]+' pyproject.toml)
# # check if the version was extracted correctly
# if [[ -z "$VERSION" ]]; then
#     echo "Failed to extract version from pyproject.toml"
#     exit 1
# fi
# # update version in various package.xml files
# update_version_in_package_xml "wrapyfi_extensions/wrapyfi_ros2_interfaces/package.xml"
# update_version_in_package_xml "wrapyfi_extensions/wrapyfi_ros_interfaces/package.xml"

# GENERATE DOCUMENTATION
#######################################################################################################################

# refactor code with black
python3 -m pip install black
black .

# compile docs with sphinx
cd docs
python3 -m pip install -r requirements.txt
./build_docs.sh
cd ../

# BUILD PACKAGE
#######################################################################################################################

# build the package resources and place them in a dist/* directory
rm -rf dist
python3 -m pip install --upgrade build
python3 -m build

# UPLOAD TO PYPI
#######################################################################################################################

# update on pypi
python3 -m pip -vvv install --upgrade --force-reinstall cffi
python3 -m pip install --upgrade twine
# upload to rest repo
#python3 -m twine upload --repository testpypi dist/*
# upload to actual pypi
python3 -m twine upload dist/*

# when prompted for username and password, generate API token from pypi.org and set username to __token__ as shown below
# [pypi]
  #  username = __token__
  #  password = pypi-******
