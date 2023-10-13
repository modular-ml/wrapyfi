#!/bin/bash

# generate pyproject.toml file
pip install pdm
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

# UPDATE ROS & ROS2 INTERFACE VERSIONS
# get the version from pyproject.toml
VERSION=$(grep -oP '(?<=version = ")[^"]+' pyproject.toml)
# check if the version was extracted correctly
if [[ -z "$VERSION" ]]; then
    echo "Failed to extract version from pyproject.toml"
    exit 1
fi
# update version in various package.xml files
update_version_in_package_xml "wrapyfi_extensions/wrapyfi_ros2_interfaces/package.xml"
update_version_in_package_xml "wrapyfi_extensions/wrapyfi_ros_interfaces/package.xml"

# generate docs
cd docs
./build_docs.sh
cd ../