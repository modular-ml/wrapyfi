#!/bin/bash


# Function to check if a Docker tag exists on Docker Hub
tag_exists() {
    local image="$1"
    local tag="$2"
    local exists=$(curl -s -o /dev/null -w "%{http_code}" https://hub.docker.com/v2/repositories/${image}/tags/${tag}/)
    if [ "$exists" == "200" ]; then
        return 0
    else
        return 1
    fi
}

cd dockerfiles

# get the version from pyproject.toml
VERSION=$(grep -oP '(?<=version = ")[^"]+' ../pyproject.toml)
# check if the version was extracted correctly
if [[ -z "$VERSION" ]]; then
    echo "Failed to extract version from pyproject.toml"
    exit 1
fi

# iterate over each .Dockerfile in the directory
for file in *.Dockerfile; do
    # extract middleware name from filename
    middleware=$(echo "$file" | sed 's/wrapyfi_//' | sed 's/\.Dockerfile//')
    image_tag="modularml/wrapyfi:$VERSION-$middleware"

    # check if the tag already exists on Docker Hub
    if tag_exists "modularml/wrapyfi" "$VERSION-$middleware"; then
        echo "Tag $image_tag already exists on Docker Hub, skipping..."
        continue
    fi

    # replace version in .Dockerfile
    sed -i "s/wrapyfi\[headless\]==[0-9]*\.[0-9]*\.[0-9]*/wrapyfi[headless]==$VERSION/" "$file"

    # extract build command from the Dockerfile
    build_command=$(grep '# docker build' "$file" | sed 's/# //')

    # execute the build command
    eval "$build_command"

    # tag the image
    docker tag "wrapyfi-$middleware" "$image_tag"

    # push the tagged image
    docker push "$image_tag"

    # delete all images
    docker rmi -f $(docker images -aq)
done

cd ../