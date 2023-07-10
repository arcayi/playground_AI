#!/bin/bash

#get highest tag number
VERSION=$(git describe --abbrev=0 --tags)

#create new tag
NEW_TAG=$(date +"%-Y.%-m.%-d.%-H.%-M")
# NEW_TAG=$(date +"%Y.%m.%d.%H.%M")

#get current hash and see if it already has a tag
GIT_COMMIT=$(git rev-parse HEAD)
#NEEDS_TAG=`git describe --contains $GIT_COMMIT`
#Errors can be silenced by redirecting stderr output to /dev/null like so
NEEDS_TAG=$(git describe --contains $GIT_COMMIT 2>/dev/null)

#only tag if no tag already (would be better if the git describe command above could have a silent option)
if [ -z "$NEEDS_TAG" ]; then
    echo "Updating $VERSION to $NEW_TAG"
    echo "Tagged with $NEW_TAG (Ignoring fatal:cannot describe - this means commit is untagged) "
    git tag $NEW_TAG -a -m "$NEW_TAG"
    git push --follow-tags
else
    echo "Already a tag on this commit"
fi
