#!/bin/bash

# Download the KuaiRec dataset
wget https://nas.chongminggao.top:4430/datasets/KuaiRec.zip --no-check-certificate

# Unzip the downloaded file
unzip KuaiRec.zip

# Clean up by removing the zip file
rm KuaiRec.zip

echo "Download and extraction complete in the data/ directory!"