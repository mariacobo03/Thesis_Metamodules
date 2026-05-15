#!/bin/bash
        
# Set up dynamic values for NAME, WD, and LR
NAME="human1"  # Replace dots with hyphens for a valid Kubernetes name

# Export variables so they are available for envsubst
export NAME

# Substitute variables in jobdefinition.yaml and save to a temp file
envsubst '${NAME}' < jobdefinition.yaml > jobdefinition.yaml.out

# Check the output to verify substitution before creating the job
cat jobdefinition.yaml.out

# Create the Kubernetes job
kubectl create -f jobdefinition.yaml.out 

# Clean up the temporary job definition file
rm jobdefinition.yaml.out
