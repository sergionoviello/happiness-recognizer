#!/bin/bash

docker build -t happiness-recognizer:latest .
docker tag happiness-recognizer:latest 465160476334.dkr.ecr.eu-west-1.amazonaws.com/happiness:latest
docker push 465160476334.dkr.ecr.eu-west-1.amazonaws.com/happiness:latest