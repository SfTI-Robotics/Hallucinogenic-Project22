#!/bin/bash
docker build --tag=p4p .

docker run -v $(pwd):/home -it  --runtime=nvidia p4p bash
