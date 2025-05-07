#!/bin/bash

# Remove previous image (ignore error if not present)
docker rmi -f ac:deploy 2>/dev/null

# Run build container from ac:base
CID=$(docker run -it -d ac:base bash)

# Clone specific directory and branch using sparse checkout
docker exec "$CID" bash -c "
  git init temprepo &&
  cd temprepo &&
  git remote add origin https://github.com/CSID-DGU/2025-1-CSC4004-1-6-OpenAimForce &&
  git fetch origin game_dev &&
  git sparse-checkout init --cone &&
  git sparse-checkout set GameServer/AC &&
  git checkout game_dev &&
  mv GameServer/AC /AC &&
  cd /AC/source/src &&
  make -j8 server_install
"

# Commit the container state as ac:deploy
docker commit "$CID" ac:deploy

# Optionally stop and remove the container
docker stop "$CID" >/dev/null
docker rm "$CID" >/dev/null
