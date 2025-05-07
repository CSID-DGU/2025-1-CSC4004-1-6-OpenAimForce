#!/bin/bash

# Usage: ./run_server.sh [PORT]
# If no PORT is given, defaults to 28763

PORT=$1

if [[ -z "$PORT" ]]; then
  PORT=28763
  echo "No port specified. Using 28763 as default."
  echo "To specify port: ./run_server.sh <external-port>"
else
  if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Invalid port: $PORT"
    echo "Usage: ./run_server.sh [PORT]"
    exit 1
  fi
fi

echo "Starting AssaultCube server: external port $PORT → internal port 28763"

docker run -it \
  -p "$PORT":28763/udp \
  -p "$PORT":28763/tcp \
  ac:deploy bash -c "cd AC && ./server.sh -f"

