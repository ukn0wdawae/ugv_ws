#!/usr/bin/env bash
# Build the ROS 2 Humble + Gazebo Classic image (run once, or after Dockerfile edits).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t ugv-humble-gazebo:latest "$HERE"
echo
echo "Image built. Next:  ./docker/shell.sh"
