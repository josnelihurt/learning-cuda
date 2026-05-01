#!/bin/bash
# Install clang-uml on Ubuntu 24.04 (Noble) via the official PPA.
# Safe to re-run — checks whether clang-uml is already installed first.
set -euo pipefail

if command -v clang-uml &>/dev/null; then
    echo "clang-uml already installed: $(clang-uml --version 2>&1 | head -1)"
    exit 0
fi

echo "Installing clang-uml via PPA (bkryza/clang-uml)..."
sudo add-apt-repository -y ppa:bkryza/clang-uml
sudo apt-get update -qq
sudo apt-get install -y clang-uml

echo "Installed: $(clang-uml --version 2>&1 | head -1)"
