#!/usr/bin/env bash
# Build script for Render deployment

# Ensure we're using Python 3.10
python3.10 --version || python3.10.12 --version || python3.10.0 --version

# Upgrade pip, setuptools, and wheel first
python3.10 -m pip install --upgrade pip setuptools wheel || python3.10.12 -m pip install --upgrade pip setuptools wheel || python3.10.0 -m pip install --upgrade pip setuptools wheel || pip3.10 install --upgrade pip setuptools wheel || pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

