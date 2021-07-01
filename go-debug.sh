#!/usr/bin/zsh

while [[ 1 ]]; do ./generator.py > input && make debug < input; done
