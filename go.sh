#!/usr/bin/zsh

while [[ 1 ]]; do ./generator.py > input && make run < input; done