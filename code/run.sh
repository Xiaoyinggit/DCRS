#!/bin/sh
mkdir log
mkdir debug
mkdir saved_models
python3 -u main.py  "$@"  
