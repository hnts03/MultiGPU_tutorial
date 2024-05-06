#!/bin/bash


make all-ignore-cpu
./cpu
./single-nostream
./single-stream
./multi-nostream
./multi-stream
make clean
