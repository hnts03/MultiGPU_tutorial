#!/bin/bash

if [ ! -n "$1" ]; then
	echo "Usage: $0 <profile dirs>"
	echo "Example: $0 profiles     // It will make the directory named profiles and save nsys-profiles in it."
	exit 0
fi

make all-ignore-cpu

mkdir -p ./$1
nsys profile --gpu-metrics-device all -o $1/single-nostream single-nostream
nsys profile --gpu-metrics-device all -o $1/single-stream single-stream
nsys profile --gpu-metrics-device all -o $1/multi-nostream multi-nostream
nsys profile --gpu-metrics-device all -o $1/multi-stream multi-stream

make clean
