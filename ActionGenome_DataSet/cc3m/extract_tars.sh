#!/bin/bash

for file in *.tar; do
  folder="${file%.tar}"

  mkdir -p "$folder"

  tar -xf "$file" -C "$folder"
done