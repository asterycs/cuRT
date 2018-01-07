#!/bin/bash

SCENEFILE="$1"
RENDERER="$2"
PATHS="$3"
if [ -z "$PATHS" ]; then
  PATHS=1
fi
OUT=$(./graphx -b -s "$SCENEFILE" -o out.png -r "$RENDERER" -p "$PATHS")

TIME=$(echo "$OUT" | grep "Rendering time" | awk '{print $NF}')

echo "$TIME" >> benchmarks.txt

echo "$TIME" milliseconds

exit 0
