#!/bin/bash

SCENEFILE="$1"
OUT=$(./graphx -b -s "$SCENEFILE" -o out.png)

TIME=$(echo "$OUT" | grep "Rendering time" | awk '{print $NF}')

echo "$TIME" >> benchmarks.txt

echo "$TIME" milliseconds

exit 0
