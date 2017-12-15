#!/bin/bash

OUT=$(./graphx -b -s last.scene -o out.png)

TIME=$(echo "$OUT" | grep "Rendering time" | awk '{print $NF}')

echo "$TIME" >> benchmarks.txt

echo "$TIME" milliseconds

exit 0
