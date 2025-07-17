#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p tw_games

# Loop to create 100 games
for i in $(seq 1 100); do
    echo "Generating game $i..."
    tw-make custom \
        --world-size 5 \
        --nb-objects 10 \
        --quest-length 5 \
        --output tw_games/custom_game_${i}.z8
done

echo "All games generated in tw_games/"
