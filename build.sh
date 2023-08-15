#!/bin/bash

# Change directory to the "pso" crate
cd rust/pso
echo "Running maturin develop for 'pso' crate..."
maturin develop

# Change directory back to the project root
cd ..

# Change directory to the "perlin" crate
cd perlin
echo "Running maturin develop for 'perlin' crate..."
maturin develop

cd ../..

echo "Done!"