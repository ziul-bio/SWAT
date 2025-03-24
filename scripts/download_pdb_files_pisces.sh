#!/bin/bash

# Check if the data directory exists, if not, create it
mkdir -p data/PISCES/pdb_files

# Read each line of the txt file
while read -r pdb_id; do
    # Check if the pdb_id isn't empty
    if [[ ! -z "$pdb_id" ]]; then
        # Download the PDB file for the current ID
        curl -f "https://files.rcsb.org/download/${pdb_id}.pdb.gz" -o "data/PISCES/pdb_files/${pdb_id}.pdb.gz"
    fi
done < data/PISCES/pisces_pdb_ids.txt