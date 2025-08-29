#!/bin/bash

# Create original array of size 500
echo "Creating array of 500 elements..."
original_array=()
for i in {1..500}; do
    original_array+=("element_$i")
done

echo "Original array size: ${#original_array[@]}"
echo ""

# Calculate how many groups we need dynamically
max_group_size=100
total_groups=$(( (${#original_array[@]} + max_group_size - 1) / max_group_size ))

echo "Splitting into arrays of max $max_group_size elements each..."
echo "Total groups needed: $total_groups"
echo ""

# Print each group dynamically
for ((group=0; group<total_groups; group++)); do
    start_element=$((group * max_group_size + 1))
    end_element=$(( (group + 1) * max_group_size ))
    if [ $end_element -gt ${#original_array[@]} ]; then
        end_element=${#original_array[@]}
    fi
    echo "=== TEMP ARRAY $((group + 1)) (Elements $start_element-$end_element) ==="
    
    # Collect and print elements for this group
    group_elements=()
    count=0
    
    for i in "${!original_array[@]}"; do
        # Check if this element belongs to current group using dynamic group size
        if [ $((i / max_group_size)) -eq $group ]; then
            group_elements+=("${original_array[$i]}")
            ((count++))
        fi
    done
    
    echo "Size: $count"
    echo "Contents: ${group_elements[@]}"
    echo ""
done

echo "=== SUMMARY ==="
echo "Original array size: ${#original_array[@]}"
echo "Split into $total_groups groups of max $max_group_size elements each using dynamic modulo arithmetic"
