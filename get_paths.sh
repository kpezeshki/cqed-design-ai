#!/bin/bash

# Check if filename argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    echo "This script returns the full path to the local Python executable and the full path to the specified file."
    exit 1
fi

filename="$1"

# Get the full path to the Python executable
# Show Poetry Python executable info
echo "=== Poetry Python Environment ==="
poetry env info | grep Executable

# Get the full path to the file
if [ -f "$filename" ]; then
    # File exists, get its absolute path
    file_path=$(realpath "$filename")
elif [ -f "./$filename" ]; then
    # File exists in current directory
    file_path=$(realpath "./$filename")
else
    # File doesn't exist, but we can still show what the absolute path would be
    file_path=$(realpath "$filename" 2>/dev/null)
    if [ -z "$file_path" ]; then
        # If realpath fails, construct the path manually
        if [[ "$filename" = /* ]]; then
            # Already an absolute path
            file_path="$filename"
        else
            # Make it relative to current directory
            file_path="$(pwd)/$filename"
        fi
    fi
    echo "Warning: File '$filename' does not exist. Showing hypothetical path."
fi

# Output the results
echo "Python executable: $python_path"
echo "File path: $file_path"
