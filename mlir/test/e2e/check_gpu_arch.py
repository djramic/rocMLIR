#!/usr/bin/env python3
"""
Check if GPU architecture has changed and trigger test regeneration if needed.
This script is called by CMake during E2E test generation.
"""
import sys
import os
import glob

def main():
    if len(sys.argv) < 3:
        print("Usage: check_gpu_arch.py <marker_file> <e2e_dir> <common_utils_path>")
        sys.exit(1)
    
    marker_file = sys.argv[1]
    e2e_dir = sys.argv[2]
    common_utils_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    if common_utils_path:
        sys.path.insert(0, common_utils_path)
    
    try:
        from common import get_default_agent
        arch = get_default_agent() or 'unknown'
    except ImportError:
        print("Warning: Could not import common module, using 'unknown' arch")
        arch = 'unknown'
    
    # Read old architecture from marker file
    try:
        with open(marker_file, 'r') as f:
            old_arch = f.read().strip()
    except FileNotFoundError:
        old_arch = ''
    
    if old_arch != arch:
        print(f'GPU architecture changed: {old_arch or "(none)"} -> {arch}. Will regenerate E2E tests.')
        
        # Write new architecture to marker file
        with open(marker_file, 'w') as f:
            f.write(arch)
        
        # Remove .copy files to force regeneration
        copy_files = glob.glob(os.path.join(e2e_dir, '*.copy'))
        for f in copy_files:
            os.remove(f)
            print(f'Removed {f}')
    else:
        print(f'GPU architecture unchanged: {arch}')

if __name__ == '__main__':
    main()
