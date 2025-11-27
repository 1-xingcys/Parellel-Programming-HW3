# Parallel Programming HW3 - CUDA Ray Marching

## Overview

This project implements a parallelized Ray Marching algorithm using CUDA, converting the original sequential CPU program into GPU parallel computation to improve rendering performance.

## Build and Run

### Build

```bash
make          # Build the CUDA version
make cpu      # Build the CPU version
```

### Run

```bash
./hw3 <camera_x> <camera_y> <camera_z> <lookat_x> <lookat_y> <lookat_z> <width> <height> <output.png>
```

### Examples

```bash
# Test case 01
./hw3 4.152 2.398 -2.601 0 0 0 512 512 output_gpu/01.png

# Test case 08
./hw3 -1.2 -0.51 -0.8 -0.271 -0.299 -0.379 4096 4096 output_gpu/08.png
```

## Project Structure

```
hw3/
├── hw3.cu              # CUDA implementation
├── hw3_cpu.cpp         # CPU reference implementation
├── Makefile            # Build configuration
├── report.pdf          # Detailed report
├── testcases/          # Test cases
├── output_gpu/         # GPU output results
└── lodepng/            # PNG encoding/decoding library
```

## References

For performance analysis and implementation details, please refer to report.pdf.