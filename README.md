# acl-bench

## Disclaimer
This is a small project to help abstract benchmarking of the Arm Compute Library for GPU vs CPU comparisons.

## Build
You need to set-up ACL_ROOT to point wherever your ACL checkout is, and it must have been already built.
To build this benchmark run either:

make ACL_ROOT=<YOUR_ACL>

OR

export ACL_ROOT=<YOUR_ACL>  
make

## Info
This currently only supports running with ConvolutionLayer as that was the main purpose.

