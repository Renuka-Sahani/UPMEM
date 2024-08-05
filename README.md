# UPMEM

You can just focus one dpu and pmr (other file h1,h2,ad.c you can ignore that is just for testing)

These were the steps I followed:
System Python Setup
1) sudo apt update
2) sudo apt install python3-pip
3) if there are some modules missing sudo pip3 install psutil

0) Create a upmem folder and put the archive in it.
1) Untar the archive
2) Go inside the upmem-2024.1.0-Linux-x86_64 folder
3) Run source upmem_env.sh
0) We create a .c file which has kmeans implementation, and it generates random points and the algorithm will try to cluster it.
command to compile the c code: 
dpu-upmem-dpurte-clang -o dpu dpu.c
command to run the compiled coe
dpu-lldb dpu
process launch
