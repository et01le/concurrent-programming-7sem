task1/pi-computation
```bash
gcc -fopenmp -o pi-computation.exe pi-computation.c
./pi-computation.exe <number of threads>
```

task1/sqr-matrix-mul
```bash
gcc -fopenmp -o sqr-matrix-mul.exe sqr-matrix-mul.c
./sqr-matrix-mul.exe <number of threads>
./sqr-matrix-mul.exe -d # or --demo alternatively, enables demo mode
```

task1/strassen-algorithm
```bash
gcc -fopenmp -o strassen-algorithm.exe strassen-algorithm.c
./strassen-algorithm.exe
./strassen-algorithm.exe -d # or --demo alternatively, enables demo mode
```
My implementation of Strassen's algorithm is naive and thus unefficient (due to many redundant memory allocations)

task2/matrix-mul
```bash
gcc -fopenmp -mfma -o matrix-mul.exe matrix-mul.c # -mfma for intrinsics
./matrix-mul.exe
```

task2/fft
```bash
gcc -fopenmp -o fft.exe fft.c
./fft.exe   # produces spectrum.txt
./script.py # produces spectrum.png based on spectrum.txt
# same steps for fft-vectorized.c
```