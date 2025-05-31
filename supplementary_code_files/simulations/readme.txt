1. Compile C/C++ code as follows

In Windows:
R CMD SHLIB -o dtr.dll dtr.c
R CMD SHLIB -o wsvm.dll svm.cpp svmR.c -static-libstdc++

In Linux:
R CMD SHLIB -o dtr.so dtr.c
R CMD SHLIB -o wsvm.so svm.cpp svmR.c --static-libstdc++

2. Open R and run the script in example.R

All the files should be in the same folder, and example.R will source other R files.
