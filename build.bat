@echo off
REM Windows build without make (needs MinGW-w64 / MSYS2 gcc on PATH)
if not exist bin mkdir bin
gcc -O3 -march=native -std=c11 -Wall -Wextra -fopenmp -Iinclude src\main.c src\mergesort2.c src\mergesort4.c src\util.c -o bin\pdcsort.exe
if errorlevel 1 exit /b 1
echo Built bin\pdcsort.exe
