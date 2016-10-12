all: cpu knl knc cpudp knldp kncdp

clean: cleancpu cleanknc cleanknl txt

txt:
	rm *.txt

cleancpu:
	rm va1D1V-*.intel64

cleanknc:
	rm va1D1V-*.mic

cleanknl:
	rm va1D1V-*.knl

gpu: va1D1V-GPU.cu histogram.h iClocks.h
	nvcc -arch=sm_21 va1D1V-GPU.cu -o va1D1V-GPU 

cpu-dev: va1D1V.cpp histogram.h iClocks.h
	icpc va1D1V.cpp -o va1D1V-SP.intel64 -fopenmp -std=c++11 -qopt-report5 -O2 -g

knc-dev: va1D1V.cpp histogram.h iClocks.h
	icpc va1D1V.cpp -o va1D1V-SP.knl -D__KNC__ -fopenmp -std=c++11 -qopt-report5 -O2 -g -mmic

knl-dev: va1D1V.cpp histogram.h iClocks.h
	icpc va1D1V.cpp -o va1D1V-SP.knl -D__KNL__ -fopenmp -std=c++11 -qopt-report5 -O2 -g -axMIC-AVX512

kncdp: va1D1V.cpp histogram.h iClocks.h
	icpc va1D1V.cpp -o va1D1V-DP.mic -D__KNC__ -D__DP__ -fopenmp -std=c++11 -O3 -mmic

knc: va1D1V.cpp histogram.h iClocks.h
	icpc va1D1V.cpp -o va1D1V-SP.mic -D__KNC__ -fopenmp -std=c++11 -O3 -mmic

knldp: va1D1V.cpp histogram.h iClocks.h
	icpc va1D1V.cpp -o va1D1V-DP.knl -D__KNL__ -D__DP__ -fopenmp -std=c++11 -O3 -axMIC-AVX512

knl: va1D1V.cpp histogram.h iClocks.h
	icpc va1D1V.cpp -o va1D1V-SP.knl -D__KNL__ -fopenmp -std=c++11 -O3 -axMIC-AVX512

cpudp: va1D1V.cpp histogram.h iClocks.h
	icpc va1D1V.cpp -o va1D1V-DP.intel64 -D__DP__ -fopenmp -std=c++11 -O3

cpu: va1D1V.cpp histogram.h iClocks.h
	icpc va1D1V.cpp -o va1D1V-SP.intel64 -fopenmp -std=c++11 -O3
