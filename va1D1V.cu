/** @file va1D1V.cpp
 *  @brief Vlasov-Ampere 1D electrostatic particle-in-cell code for GPU
 *  @author Diego GONZALEZ-HERRERO <diego.gonzalez@kuleuven.be>
 *  @date June, 2016
 *
 *   Copyright (c) 2016 KU Leuven University
 *   Some rights reserved. See COPYING, AUTHORS.
 *
 *  @license GPL-3.0 <https://opensource.org/licenses/GPL-3.0>
 */

////////////////////////////////////////////////////////////////////////////////
////                                                                        ////
////  compile with CUDA3.2: nvcc -arch=sm_21 main.cu                        ////
////                                                                        ////
////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include "histogram.h"

#define NUMBER float

#define WITH_PART_IO

typedef struct {
  char   name[256];
  int    nc;
  int    nppc;
  int    cycles;
  NUMBER dt;
  NUMBER dx;
  NUMBER rho;
  NUMBER qom;
  NUMBER vth;
  NUMBER v0;
  int    ioc;
  } InputData;


 __device__ inline void atomicAdd (double *address, double value)
{
  unsigned long long oldval, newval, readback;

  oldval = __double_as_longlong(*address);
  newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  while ((readback=atomicCAS((unsigned long long *)address, oldval, newval)) != oldval)
  {
    oldval = readback;
    newval = __double_as_longlong(__longlong_as_double(oldval) + value);
  }
}

// <<1, NumNodes>>
__global__ void set_current( NUMBER *jx, int nodThre)
{
//if ((threadIdx.x == 0) && (blockIdx.x == 0)) printf("set current\n");
  int tid = (threadIdx.x + blockIdx.x * blockDim.x)*nodThre;
  for (int i=0; i<nodThre; i++, tid++) jx[tid] = 0;
}

// <<NumCell, threads>>
__global__ void add_current(NUMBER *Jx, NUMBER *rx, NUMBER *vx, NUMBER dx, NUMBER q, int parThre)
{
//if ((threadIdx.x == 0) && (blockIdx.x == 0)) printf("Add current\n");
  // Particle index
  int tid = (threadIdx.x + blockIdx.x * blockDim.x)*parThre;
  // Indes of last node:
  int lastN = gridDim.x;
  for (int i=0; i<parThre; i++, tid++) {
    NUMBER x = rx[tid];
    NUMBER v = vx[tid];

    // The particle has to be allways between the first and the last node
    int idc1 = int(x/dx);  // Cell on the left of the particle
    int idc2 = idc1+1;           // Cell of the right of the particle
    NUMBER x_dx= x/dx;
    NUMBER w1 = x_dx - idc1;
    NUMBER w2 = idc2 - x_dx;
  
    NUMBER qv = q*v/2.0;
    NUMBER val1 = qv*w1;
    NUMBER val2 = qv*w2;
  
    atomicAdd(&Jx[idc1], val1);    // sm_20 or more
    atomicAdd(&Jx[idc2], val2);    // sm_20 or more
  
    if (idc1 == 0) atomicAdd(&Jx[lastN], val1);
    if (idc2 == lastN) atomicAdd(&Jx[0], val2);
  }	
}
// <<1, NumNodes>>
__global__ void calc_field( NUMBER *Jx, NUMBER *Ex, NUMBER dt, int nodThre)
{
//if ((threadIdx.x == 0) && (blockIdx.x == 0)) printf("Calc field\n");
  int tid = (threadIdx.x + blockIdx.x * blockDim.x)*nodThre;
  for (int i=0; i<nodThre; i++, tid++) Ex[tid] -= Jx[tid]*dt;
}

// <<NumCell, threads>>
__global__ void update_part(NUMBER *rx, NUMBER *vx, NUMBER *Ex, NUMBER lbox, NUMBER qm, NUMBER dt, 
                           NUMBER dx, int parThre)
{
//if ((threadIdx.x == 0) && (blockIdx.x == 0)) printf("Update part\n");
  int tid = (threadIdx.x + blockIdx.x * blockDim.x)*parThre;

  for (int i=0; i<parThre; i++, tid++) {
    NUMBER x = rx[tid];
    NUMBER v = vx[tid];

    // Interpolate the electric field
    int idc1 = int(x/dx);  // Cell on the left of the particle
    int idc2 = idc1+1;           // Cell of the right of the particle
    NUMBER x_dx= x/dx;
    NUMBER w1 = x_dx - idc1;
    NUMBER w2 = idc2 - x_dx;
    NUMBER E = Ex[idc1]*w1 + Ex[idc2]*w2;
   
    // Update the velocity
    v += qm*E*dt;

    // Update the position
    x = x + v*dt;
    while (x < 0.0) x = x + lbox;
    while (x >= lbox) x = x - lbox;

    rx[tid] = x;
    vx[tid] = v;
  }
}


NUMBER Eenergy(NUMBER *Ex, int n, NUMBER dx)
{
  NUMBER en = 0;
  NUMBER fac = dx/(8*M_PI);
  for (int i=0; i<n; i++) {
    en += fac*Ex[i]*Ex[i];
  }
  return en;
}

NUMBER Kenergy(NUMBER *vx, int n)
{
  NUMBER en = 0;
  for (int i=0; i<n; i++) {
    en += 0.5*vx[i]*vx[i];
  }
  return en;
}


void recordHis(Histogram H, NUMBER* data, int n, const char* name)
{
  for (int i=0; i<n; i++) 
    H.count(data[i]);

  FILE* fd = fopen(name, "w");

  fprintf(fd,"# nd = %d \n", H.nd);
  fprintf(fd,"#H0: %d \n", H.H0);
  for (int i=0;i<H.nc;i++) {
    fprintf(fd, "%10.3e %13.6e \n",H.x0 + H.dx*(i+0.5),((NUMBER)H.H[i]/(H.nd*H.dx)));
  }
  fprintf(fd,"#H1: %d \n", H.H1);

  fclose(fd);
}


InputData readInput(const char* file) 
{
  FILE* fd = fopen(file,"r");
  if (NULL == fd) {
    fprintf(stderr,"ERROR: file %s cannot be opened for reading\n\n",file);
    exit(-1);
  }
  printf("Reading %s \n",file);

  InputData data;

  char line[256];
  char *txt;
  
  strncpy(line, file, sizeof(line));
  txt = strtok(line, "."); 
  strncpy(data.name, txt, sizeof(data.name));
  
  // First line not used
  fgets(line, sizeof(line), fd);

  fgets(line, sizeof(line), fd);
  txt = strtok(line, ":"); txt = strtok(NULL, ":");
  data.nc = atoi(txt);

  fgets(line, sizeof(line), fd);
  txt = strtok(line, ":"); txt = strtok(NULL, ":");
  data.nppc = atoi(txt);

  fgets(line, sizeof(line), fd);
  txt = strtok(line, ":"); txt = strtok(NULL, ":");
  data.cycles = atoi(txt);

  fgets(line, sizeof(line), fd);
  txt = strtok(line, ":"); txt = strtok(NULL, ":");
  data.dt = atof(txt);

  fgets(line, sizeof(line), fd);
  txt = strtok(line, ":"); txt = strtok(NULL, ":");
  data.dx = atof(txt);

  fgets(line, sizeof(line), fd);
  txt = strtok(line, ":"); txt = strtok(NULL, ":");
  data.rho = atof(txt);

  fgets(line, sizeof(line), fd);
  txt = strtok(line, ":"); txt = strtok(NULL, ":");
  data.qom = atof(txt);

  fgets(line, sizeof(line), fd);
  txt = strtok(line, ":"); txt = strtok(NULL, ":");
  data.vth = atof(txt);

  fgets(line, sizeof(line), fd);
  txt = strtok(line, ":"); txt = strtok(NULL, ":");
  data.v0 = atof(txt);

  fgets(line, sizeof(line), fd);
  txt = strtok(line, ":"); txt = strtok(NULL, ":");
  data.ioc = atoi(txt);

  fclose(fd);

  return data;
}

////////////////////////////////////////////////////////////////////////////////
////
////    MAIN 
////
////////////////////////////////////////////////////////////////////////////////
////
////  Input file format: 
////           # First line as a comment
////           Number of cells              : 256
////           Number of particles per cell : 1000
////           Number of cycles             : 8000
////           Time step (dt)               : 0.1
////           Cell size (dx)               : 0.04
////           Charge density               : 1.0
////           Charge mass ratio (qom)      : -1.0
////           Thermal velocity             : 0.02
////           Drift velocity               : 0.1
////           IO cycles                    : 1000
////
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) 
{

  if (argc < 3) {
    fprintf(stderr,"\tUse %s <input file> <Threads per block>\n\n",argv[0]);
    exit(-1);
  }

  // Setup timer
  cudaEvent_t start, stop;
  float cutime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  InputData data = readInput(argv[1]);
  int threads = atoi(argv[2]);

  int NumCell  = data.nc;
  int NumPart  = data.nppc;
  int Cycles   = data.cycles;
  int NumNodes = NumCell+1;

  if (0 != NumPart % threads) {
    fprintf(stderr,"ERROR: Number of particles per cell is not a multiple of the number of threads\n");
    exit(-1);
  }
  if (0 != NumNodes % threads) {
    fprintf(stderr,"ERROR: Number of nodes is not a multiple of the number of threads\n");
    exit(-1);
  }
 
  int ParThread = NumPart/threads;
  int NodThread = NumNodes/threads;

  // Histograms
  Histogram HisE;
  Histogram HisV;
  Histogram HisX;


  int TotPart = NumPart*NumCell;
  const NUMBER dt  = data.dt;
  const NUMBER dx  = data.dx;
  const NUMBER L   = NumCell*dx;
  const NUMBER rho = data.rho;

  const NUMBER qom = data.qom;
  const NUMBER v0  = data.v0;
  const NUMBER vth = data.vth;
  const NUMBER q   = (qom/fabs(qom))*(rho/NumPart)*dx;

  HisE.init(-1,1);
  HisV.init(-2*v0,2*v0);
  HisX.init(0,L);
  
  long long seed = 12313132;
  srand48(seed); 

  fprintf(stdout,"#######################################\n"
                 "#  GPU:  %s                            \n"
                 "#######################################\n"
                 " sizeof(NUMBER) = %d \n"
                 " Random numbers seed: %ld \n"
                 " Number of threads per block: %d\n"
                 "#######################################\n"
                 " Number of cells:           %d \n"
                 " Number of nodes:           %d \n"
                 " Particles per cell:        %d \n"
                 " Total number of particles: %d \n"
                 " Cycles:                    %d \n"
                 "#######################################\n"
                 " Time step dt = %g \n"
                 " Box size  L = %g \n"
                 " Delta X dx=%g\n"
                 " Charge/mass qom=%g\n"
                 " Charge q=%g\n"
                 " Initial velocity v0  = %g\n"
                 " Thermal velocity vth = %g\n"
                 "#######################################\n"
                 ,data.name
                 ,((int)sizeof(NUMBER))
                 ,seed
                 ,threads
                 ,NumCell
                 ,NumNodes
                 ,NumPart
                 ,TotPart
                 ,Cycles
                 ,dt
                 ,L
                 ,dx
                 ,qom
                 ,q
                 ,v0
                 ,vth
                 );
                  
  fprintf(stdout,"*** Memory allocation ***\n");

  size_t mem_tot  = 0;
  size_t mem_free = 0;
  cudaMemGetInfo  (&mem_free, & mem_tot);
  fprintf(stdout,"Total memory: %.3f Gb    Free memory %.3f Gb\n", ((double)mem_tot/(1024*1024*1024)) , ((double)mem_free/(1024*1024*1024)));
  size_t memreq = 2*NumNodes*sizeof(NUMBER) + 2*TotPart*sizeof(NUMBER);
  fprintf(stdout,"Memory used: %.3f Gb \n", ((double)memreq)/(1024*1024*1024));

  /***********************************************/
  /*           Memory allocation                 */
  /***********************************************/
  // HOST:
  NUMBER h_Jx[NumNodes];
  NUMBER h_Ex[NumNodes];
  NUMBER h_vx[TotPart];
  NUMBER h_rx[TotPart];
  // DEVICE:
  NUMBER *d_Jx, *d_Ex, *d_rx, *d_vx;
  cudaMalloc( (void**)&d_Jx, NumNodes*sizeof(NUMBER));
  cudaMalloc( (void**)&d_Ex, NumNodes*sizeof(NUMBER));
  cudaMalloc( (void**)&d_rx, TotPart*sizeof(NUMBER));
  cudaMalloc( (void**)&d_vx, TotPart*sizeof(NUMBER));
	
  /***********************************************/
  /*     Initialization and copy to GPU          */
  /***********************************************/
  // Current and fields
  for (int i=0;i<NumNodes;i++) {
    h_Jx[i] = 0.0;
    h_Ex[i] = 0.0;
  }


  NUMBER sbp = L/(TotPart+1);

  // Position and velocity
  for (int i=0; i<TotPart;i++) {
#define RANDOM_SORT
#ifdef RANDOM_SORT
    h_rx[i] = drand48()*L;   // Random in all box
#else
    h_rx[i] = i*sbp;
#endif
    if (i % 2 == 0) h_vx[i] =   v0 + drand48()*vth;
    else            h_vx[i] = -(v0 + drand48()*vth);

//pow(-1.0,i)*(v0 + vt * (rand() % 1)); // base velocity + noise from thermal velocity
  }
 
  // Copy to the GPU
  cudaMemcpy( d_Jx, h_Jx, NumNodes*sizeof(NUMBER), cudaMemcpyHostToDevice);
  cudaMemcpy( d_Ex, h_Ex, NumNodes*sizeof(NUMBER), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rx, h_rx, TotPart*sizeof(NUMBER), cudaMemcpyHostToDevice);
  cudaMemcpy( d_vx, h_vx, TotPart*sizeof(NUMBER), cudaMemcpyHostToDevice);

  /***********************************************/
  /*                 MAIN LOOP                   */
  /***********************************************/
  //Start kernel timer
  cudaEventRecord(start, 0);

  char name[128];
  for (int c=0; c<Cycles; c++) {
	
    update_part<<<NumCell,threads>>>(d_rx, d_vx, d_Ex, L, qom, dt, dx, ParThread);	
    set_current<<<1, threads>>>(d_Jx, NodThread);
    add_current<<<NumCell,threads>>>(d_Jx, d_rx, d_vx, dx, q, ParThread);
    calc_field<<<1, threads>>>(d_Jx, d_Ex, dt, NodThread);

    if ((c) % data.ioc == 0) {
      printf("Cycle %d \n",c);
#if defined(WITH_HISTOGRAMS) || defined(WITH_PART_IO)
      cudaMemcpy(h_rx, d_rx, TotPart*sizeof(NUMBER), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_vx, d_vx, TotPart*sizeof(NUMBER), cudaMemcpyDeviceToHost);
#endif

#ifdef WITH_HISTOGRAMS
      sprintf(name,"%s_rx_%d.txt",data.name, c);
      HisX.clean();
      recordHis(HisX, h_rx, TotPart, name);
      HisV.clean();
      sprintf(name,"%s_vx_%d.txt",data.name, c);
      recordHis(HisV, h_vx, TotPart, name);
#endif

#ifdef WITH_PART_IO
      sprintf(name,"%s_part_%d.txt",data.name, c);
      FILE* fd = fopen(name,"w");
      for (int i=0; i<TotPart; i++)
        fprintf(fd, "%13.6e %13.6e \n"
                ,h_rx[i], h_vx[i]);
      fclose(fd);
#endif
    }
  }

  /***********************************************/
  /*  Copy the results from GPU to the host      */
  /***********************************************/

  // copy the results back to the host memory
  cudaMemcpy(h_Jx, d_Jx, NumNodes*sizeof(NUMBER), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_Ex, d_Ex, NumNodes*sizeof(NUMBER), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_rx, d_rx, TotPart*sizeof(NUMBER), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_vx, d_vx, TotPart*sizeof(NUMBER), cudaMemcpyDeviceToHost);
	
  // Stop kernel timer
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cutime, start, stop);
  printf("Kernel execution time: %.3f s\n", cutime/1000);

#ifdef WITH_HISTOGRAMS
  sprintf(name,"%s_rx_%d.txt",data.name, Cycles);
  HisX.clean();
  recordHis(HisX, h_rx, TotPart, name);
  HisV.clean();
  sprintf(name,"%s_vx_%d.txt",data.name, Cycles);
  recordHis(HisV, h_vx, TotPart, name);
#endif

#ifdef WITH_PART_IO
  sprintf(name,"%s_part_%d.txt",data.name, Cycles);
  FILE* fd = fopen(name,"w");
  for (int i=0; i<TotPart; i++)
    fprintf(fd, "%13.6e %13.6e \n"
            ,h_rx[i], h_vx[i]);
  fclose(fd);
#endif

  // free the device memory
  cudaFree(d_Jx);
  cudaFree(d_Ex);
  cudaFree(d_rx);
  cudaFree(d_vx);


  // write output
  sprintf(name,"%s_Jx_GPU.txt",data.name);
  FILE* fd_Jx=fopen(name, "w");
  sprintf(name,"%s_Ex_GPU.txt",data.name);
  FILE* fd_Ex=fopen(name, "w");
	
  // write files
  for (int i=0;i<NumNodes;i++) {
    fprintf(fd_Jx, "%13.6e \n",h_Jx[i]);
    fprintf(fd_Ex, "%13.6e \n",h_Ex[i]);
  }
  // close files
  fclose(fd_Ex);
  fclose(fd_Jx);

  return(0);
} 
