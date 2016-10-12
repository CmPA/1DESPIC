////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////                                                                        ////
////  Vlasov-Ampere apprach for Particle In Cell simulations: 1D            ////
////                                                                        ////
////  Version for CPU (it does exaclty the same than the GPU version)       ////
////                                                                        ////
////  Diego Gonzalez                                                        ////
////  June of 2016                                                          ////
////                                                                        ////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <iostream>
#include "histogram.h"
#include "iClocks.h"
#include <omp.h>

#ifdef __DP__
  #define NUMBER double
#else
  #define NUMBER float
#endif
#define NNMAX 18000

#if defined(__MIC__) || defined(__KNL__)
  #define REGSIZE 64
  #ifdef __DP__
    #define inREG   16
  #else
    #define inREG   8
  #endif
#else
  #define REGSIZE 32
  #ifdef __DP__
    #define inREG   8
  #else
    #define inREG   4
  #endif
#endif
#ifdef __DP__
  #define ONE     1.0
  #define HALF    0.5
  #define ZERO    0.0
#else
  #define ONE     1.0f
  #define HALF    0.5f
  #define ZERO    0.0f
#endif

//#define WITH_PART_IO

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

// <<1, NumNode>>
void set_current(int nc, NUMBER *jx)
{
  #pragma omp parallel for
  for (int tid=0; tid<nc; tid++) {
    jx[tid] = ZERO;
  }

}

// <<NumCell, NumPart>>
void move_part(int np, NUMBER *xp, NUMBER *vp, NUMBER lbox, NUMBER dt)
{
  NUMBER* __restrict__ l_xp = xp;
  NUMBER* __restrict__ l_vp = vp;

  #pragma omp parallel
  #pragma omp for simd aligned (l_xp,l_vp:REGSIZE)
  for (int tid =0; tid<np; tid++) {
    NUMBER x = l_xp[tid];
    NUMBER v = l_vp[tid];

    x = x + v*dt;

    if (x < ZERO) {
      x = x + lbox;
    }
    if (x >= lbox) {
      x = x - lbox;
    }
    l_xp[tid] = x;
  }
}

// <<NumCell, NumPart>>
void add_current(int np, int nn, NUMBER *Jx, NUMBER *rx, NUMBER *vx, NUMBER _dx, NUMBER q)
{
  NUMBER* __restrict__ l_Jx = Jx;
  NUMBER* __restrict__ l_rx = rx;
  NUMBER* __restrict__ l_vx = vx;

  #pragma omp parallel for
  for (int i=0; i<nn; i++) l_Jx[i] = ZERO;

  #pragma omp parallel
  {

    //NUMBER* tmp = new NUMBER[nn];
    NUMBER tmp[NNMAX] __attribute__((aligned(REGSIZE)));

    #pragma simd aligned (tmp:REGSIZE)
    for (int i=0; i<NNMAX; i++) tmp[i] = 0;

    int lastN = nn-1;
    #pragma omp for simd aligned (l_rx,l_vx,tmp:REGSIZE)
    for (int tid=0; tid<np; tid++) {

      // The particle has to be allways between the first and the last node
      NUMBER x_dx= l_rx[tid]*_dx;
      int   idc1 = int(x_dx);  // Cell on the left of the particle
      int   idc2 = idc1+1;           // Cell of the right of the particle
      NUMBER  w1 = x_dx - idc1;
      NUMBER  w2 = ONE - w1;
    
      NUMBER qv = HALF*q*l_vx[tid];
      NUMBER val1 = qv*w1;
      NUMBER val2 = qv*w2;

      tmp[idc1] += val1;
      tmp[idc2] += val2;
    }

    // This code is an alternative...
    tmp[0] += tmp[lastN];
    tmp[lastN] = tmp[0];
    // ... to this code that does not allow vectorization:
      //if (idc1 == 0) tmp[lastN] += val1;
      //if (idc2 == lastN) tmp[0] += val2;
    //}

    #pragma omp for schedule(static) 
    for (int i=0; i<nn; i++) {
      #pragma omp atomic write
      l_Jx[i] = l_Jx[i] + tmp[i];
    }

    //delete [] tmp;
  }

}	

// <<1, NumNode>>
void calc_field(int nc, NUMBER *Jx, NUMBER *Ex, NUMBER dt)
{
  #pragma omp parallel for
  for (int tid=0; tid<nc; tid++) {
    Ex[tid] -= Jx[tid]*dt;
  }
}

// <<NumCell, NumPart>>
void update_vel(int np, int nn, NUMBER *rx, NUMBER *vx, NUMBER *g_Ex, NUMBER qm, NUMBER dt, NUMBER _dx)
{

  NUMBER* __restrict__ l_rx = rx;
  NUMBER* __restrict__ l_vx = vx;
  NUMBER* __restrict__ l_Ex = g_Ex;

  #pragma omp parallel
  {
    //NUMBER* Ex = new NUMBER[nn];
    NUMBER Ex[NNMAX] __attribute__((aligned(REGSIZE)));

    #pragma simd aligned (Ex:REGSIZE)
    for (int i=0;i<nn;i++) Ex[i] = l_Ex[i];

    #pragma omp for simd aligned (l_rx,l_vx,Ex:REGSIZE)
    for (int tid=0; tid<np; tid++) {
      NUMBER x_dx = l_rx[tid]*_dx;
      int   idc1  = int(x_dx);        // Cell on the left of the particle
      NUMBER  w1  = x_dx - idc1;
      int   idc2  = idc1+1;           // Cell of the right of the particle
      NUMBER  w2  = ONE - w1;

      NUMBER   E = Ex[idc1]*w1+Ex[idc2]*w2;
    
      l_vx[tid] += qm*E*dt;
    }
    //delete [] Ex;
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
    en += HALF*vx[i]*vx[i];
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
    fprintf(fd, "%10.3e %13.6e \n",H.x0 + H.dx*(i+HALF),((NUMBER)H.H[i]/(H.nd*H.dx)));
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
////////////////////////////////////////////////////////////////////////////////
////
////    MAIN 
////
////////////////////////////////////////////////////////////////////////////////
////
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

  if (argc < 2) {
    fprintf(stderr,"\tUse %s <input file> \n\n",argv[0]);
    exit(-1);
  }

  char name[128];

  Clock clocks(7);
  clocks.init(0);
  clocks.start(0);

  InputData data = readInput(argv[1]);

  int numThreads;
  int NumCell = data.nc;
  int NumPart = data.nppc;
  int Cycles  = data.cycles;
  int NumNode = NumCell + 1;

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

  #pragma omp parallel
  {
    #pragma omp master
    numThreads = omp_get_num_threads();
  }

  fprintf(stdout,"#######################################\n"
                 "#  CPU : %s                            \n"
                 "#######################################\n"
                 " sizeof(NUMBER) = %d \n"
                 " Random numbers seed: %ld \n"
                 " Number of threads %d \n"
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
                 " Initial velocity v0 = %g\n"
                 " Thermal velocity vt = %g\n"
                 "#######################################\n"
                 ,data.name
                 ,((int)sizeof(NUMBER))
                 ,seed
                 ,numThreads
                 ,NumCell
                 ,NumNode
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
  size_t memreq = 2*NumNode*sizeof(NUMBER) + 2*TotPart*sizeof(NUMBER);
  fprintf(stdout,"Memory used: %.3f Gb \n", ((double)memreq)/(1024*1024*1024));

  /***********************************************/
  /*           Memory allocation                 */
  /***********************************************/
  NUMBER* Jx  = (NUMBER*) _mm_malloc(NumNode*sizeof(NUMBER),REGSIZE); //new NUMBER[NumNode];
  NUMBER* Ex  = (NUMBER*) _mm_malloc(NumNode*sizeof(NUMBER),REGSIZE); //new NUMBER[NumNode];
  NUMBER* vx  = (NUMBER*) _mm_malloc(TotPart*sizeof(NUMBER),REGSIZE); //new NUMBER[TotPart];
  NUMBER* rx  = (NUMBER*) _mm_malloc(TotPart*sizeof(NUMBER),REGSIZE); //new NUMBER[TotPart];
	
  clocks.stop(0);
  clocks.init(0);
  clocks.init(1);
  clocks.init(2);
  clocks.init(3);
  clocks.init(4);
  clocks.init(5);
  clocks.init(6);

  /***********************************************/
  /*     Initialization                          */
  /***********************************************/
  // Current and fields initialization and warm up
  #pragma omp parallel for
  for (int i=0;i<NumNode;i++) {
    Jx[i] = ZERO;
    Ex[i] = ZERO;
  }

  #pragma omp parallel for
  for (int i=0; i<TotPart;i++) {
    vx[i] = ZERO;
  }

  // Position and velocity initialization and warm up
  #pragma omp parallel for
  for (int i=0; i<TotPart;i++) {
    rx[i] = drand48()*L;   // Random in all box
  }

  for (int i=0; i<TotPart;i+=2) {
    vx[i] =   v0 + drand48()*vth;
  }
  for (int i=1; i<TotPart;i+=2) {
    vx[i] = -(v0 + drand48()*vth);
  }

  NUMBER _dx = ONE/dx;

  clocks.start(1);
  /***********************************************/
  /*                 MAIN LOOP                   */
  /***********************************************/

  for (int c=0; c<Cycles; c++) {
	
    clocks.start(2);
    update_vel(TotPart, NumNode, rx, vx, Ex, qom, dt, _dx);	
    clocks.stop(2);
    clocks.start(3);
    move_part(TotPart, rx, vx, L, dt);
    clocks.stop(3);
    clocks.start(4);
    set_current(NumNode, Jx);
    clocks.stop(4);
    clocks.start(5);
    add_current(TotPart, NumNode, Jx, rx, vx, _dx, q);
    clocks.stop(5);
    clocks.start(6);
    calc_field(NumNode,Jx, Ex, dt);
    clocks.stop(6);

//     if ((c) % data.ioc == 0) {
//       printf("Cycle %d \n",c); 
// #ifdef WITH_HISTOGRAMS
//       sprintf(name,"%s_rx_%d.txt",data.name,c);
//       HisX.clean();
//       recordHis(HisX, rx, TotPart, name);
//       HisV.clean();
//       sprintf(name,"%s_vx_%d.txt",data.name,c);
//       recordHis(HisV, vx, TotPart, name);
// #endif
// #ifdef WITH_PART_IO
//       sprintf(name,"%s_part_%d.txt",data.name,c);
//       FILE* fd = fopen(name,"w");
//       for (int i=0; i<TotPart; i++) 
//         fprintf(fd, "%13.6e %13.6e \n"
//                 ,rx[i], vx[i]);
//       fclose(fd);
// #endif
//     }
  }
  clocks.stop(1);
  printf("Init time: %.3f s\n",clocks.get(0));
  printf("Loop time: %.3f s\n",clocks.get(2)+clocks.get(3)+clocks.get(4)+clocks.get(5)+clocks.get(6));
  printf("-- pmover: %.3f s\n",clocks.get(2)+clocks.get(3));
  printf("-- set cu: %.3f s\n",clocks.get(4));
  printf("-- add cu: %.3f s\n",clocks.get(5));
  printf("-- calc f: %.3f s\n",clocks.get(6));

  /***********************************************/
  /*  Save the results                           */
  /***********************************************/
#ifdef WITH_HISTOGRAMS
  sprintf(name,"%s_rx_%d.txt",data.name, Cycles);
  HisX.clean();
  recordHis(HisX, rx, TotPart, name);
  HisV.clean();
  sprintf(name,"%s_vx_%d.txt",data.name, Cycles);
  recordHis(HisV, vx, TotPart, name);
#endif


#ifdef WITH_PART_IO
  sprintf(name,"%s_part_%d.txt",data.name, Cycles);
  FILE* fd = fopen(name,"w");
  for (int i=0; i<TotPart; i++) 
    fprintf(fd, "%13.6e %13.6e \n"
            ,rx[i], vx[i]);
  fclose(fd);
#endif

  // write output
  sprintf(name,"%s_Jx_GPU.txt",data.name);
  FILE* fd_Jx=fopen(name, "w");
  sprintf(name,"%s_Ex_GPU.txt",data.name);
  FILE* fd_Ex=fopen(name, "w");
	
  // write files
  for (int i=0;i<NumNode;i++) {
    fprintf(fd_Jx, "%13.6e \n",Jx[i]);
    fprintf(fd_Ex, "%13.6e \n",Ex[i]);
  }
  // close files
  fclose(fd_Ex);
  fclose(fd_Jx);

  _mm_free(Jx);
  _mm_free(Ex);
  _mm_free(vx);
  _mm_free(rx);

  return(0);
} 
