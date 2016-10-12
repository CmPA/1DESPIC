#ifndef __CLOCKS_H__
#define __CLOCKS_H__

#include <chrono>
#include <algorithm>

#ifdef _OPENMP
  #include <omp.h>
  #define Clock ompClocks
#else
  #define omp_get_wtime() 0
  #define Clock iClocks
#endif

#define Float float

class ompClocks{
  public:
    ompClocks();
    ompClocks(const int nclocks){
      n  = nclocks;
      t  = new double[n];
      ts = new double[n];
      te = new double[n];
      std::fill(t, t+n, 0);
    };
    ~ompClocks() {
      delete [] t;
      delete [] ts;
      delete [] te;
    };

    inline void init (const int i) {t [i] = 0.0;};
    inline void start(const int i) {ts[i] = omp_get_wtime();};
    inline void stop (const int i) {
      te[i] = omp_get_wtime();
      t [i]+= te[i] - ts[i];
    }
    inline double get(const int i) {return t[i];};

  private:
    int n;
    double *t;
    double *ts;
    double *te;
};

class iClocks{
  public:
    iClocks();
    iClocks(const int nclocks) {
      n  = nclocks;
      t  = new Float[n];
      ts = new std::chrono::system_clock::time_point[n];
      te = new std::chrono::system_clock::time_point[n];
      std::fill(t, t+n, 0);
    };
    ~iClocks() {
      delete [] t;
      delete [] ts;
      delete [] te;
    };

    inline void init (int i){t [i] = 0.0;};
    inline void start(int i){ts[i] = std::chrono::high_resolution_clock::now();};
    inline void stop (int i){
      te[i] = std::chrono::high_resolution_clock::now();
      t [i]+= std::chrono::duration_cast<std::chrono::milliseconds>(te[i]-ts[i]).count();
    };
    Float get(int i) {return t[i];};

  private:
    int n;
    Float* t;
    std::chrono::system_clock::time_point* ts;
    std::chrono::system_clock::time_point* te;
};

#endif
