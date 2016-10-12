#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#define NCAN 100

class Histogram {
public:
	long	nc;	// Número de canales
	long	nd;	// Número de datos introducidos
	double	x0, x1;	// Extremos del intervalo cubierto
	double	dx;	// Tamaño de la celda

	double	sx;	// Suma de todos los valores tomados
	double	sx2;	// Suma de cuadrados de todos los
			// valores tomados

	long	H0;	// Valor del intervalo (-infinito, x0)
	long	H1;	// Valor del intervalo (x1, +infinito)
	long	H[NCAN];	// Valores de las nc celdas

	// Métodos
	//
	void clean() {
	    nd = 0;
	    H0 = 0;
	    H1 = 0;
	    sx  = 0.;
	    sx2 = 0.;
	    for (int i=0; i<NCAN; i++) H[i] = 0;
	}


	void init(double X0, double X1) {
	    nc = NCAN;
	    clean();
	    x0 = X0;
	    x1 = X1;
	    dx = (x1-x0)/NCAN;
	}

	Histogram() { init(0.,1.); }

	~Histogram() { }

	void count(double x) {
	    int i = (int) floor((x-x0)/dx);
	    if (i < 0) H0++;
	    else if (i<NCAN) H[i]++;
	    else H1++;
	    sx  += x;
	    sx2 += x*x;
	    nd++;
	}

	void acumulate(const Histogram* h) {
	    nd  += h->nd;
	    H0  += h->H0;
	    H1  += h->H1;
	    sx  += h->sx;
	    sx2 += h->sx2;
	    for (int i=0; i<NCAN; i++) H[i] += h->H[i];
	}

	double average() {
	    if (nd>0) return sx/nd; else return 0.;
	}

	double sigma2() {
	    if (nd>0) return (sx2-sx*sx/nd)/nd; else return -1.;
	}
};
#endif
