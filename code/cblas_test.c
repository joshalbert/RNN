#include <stdio.h>
#include <cblas.h>

int main ()
{
	double x[5] = {1.,2.,3.,4.,5.};
	double y[5] = {1,1,1,1,1};
	printf ("%lf\n", cblas_ddot (2, x, 2, y, 2));
	return 0;
}
