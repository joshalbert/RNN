#include <stdio.h>
#include <stdlib.h>
#include "node.h"

int main (const int argc, const char ** argv)
{
	unsigned int i;
  	if (argc != 2)
  	{
    		printf ("USAGE %s debug(0|1)\n",argv[0]);
    		return 1;
  	}
  	
  	/*sscanf (argv[2],"%d", &debug);*/
  	
  	if (!debug) printf ("No debugging enabled\n");

  	
	input_list_size = 55;
	output_list_size = 15;
	historylength = 100;
	network_size = 100;
	network_init ();
	if (network == NULL)
	{
		if (debug) printf ("failed to initialize network, \
				possibly to do with parameters.\n");
		return 1;
	}
	for (i=0; i<historylength; ++i)
	{
		network_forward (i+1);
	}
	network_term ();
  	return 0;
}
