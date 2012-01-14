#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <lapacke.h>
#include <cblas.h>

#ifndef NODE_H
#define NODE_H


int historylength = 12;
unsigned int network_size = 2; /*not including bias or inputs*/
int debug = 1;

/* 
 * to treat this with lapacke we will need to have the nodes in an array when 
 * we do training however there will be several arrays. The way it shall work 
 * is the network is an array of arrays. Thus the network is.
 */

double ** network;
double * W;
/* 
 * oder doesn't matter in the network so inputs will be first input_list_size 
 * nodes in network
 */
int input_list_size;
/*
 * order matters for the output since it might matter if the output is an input
 * so let's say that the output start at the last index of the true nodes and
 * work backwards. We may overlap with input but we will warn the user.
 */
int output_list_size;
unsigned int D;

void W_init ()
{	
	if (debug) printf ("Initializing W randomly\n");
	unsigned int i;
	W = (double *) malloc (sizeof (double) * D * network_size);
	for (i=0; i< (D * network_size); ++i)
	{
		W [i] = (double)rand()/RAND_MAX;
		/*if (debug) printf ("%d: W[%u] = %f\n",clock, i, W [i]);*/
	}
	if (debug) printf ("%ld: W initialized:\n\
			%ld @%p\n",\
			clock, sizeof (double) * D * network_size, W);
}

void W_term ()
{
	if (debug) printf ("Terminating W\n");
	free (W);
	if (debug) printf ("W terminated\n");
}

double get_input (unsigned int t, unsigned int input)
{
	int success = 1;
	double retrieved_input = (double)input / (t + 1.);
	if (!success) 
	{
		if (debug) printf ("Failed to get input for:\n\
				t=%u\n\
				input=%u\n",t,input);
		return 0.;
	}
	return retrieved_input;
}

void update_input ()
{
	unsigned int i,j;
	for (i=0; i<historylength; ++i)
	{
		for (j=0; j<input_list_size; ++j)
		{
			network [i][network_size + 1 + j] = get_input (i,j);
		}
	}
}

void network_init ()
{
	/*
	 * check that we have properly defined all parameters first
	 */

	if (network_size <= 1) return;
	if (input_list_size > network_size) return;
	if (output_list_size > network_size) return;
	if (historylength == 0) return;
	/*
	 * should now query in the interface if network=NULL
	 */
	D = 2 * network_size + 1;
	if (debug) printf ("Initializing network with parameters:\n\
			history length = %d\n\
			network size = %u\n", \
			historylength, network_size);
	unsigned int i,j;
	network = (double **) malloc (sizeof (double *) * historylength);
	for (i=0; i<historylength; ++i)
	{
		/*bias plus one input for every node*/
		network [i] = (double *) malloc (sizeof (double) * D);
		for (j=0; j<D; ++j)
		{
			network [i][j] = 0.;
		}
	}
	if (debug) printf ("Network initialized:\n\
			%ld @%p\n", \
			sizeof (double) * D * historylength, network);
	if (debug) printf ("Setting inputs to network\n");
	update_input ();
	if (debug) printf ("Successfully set all inputs\n");
	if (input_list_size + output_list_size > network_size) \
		printf ("%u overlapping input and output\n", \
				output_list_size + input_list_size -\
			       	network_size);
	W_init ();
}

void network_term ()
{
	unsigned int i;
	if (debug) printf ("Terminating network\n");
	for (i=0; i<historylength; ++i)
	{
		free (network [i]);
	}
	free (network);
	if (debug) printf ("Freed up memory\n");
	W_term ();
}

double sigmoid (double X)
{
	/*
	 * I should use tanh (t/2) as outlined in my paper but this is quicker
	 */
	return tanh (X/20.);
}

void network_forward (unsigned int t)
{
	unsigned int i;
	if (t >= historylength)
	{
		if (debug) printf ("Too many forward steps! Only room for:\
				%d\n", historylength);
		return;
	}
	printf ("Moving forward to t = %u\n", t);
	/*
	 * Y_k (t) = <e_k|W^T|Y(t-1)>
	 * remember w_x0 = 0 and W|_(2n+1)x(n)
	 * say we remove first colum since it must be 0 anyway
	 * so we just have <e_k| is 1x(n) 0 everywhere except 1 at k
	 * |Y(t-1)> is thus (2n+1)x1
	 */
	for (i=0; i< network_size; ++i)
	{
		network [t][i] = sigmoid (\
				cblas_ddot (D, \
					&(W [D * i]), 1, \
					network [t - 1], 1));
	}
	printf ("Output for t = %u\n",t);
	for (i=0; i<output_list_size; ++i)
	{
		printf ("\tO[%u] = %f\n", i + 1, \
				network [t][network_size + 1 - i]);
	}
}

	  







/*
 *
 * OLD STUFF
 *
 * 

struct node
{
  unsigned int id;
  double ** history;
};

void init_node (struct node * n, unsigned int id)
{
  unsigned int i;
  n->id = id;
  n->history = (double **) malloc (sizeof (double *) * historylength);
  for (i=0; i<historylength; ++i)
  {
    n->history [i] = (double *) malloc (sizeof (double) * 2);
    n->history [i][0] = 0.;
    n->history [i][1] = 0.;
  }
  if (debug) printf ("node (id) %u set, (history) %ld bytes, @%p\n", id, sizeof (double) * 2 * historylength, n->history);
}

void term_node (struct node * n)
{
  unsigned int i;
  unsigned int id = n->id;
  for (i=0; i<historylength; ++i)
  {
    free (n->history [i]);
  }
  free (n->history);
  if (debug) printf ("freed up node %u\n", id);
}

struct network
{
  unsigned int num;
  struct node * nodes;
  double ** weightmatrix;
  long time;
};

void init_network (struct network * net, unsigned int num)
{
  unsigned int i,j;
  // set number of nodes
  printf ("building network, @%p\n...\n",net);
  printf ("set number of nodes to %u\n",num);
  net->num = num;
  // malloc room for nodes
  printf ("nodes are %ld bytes each\n", sizeof (struct node)); 
  net->nodes = (struct node *) malloc (sizeof (struct node) * num);
  printf ("node array is %ld bytes, @%p\n", sizeof (struct node) * num, net->nodes);
  printf ("initializing each node\n");
  for (i=0; i<num; ++i)
  {
    // init each node using (i) as id
    init_node (&(net->nodes [i]), i);
  }
  // size of weight matrix is (n+1) x (n+1) for extra bias
  net->weightmatrix = (double **) malloc (sizeof (double *) * (num + 1));
  printf ("initialized weight matrix, %ld bytes, @%p\n", sizeof (double) * (num + 1) * (num + 1), net->weightmatrix);
  for (i=0; i<(num + 1); ++i)
  {
    net->weightmatrix [i] = (double *) malloc (sizeof (double) * (num + 1));
    for (j=0; j<(num + 1); ++j)
    {
      if (i == 0)
      {
        net->weightmatrix [i][j] = 0.;
      } else {
      // assign random number between 0 and 1.
        net->weightmatrix [i][j] = (double) rand() / (double) RAND_MAX;
      }
      if (debug) printf ("weightmatrix [%u][%u] = %lf\n",i,j,net->weightmatrix [i][j]);
    }
  }
  net->time = 0L;
}

void term_network (struct network * net)
{
  unsigned int i;
  printf ("terminating network @%p\n", net);
  printf ("network has %u nodes, time %ld\n", net->num, net->time);
  for (i=0; i<net->num; ++i)
  {
    term_node (&(net->nodes [i]));
  }
  free (net->nodes);
  for (i=0; i<(net->num + 1); ++i)
  {
    free (net->weightmatrix [i]);
  }
  free (net->weightmatrix);
}

double sigmoid (double x)
{
  // standard logistic function ranging from -1 to 1. y(t)=2/(1+%e^-t)-1
  return 2 / (1 + exp (-x)) - 1;
}

double calculateinput (struct network * net, unsigned int nodeid)
{


};
*/

#endif
