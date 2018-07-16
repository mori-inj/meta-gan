#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DATA_PER_CLASS 1000
#define NUM_OF_CLASS 10
int main()
{
	int i,j;
	char q[1024]={};
	for(i=0; i<DATA_PER_CLASS; i++) {
		for(j=0; j<NUM_OF_CLASS; j++) {
			sprintf(q,"CUDA_VISIBLE_DEVICES=3 python3 gan_per_class_mnist_resize.py %d %d",j,i);
			system(q);
		}
	}

	return 0;
}
