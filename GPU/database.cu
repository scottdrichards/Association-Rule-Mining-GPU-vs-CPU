#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define MAX_ITEMSETS 8
#define NUM_TRANSACTIONS 16

__host__ 

int main()  {
    int i,j,k;
    int num_itemsets;

    FILE *fp;
    fp = fopen("database.txt","w");
    fprintf(fp,"%d\n%d\n",NUM_TRANSACTIONS,MAX_ITEMSETS);

    for (i = 1; i <= NUM_TRANSACTIONS; i++)  {
	fprintf(fp,"t%d: {",i);
	k = 65;
	num_itemsets = (rand() % (MAX_ITEMSETS - 1)) + 1;
	for (j = 1; j <= num_itemsets; j++)  {
	    if (j == num_itemsets)  fprintf(fp,"%c}\n",k);
	    else  fprintf(fp,"%c,",k);
	    k++;
	}
    }

    fclose(fp);
    return 0;
}

