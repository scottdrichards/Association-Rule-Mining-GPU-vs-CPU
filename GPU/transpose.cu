#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "kernel.cu"

#define TILE_WIDTH 4

__host__

int main()  {
/***************  READING DATABASE FILE  **************************/
    int i,j;
    int num_of_itemsets, num_of_transactions;

    FILE *fp;
    fp = fopen("database.txt","r");
    fscanf(fp,"%d",&num_of_transactions);
    fscanf(fp,"%d",&num_of_itemsets);

    char *db[num_of_transactions][num_of_itemsets];
    for (i = 0; i < num_of_transactions; i++)  {
        for (j = 0; j < num_of_itemsets; j++)  {
	    db[i][j] = (char *) malloc (100 * sizeof(char));
	    memcpy(db[i][j],"",100 * sizeof(char));
	}
    }

    i = 0, j = 0;
    const char delim[20] = "\n ,{}:";
    char *token;
    int approx_size = (num_of_itemsets+2) * 100 * sizeof(char);
    char dummy[approx_size];
    memset(dummy,0,approx_size);

    while(fgets(dummy,approx_size,fp) != NULL)  {
	token = strtok(dummy,delim);
	if (token == NULL)  continue;   // To flush out the termination null char
	token = strtok(NULL,delim);
	while (token != NULL)  {
	    memcpy(db[i][j],token,100 * sizeof(char));
	    token = strtok(NULL,delim);
	    j++;
	}
	j = 0;
	i++;
    }		

    for (i = 0; i < num_of_transactions; i++)  {
        for (j = 0; j < num_of_itemsets; j++)  {
            printf("%s ",db[i][j]);
        }
        printf("\n");
    }

    fclose(fp);
/*******************  END  ***********************/

/*******************  CUDA HOST TO DEVICE COPY  ***************************************/
    char *db_cu[num_of_transactions][num_of_itemsets];
    char *db_tran_cu[num_of_itemsets][num_of_transactions];

    for(i = 0; i < num_of_transactions; i++)  {
	for (j = 0; j < num_of_itemsets; j++)  {
	    cudaMalloc((void **)&db_cu[i][j],100*sizeof(char));
	}
    }

    for (i = 0; i < num_of_itemsets; i++)  {
        for (j = 0; j < num_of_transactions; j++)  {
            cudaMalloc((void **)&db_tran_cu[i][j],100*sizeof(char));
	}
    }

    for (i = 0; i < num_of_transactions; i++)  {
        for (j = 0; j < num_of_itemsets; j++)  {
	    cudaMemcpy(db_cu[i][j],db[i][j],100 * sizeof(char),cudaMemcpyHostToDevice);
	}
    }

    char *db_tran[num_of_itemsets][num_of_transactions];
    for (i = 0; i < num_of_itemsets; i++)  {
        for (j = 0; j < num_of_transactions; j++)  {
            db_tran[i][j] = (char *) malloc (100 * sizeof(char));
            memcpy(db_tran[i][j],"",100 * sizeof(char));
        }
    }

/*******************  END  *****************************/

/*******************  CUDA KERNEL LAUNCH  *************************************/
    dim3 dimGrid( ((num_of_transactions - 1)/TILE_WIDTH) + 1, ((num_of_itemsets - 1)/TILE_WIDTH) + 1, 1);
    dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
    cudaTranspose <<< dimGrid, dimBlock >>> (num_of_transactions, num_of_itemsets, &db_cu[0][0], &db_tran_cu[0][0]);
/*******************  END  *****************************/

/*******************  CUDA DEVICE TO HOST COPY  ********************************************************/
    for (i = 0; i < num_of_itemsets; i++)  {
        for (j = 0; j < num_of_transactions; j++)  {
            cudaMemcpy(db_tran[i][j],db_tran_cu[i][j],100*sizeof(char),cudaMemcpyDeviceToHost);
        }
    }

    for (i = 0; i < num_of_itemsets; i++)  {
        for (j = 0; j < num_of_transactions; j++)  {
            printf("%s ",db_tran[i][j]);
        }
        printf("\n");
    }
/*******************  END  *****************************/

//  free(db);
    return 0;
}

