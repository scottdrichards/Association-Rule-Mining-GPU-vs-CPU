#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include "kernel_transpose.h"
#include "kernel_EClen2.h"

__host__ int main()  {
/***************  READING DATABASE FILE  **************************/
    int i,j,k;
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

    i = 0;
    j = 0;
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
	memset(dummy,0,approx_size);
    }		

    printf("++++++++++  Database  ++++++++++\n");
    for (i = 0; i < num_of_transactions; i++)  {
        for (j = 0; j < num_of_itemsets; j++)  {
            printf("%s ",db[i][j]);
        }
        printf("\n");
    }

    fclose(fp);
/*******************  END  ***********************/

/*******************  OTHER INITIALIZATIONS  ********************************/
    char *db_tran[num_of_itemsets*num_of_transactions][num_of_transactions+1];
    for (i = 0; i < num_of_itemsets*num_of_transactions; i++)  {
        for (j = 0; j < num_of_transactions+1; j++)  {
            db_tran[i][j] = (char*) malloc (100 * sizeof(char));
            memcpy(db_tran[i][j],"",100 * sizeof(char));
        }
    }

    int EC_size = num_of_itemsets * (num_of_itemsets - 1) / 2;
    char *EC_len2[num_of_transactions*EC_size][(num_of_transactions+2)*27];
    for (i = 0; i < num_of_transactions*EC_size; i++)  {
	for (j = 0; j < (num_of_transactions+2)*27; j++)  {
	    EC_len2[i][j] = (char*) malloc (100 * sizeof(char));
            memcpy(EC_len2[i][j],"",100 * sizeof(char));
	}
    }
/*******************  END  ***********************/

/*******************  CUDA HOST TO DEVICE COPY  ***************************************/
    dim3 db_gpuDim(100,num_of_transactions,num_of_itemsets);
    cudaExtent db_gpuVolSizeBytes = make_cudaExtent(sizeof(char) * db_gpuDim.x, db_gpuDim.y, db_gpuDim.z);
    cudaPitchedPtr db_gpuPitchPtr;
    cudaMalloc3D(&db_gpuPitchPtr,db_gpuVolSizeBytes);
    cudaMemset3D(db_gpuPitchPtr,0,db_gpuVolSizeBytes);

    dim3 db_tran_gpuDim(100,num_of_itemsets*num_of_transactions,num_of_transactions+1);
    cudaExtent db_tran_gpuVolSizeBytes = make_cudaExtent(sizeof(char) * db_tran_gpuDim.x, db_tran_gpuDim.y, db_tran_gpuDim.z);
    cudaPitchedPtr db_tran_gpuPitchPtr;
    cudaMalloc3D(&db_tran_gpuPitchPtr,db_tran_gpuVolSizeBytes);
    cudaMemset3D(db_tran_gpuPitchPtr,0,db_tran_gpuVolSizeBytes);

    dim3 EC_len2_gpuDim(100,num_of_transactions*EC_size,(num_of_transactions+2)*27);
    cudaExtent EC_len2_gpuVolSizeBytes = make_cudaExtent(sizeof(char) * EC_len2_gpuDim.x, EC_len2_gpuDim.y, EC_len2_gpuDim.z);
    cudaPitchedPtr EC_len2_gpuPitchPtr;
    cudaMalloc3D(&EC_len2_gpuPitchPtr,EC_len2_gpuVolSizeBytes);
    cudaMemset3D(EC_len2_gpuPitchPtr,0,EC_len2_gpuVolSizeBytes);

    char *db_ptr = (char *)db_gpuPitchPtr.ptr;
    size_t db_pitch = db_gpuPitchPtr.pitch;
    size_t db_slicePitch = db_pitch * num_of_transactions;
    char *db_current_slice;
    char *db_element;

    for (i = 0; i < num_of_transactions; i++)  {
        for (j = 0; j < num_of_itemsets; j++)  {
	    db_current_slice = db_ptr + j * db_slicePitch;
	    db_element = (char *)(db_current_slice + i * db_pitch);
	    cudaMemcpy(db_element,db[i][j],100*sizeof(char),cudaMemcpyHostToDevice); 
	}
    }
/*******************  END  *****************************/

/*******************  CUDA KERNEL LAUNCH  *************************************/
    dim3 dimGrid(ceil(sqrt(num_of_transactions)),ceil(sqrt(num_of_transactions)),1); 
    dim3 dimBlock(ceil(sqrt(num_of_itemsets)),ceil(sqrt(num_of_itemsets)),1);

    // Launch kernel to find the transpose of db
    cudaTranspose <<< dimGrid, dimBlock >>> (db_gpuPitchPtr, db_gpuDim, db_tran_gpuPitchPtr);
    cudaDeviceSynchronize();

    // Launch kernel to find all the entries of the Equivalent Classes of upto length 2 and sort them simultaneously
    cudaEClen2Part1 <<< dimGrid, dimBlock >>> (db_gpuPitchPtr, db_gpuDim, db_tran_gpuPitchPtr,EC_len2_gpuPitchPtr, EC_size);
    cudaDeviceSynchronize();
/*******************  END  *****************************/

/*******************  CUDA DEVICE TO HOST COPY  ********************************************************/
    char *db_tran_ptr = (char*)db_tran_gpuPitchPtr.ptr;
    size_t db_tran_pitch = db_tran_gpuPitchPtr.pitch;
    size_t db_tran_slicePitch = db_tran_pitch * num_of_itemsets * num_of_transactions;
    char *db_tran_current_slice;
    char *db_tran_element;

    for (i = 0; i < num_of_itemsets*num_of_transactions; i++)  {
        for (j = 0; j < num_of_transactions+1; j++)  {
            db_tran_current_slice = db_tran_ptr + j * db_tran_slicePitch;
            db_tran_element = (char *)(db_tran_current_slice + i * db_tran_pitch);
            cudaMemcpy(db_tran[i][j],db_tran_element,100*sizeof(char),cudaMemcpyDeviceToHost);
        }
    }

    printf("++++++++++  Transpose of the Database  ++++++++++\n");
    for (i = 0; i < num_of_itemsets*num_of_transactions; i++)  {
	if (strcmp(db_tran[i][0],"") == 0)  continue;
	printf("%s: ",db_tran[i][0]);
        for (j = 1; j < num_of_transactions+1; j++)  {
            printf("-%s-",db_tran[i][j]);
        }
        printf("\n");
    }

    char *EC_len2_ptr = (char *)EC_len2_gpuPitchPtr.ptr;
    size_t EC_len2_pitch = EC_len2_gpuPitchPtr.pitch;
    size_t EC_len2_slicePitch = EC_len2_pitch * num_of_transactions * EC_size;
    char *EC_len2_slice;
    char *EC_len2_element;

    for (i = 0; i < num_of_transactions*EC_size; i++)  {
	for (j = 0; j < (num_of_transactions+2)*27; j++)  {
	    EC_len2_slice = EC_len2_ptr + i * EC_len2_pitch;
	    EC_len2_element = (char *)(EC_len2_slice + j * EC_len2_slicePitch);
	    cudaMemcpy(EC_len2[i][j],EC_len2_element,100*sizeof(char),cudaMemcpyDeviceToHost);
	}
    }

    j = 0;
    printf("++++++++++  Equivalent Class of upto length 2  ++++++++++\n");
    for (k = 0; k < 27; k++)  {
        for (i = 0; i < num_of_transactions*EC_size; i++)  {
            if (strcmp(EC_len2[i][(k*(num_of_transactions+2))+0],"") == 0)  continue;
	    printf("%s%s: ",EC_len2[i][(k*(num_of_transactions+2))+0],EC_len2[i][(k*(num_of_transactions+2))+1]);
            for (j = 2; j < num_of_transactions+2; j++)  {
                printf("-%s-",EC_len2[i][(k*(num_of_transactions+2))+j]);
            }
            printf("\n");
	}
    }
/*******************  END  *****************************/
    printf("++++++++++  End  ++++++++++\n");

    cudaFree(db_gpuPitchPtr.ptr);
    cudaFree(db_tran_gpuPitchPtr.ptr);

//  free(db);
    return 0;
}

