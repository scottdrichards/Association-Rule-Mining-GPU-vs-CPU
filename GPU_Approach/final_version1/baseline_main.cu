/*********************  ASSUMPTIONS  ******************************
1) num_of_itemsets is the Upper limit of num of items per transaction; 
   i.e. num of items per transaction <= num_of_itemsets
2) string length = 100 characters max
*********************  END  ******************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include "kernel_transpose.h"
#include "kernel_EClen2.h"

__host__ int main()  {
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

    i = 0;
    j = 0;
    const char delim[20] = "\n ,{}:";
    char *token;
    int approx_size = (num_of_itemsets+2) * 100 * sizeof(char);  // +2 is due to delimeters , { } : etc. Size was 1<n<2, so I took 2
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
    // There can be atmost num_of_itemsets*num_of_transactions different elements - HIGH SKEW db case
    // Column 0 is dedicated for elements, column 1 to num_of_transactions+1 are dedicated for attendance

    for (i = 0; i < num_of_itemsets*num_of_transactions; i++)  {
        for (j = 0; j < num_of_transactions+1; j++)  {
            db_tran[i][j] = (char*) malloc (100 * sizeof(char));
            memcpy(db_tran[i][j],"",100 * sizeof(char));
        }
    }

    int EC_size = num_of_itemsets * (num_of_itemsets - 1) / 2;
/*************  Redundant  *****************
    char *EC_len2[num_of_transactions*EC_size][(num_of_transactions+2)*27];

    for (i = 0; i < num_of_transactions*EC_size; i++)  {
	for (j = 0; j < (num_of_transactions+2)*27; j++)  {
	    EC_len2[i][j] = (char*) malloc (100 * sizeof(char));
            memcpy(EC_len2[i][j],"",100 * sizeof(char));
	}
    }
*************  End  ******************/
    char *output_element_len2[num_of_transactions*EC_size*2][2];  // say (A,B) then o/p is A => B and B => A, so *2 num of rows
    for (i = 0; i < num_of_transactions*EC_size*2; i++)  {
        for (j = 0; j < 2; j++)  {
            output_element_len2[i][j] = (char*) malloc (100 * sizeof(char));
            memcpy(output_element_len2[i][j],"",100 * sizeof(char));
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

    // There can be atmost num_of_transactions*EC_size different length=2 elements - HIGH SKEW db case
    // Column 0 & 1 are dedicated for the elements,
    // column 2 to num_of_transactions+2 are dedicated for attendance

    /*******************  SPACE vs (PERFORMANCE & CODING SIMPLICITY) TRADE-OFF  *************************
        I chose performance and coding simplicity over space
        space*27 means 27 different sets - 26 sets for english alphabets, 1 set for non-alphabets which include numbers also

        1) Implementing a sorting algorithm in GPU is very complicated
        2) Fastest available sorting algorithm takes O(nlog n) time complexity
        Solution --> Using 27 sets will not only take O(const) time complexity but also make coding simpler
    ********************  END  *********************/

    dim3 output_element_len2_gpuDim(100,num_of_transactions*EC_size*2,2);
    cudaExtent output_element_len2_gpuVolSizeBytes = make_cudaExtent(sizeof(char) * output_element_len2_gpuDim.x, output_element_len2_gpuDim.y, output_element_len2_gpuDim.z);
    cudaPitchedPtr output_element_len2_gpuPitchPtr;
    cudaMalloc3D(&output_element_len2_gpuPitchPtr,output_element_len2_gpuVolSizeBytes);
    cudaMemset3D(output_element_len2_gpuPitchPtr,0,output_element_len2_gpuVolSizeBytes);

    dim3 output_support_confidence_len2_gpuDim(1,num_of_transactions*EC_size*2,8);
    cudaExtent output_support_confidence_len2_gpuVolSizeBytes = make_cudaExtent(sizeof(float) * output_support_confidence_len2_gpuDim.x, output_support_confidence_len2_gpuDim.y, output_support_confidence_len2_gpuDim.z);
    cudaPitchedPtr output_support_confidence_len2_gpuPitchPtr;
    cudaMalloc3D(&output_support_confidence_len2_gpuPitchPtr,output_support_confidence_len2_gpuVolSizeBytes);
    cudaMemset3D(output_support_confidence_len2_gpuPitchPtr,0.0000,output_support_confidence_len2_gpuVolSizeBytes);

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
    // Each block will corresponds to 1 transaction
    // and the threads within a block correspond to items in that transaction
    // I purposely set them in these ways to avoid/miminize control divergence problems - Better Performance
    // This also makes coding slightly simpler (not much), btw! 
    dim3 dimGrid(ceil(sqrt(num_of_transactions)),ceil(sqrt(num_of_transactions)),1); 
    dim3 dimBlock(ceil(sqrt(num_of_itemsets)),ceil(sqrt(num_of_itemsets)),1);

    // Launch kernel to find the transpose of db
    cudaTranspose <<< dimGrid, dimBlock >>> (db_gpuPitchPtr, db_gpuDim, db_tran_gpuPitchPtr);
    cudaDeviceSynchronize();

    // Launch kernel to find all the entries of the Equivalent Classes of upto length 2 and sort them simultaneously
    cudaEC_len2 <<< dimGrid, dimBlock >>> (db_gpuPitchPtr, db_gpuDim, db_tran_gpuPitchPtr, EC_len2_gpuPitchPtr, output_element_len2_gpuPitchPtr, output_support_confidence_len2_gpuPitchPtr, EC_size);
    cudaDeviceSynchronize();
/*******************  END  *****************************/

/*******************  CUDA DEVICE TO HOST COPY  ***********************************************/
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
/*************  Redundant  *************
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

    int k;
    j = 0;
    printf("++++++++++  Equivalent Class of upto length 2  ++++++++++\n");
    for (k = 0; k < 27; k++)  {
        for (i = 0; i < num_of_transactions*EC_size; i++)  {
            if (strcmp(EC_len2[i][(k*(num_of_transactions+2))+0],"") == 0)  continue;
	    printf("{%s,%s}: ",EC_len2[i][(k*(num_of_transactions+2))+0],EC_len2[i][(k*(num_of_transactions+2))+1]);
            for (j = 2; j < num_of_transactions+2; j++)  {
                printf("-%s-",EC_len2[i][(k*(num_of_transactions+2))+j]);
            }
            printf("\n");
	}
    }
*******************  End  *******************/
    char *op_len2_ptr = (char *)output_element_len2_gpuPitchPtr.ptr;
    size_t op_len2_pitch = output_element_len2_gpuPitchPtr.pitch;
    size_t op_len2_slicePitch = op_len2_pitch * num_of_transactions * EC_size * 2;
    char *op_len2_slice;
    char *op_len2_element;

    for (i = 0; i < num_of_transactions*EC_size*2; i++)  {
        for (j = 0; j < 2; j++)  {
	    op_len2_slice = op_len2_ptr + i * op_len2_pitch;
	    op_len2_element = (char *)(op_len2_slice + j * op_len2_slicePitch);
	    cudaMemcpy(output_element_len2[i][j],op_len2_element,100*sizeof(char),cudaMemcpyDeviceToHost);
	}
    }

    float *op_sc_len2_ptr = (float *)output_support_confidence_len2_gpuPitchPtr.ptr;
    size_t op_sc_len2_pitch = output_support_confidence_len2_gpuPitchPtr.pitch;
    size_t op_sc_len2_slicePitch = op_sc_len2_pitch * num_of_transactions * EC_size * 2;
    float *op_sc_len2_slice;
    float *op_sc_len2_element;

    float output_support_confidence_len2[num_of_transactions*EC_size*2][2];

    for (i = 0; i < num_of_transactions*EC_size*2; i++)  {
        for (j = 0; j < 2; j++)  {
            op_sc_len2_slice = op_sc_len2_ptr + i * op_sc_len2_pitch;
            op_sc_len2_element = (float *)(op_sc_len2_slice + j * op_sc_len2_slicePitch);
            cudaMemcpy(&output_support_confidence_len2[i][j],&op_sc_len2_element[0],sizeof(float),cudaMemcpyDeviceToHost);
        }
    }

    j = 0;
    printf("++++++++++  Results upto length 2  ++++++++++\n");
    printf("{X} => {Y}:  support %%         confidence %% \n\n");
    for (i = 0; i < num_of_transactions*EC_size*2; i++)  {
        if (strcmp(output_element_len2[i][0],"") == 0)  continue;
        printf("{%s} => {%s}:  ",output_element_len2[i][0],output_element_len2[i][1]);
        for (j = 0; j < 2; j++)  {
                printf("%.4f %%        ",output_support_confidence_len2[i][j]);
        }
        printf("\n");
    }
/*******************  END  *****************************/
    printf("++++++++++  End  ++++++++++\n");

    cudaFree(db_gpuPitchPtr.ptr);
    cudaFree(db_tran_gpuPitchPtr.ptr);

//  free(db);
    return 0;
}

