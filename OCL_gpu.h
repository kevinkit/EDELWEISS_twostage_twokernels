#ifndef OCL_GPU_H
#define OCL_GPU_H

#include "ocl.h"



OclPlatform *ocl;
cl_context context;
cl_int errcode;
cl_mem input;
cl_mem output;
cl_mem output_filter_1;
cl_mem settings;
cl_program program_1;
cl_program program_2;
cl_kernel filter_1;
cl_kernel filter_2;
cl_command_queue queue;



size_t
ocl_get_local_size(size_t global_size, size_t min, int trials)
{
        size_t local_size = min;
	int i = 0;


        	for(;;)
        	{
                	if(global_size%local_size == 0)
                	{
                        	
                        	if(local_size == global_size)
                        	{
                                	printf("na toll .... ! \n \n \n");
                                	return 0;
                        	}
				i++;
				if(i == trials)
				{	
                        		return local_size;
                		}
			}
                	else
                	{
                        	local_size++;
                	}


        }
}




#endif

