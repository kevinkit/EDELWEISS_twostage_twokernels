#include <stdlib.h>
#include <stdio.h>
#include <glib.h>
#include "ocl.h"
#include "settings.h"
#include "input.h"
#include "OCL_gpu.h"
#include "writeback.h"





int main(int argc, char* argv[])
{




	cl_event event, event_2,write,readback;
	cl_ulong queued, submit,start, end;
	cl_ulong queued_2, submit_2, start_2, end_2;
	cl_ulong write_start, write_end, read_start, read_end;

	double total_time, total_time_2, stopped_time, read_time,write_time;
	int run = 0;
	char timefile[1024] = "time_first_fil2.txt";
	char timefile_2[1024] = "time_second_fil2.txt";
	char summaryfile[1024] = "time_all_fil2.txt";
	char realtime[1024] = "time_all_stopped_fil2.txt";
	char readbacktime[1024] = "time_readback_fil2.txt";
	char writetime[1024] = "time_write_fil2.txt";
	FILE *file;
	FILE *file_2;
	FILE *file_3;

	FILE *file_4;
	FILE *file_5;
	GTimer* timer = g_timer_new();
	//Filterlength 1024 does not need anybody...
	for(; run <= 9; run++)
	{

		switch(run)
		{
			case 0: filter_length = 2; break;
			case 1: filter_length = 4; break;
			case 2: filter_length = 8; break;
			case 3: filter_length = 16; break;
			case 4: filter_length = 32; break;
			case 5: filter_length = 64; break;
			case 6: filter_length = 128; break;
			case 7: filter_length = 256; break;
			case 8: filter_length = 512; break;
			case 9: filter_length = 1024; break;
		}


		sprintf(timefile, "time_first_fil%d.txt", filter_length);
		sprintf(timefile_2, "time_second_fil%d.txt", filter_length);
		sprintf(summaryfile, "time_all_fil%d.txt", filter_length);
		sprintf(realtime,"time_all_stopped_fil%d.txt", filter_length);
		sprintf(readbacktime,"time_readback_fil%d.txt", filter_length);
		sprintf(writetime,"time_write_fil%d.txt",filter_length);




		const size_t SIZE_exec_bit_1 = (input_length - 2*filter_length +1);
		const size_t SIZE_exec_bit_2 = (input_length - 4*filter_length +1);
		const size_t SIZE_input_bit = sizeof(int) *(input_length +1);
		const size_t output_bit_fil_1 = sizeof(int) * (SIZE_exec_bit_1 +5);
		const size_t SIZE_setting_bit = sizeof(int) *4;

		size_t output_bit_on_counts;

		int* filtersettings = (int*) malloc(sizeof(int)*(SIZE_setting_bit));
		int* input_vector = (int*) malloc(sizeof(int) *(SIZE_input_bit));
		int* positions = (int*) malloc(SIZE_input_bit);



		filtersettings[0] = filter_length;
		filtersettings[1] = threshhold;
		filtersettings[2] = input_length;
		filtersettings[3] = 0;

		ocl = ocl_new(CL_DEVICE_TYPE_GPU,1);
		context = ocl_get_context(ocl);
		queue = ocl_get_cmd_queues (ocl)[0];
		clFinish(queue);


		program_1 = ocl_create_program_from_file(ocl, "edel_kernel_one.cl", NULL, &errcode);
		program_2 = ocl_create_program_from_file(ocl, "edel_kernel_two.cl", NULL, &errcode);

		filter_1 = clCreateKernel(program_1,"first_filter", &errcode);
		filter_2 = clCreateKernel(program_2,"second_filter", &errcode);

		settings = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SIZE_setting_bit, filtersettings, &errcode);
		OCL_CHECK_ERROR(errcode);

		input = clCreateBuffer(context, CL_MEM_READ_ONLY, SIZE_input_bit, NULL, &errcode);
		OCL_CHECK_ERROR(errcode);

		output_filter_1 = clCreateBuffer(context, CL_MEM_READ_WRITE, output_bit_fil_1, NULL, &errcode);
		OCL_CHECK_ERROR(errcode);


		srand((unsigned) time( NULL ));
		counter = rand_rects(expected,1,input_length,3*filter_length,3*filter_length,3*filter_length,peak_length,base+peak, input_vector, noise, base, 0,positions);



		output_bit_on_counts = sizeof(gint32) * safetyfactor * 2*((counter + 2));
		int* energy_time = (int*)malloc(output_bit_on_counts);


		output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_bit_on_counts, NULL , &errcode);
		OCL_CHECK_ERROR(errcode);



		clEnqueueWriteBuffer(queue, input, CL_TRUE, 0, SIZE_input_bit, input_vector, 0, NULL, &write);



		OCL_CHECK_ERROR(clSetKernelArg(filter_1, 0, sizeof(cl_mem), &input));
		OCL_CHECK_ERROR(clSetKernelArg(filter_1, 1, sizeof(cl_mem), &output_filter_1));
		OCL_CHECK_ERROR(clSetKernelArg(filter_1, 2, sizeof(cl_mem), &settings));



		size_t local_item_size_1;
		size_t local_item_size_2;
		size_t global_item_size_1 = (size_t) (input_length - 2*filter_length +1);
		size_t global_item_size_2 = (size_t) (input_length - 4*filter_length +1);

		local_item_size_1 = ocl_get_local_size(global_item_size_1, 2,1);
		local_item_size_2 = ocl_get_local_size(global_item_size_2, 2,1);



		if(debugmode != 0)
		{
			if(local_item_size_1!= 0 || local_item_size_2 != 0)
			{
				//some debug if you need...
			}
			else
			{
				FILE* attention;
				attention = fopen("filterlengthbad", "a+");
				if(attention == NULL)
				{
					printf("error in opening debug file \n");
					exit(1);
				}
				fprintf(attention, "The filterlength %d is not good for this filter, choose another filterlength ! \n", filter_length);
				fclose(attention);
				printf("There is no way to fit it evenly divided to workgroups, just let OpenCL do it \n");
			}
			if(harddebug != 0)
			{
				getchar();
			}

		}


		
		g_timer_start(timer);
		//execute first kernel
		if(local_item_size_1 == 0)
		{
			OCL_CHECK_ERROR(clEnqueueNDRangeKernel(queue, filter_1, 1, NULL, &global_item_size_1, NULL, 0, NULL, &event));
		}
		else
		{
			OCL_CHECK_ERROR(clEnqueueNDRangeKernel(queue, filter_1, 1, NULL, &global_item_size_1, &local_item_size_1, 0, NULL, &event));
		}
		

		clFinish(queue);




		OCL_CHECK_ERROR(clSetKernelArg(filter_2, 0, sizeof(cl_mem), &output_filter_1));
		OCL_CHECK_ERROR(clSetKernelArg(filter_2, 1, sizeof(cl_mem), &output));
		OCL_CHECK_ERROR(clSetKernelArg(filter_2, 2, sizeof(cl_mem), &settings));





		//execute second kernel	
		if(local_item_size_2 == 0)
		{
			OCL_CHECK_ERROR(clEnqueueNDRangeKernel(queue, filter_2, 1, NULL, &global_item_size_2, NULL, 0, NULL, &event_2));
		}
		else
		{
			OCL_CHECK_ERROR(clEnqueueNDRangeKernel(queue, filter_2, 1, NULL, &global_item_size_2, &local_item_size_2, 0, NULL, &event_2));
		}

		clFinish(queue);

		g_timer_stop(timer);

		clWaitForEvents(1,&event);
		clWaitForEvents(1,&event_2);
		
		g_timer_stop(timer);
		clEnqueueReadBuffer(queue, output, CL_TRUE, 0, output_bit_on_counts, energy_time, 0, NULL, &readback);
		clEnqueueReadBuffer(queue, settings, CL_TRUE, 0, SIZE_setting_bit, filtersettings, 0, NULL, NULL);

		for(i = 0; i < filtersettings[3]; i++)
		{
			writing_back(filemode, filename, filename_e,filename_t, energy_time,i);
		}




		if(debugmode != 0)
		{
			printf("The Positions are:\n");
			for(i=0; i < counter; i++)
			{
				printf("%d\t", positions[i]);
				printf("note that this postion is the middle of the rect \n");
			}
		}
		//Safetychanges
		if(filtersettings[3] > counter)
		{
			safetyfactor = safetyfactor + 5*(filtersettings[3] - counter);
			if(safetyfactor <= 0)
			{
				safetyfactor = 10;
			}

			notexpect = filtersettings[3] - expected;
			if(safemode != 0 && notexpect >= notexpect_max)
			{
				printf("The Filter found to many peaks it. It expected %d. It found %d times more than expected.\n", expected, notexpect);
				printf("Safemode is on. Exit program \n");
				OCL_CHECK_ERROR(clReleaseMemObject(input));
				OCL_CHECK_ERROR(clReleaseMemObject(output));
				OCL_CHECK_ERROR(clReleaseMemObject(output_filter_1));
				OCL_CHECK_ERROR(clReleaseMemObject(settings));
				OCL_CHECK_ERROR(clReleaseKernel(filter_1));
				OCL_CHECK_ERROR(clReleaseKernel(filter_2));
				OCL_CHECK_ERROR(clReleaseProgram(program_1));
				OCL_CHECK_ERROR(clReleaseProgram(program_2));


				ocl_free(ocl);

				free(input_vector);
				free(energy_time);
				free(positions);
				free(filtersettings);

			}
			else
			{
				printf("The Filter found to many peaks it. It expected %d. It found %d times more than expected \n", expected, notexpect);
			}
		}


		stopped_time = g_timer_elapsed(timer, NULL);

		
                file = fopen(timefile, "a+");
                if(file == NULL)
                {
                        printf("could not open file \n");
                        exit(1);
                }

		
                file_2 = fopen(timefile_2, "a+");
                if(file == NULL)
                {
                        printf("could not open file \n");
                        exit(1);
                }



		file_4 = fopen(readbacktime, "a+");
		if(file_4 == NULL)
		{
			printf("could not open file \n");
			exit(1);
		}	


		file_5 = fopen(writetime, "a+");
		if(file_5 == NULL)
		{
			printf("could not open file \n");
			exit(1);
		}



		clGetEventProfilingInfo(readback, CL_PROFILING_COMMAND_START,sizeof(read_start), &read_start, NULL);
		clGetEventProfilingInfo(readback, CL_PROFILING_COMMAND_END, sizeof(read_end), &read_end, NULL);

		clGetEventProfilingInfo(write, CL_PROFILING_COMMAND_START,sizeof(write_start), &write_start,NULL);
		clGetEventProfilingInfo(write, CL_PROFILING_COMMAND_END, sizeof(write_end), &write_end, NULL);



		read_time = read_end - read_start;
		fprintf(file_4, "%0.3f\t", (read_time/1000000.0));


		write_time = write_end - write_start;
		fprintf(file_5, "%0.3f\t", (read_time/1000000.0));
		
	




		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT,sizeof(submit), &submit, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,sizeof(start), &start, NULL);
		clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,sizeof(end), &end, NULL);

		total_time = end - start;
		printf("total time = %f\n",total_time);
		printf("Queued %llu\t Submit: %llu\t Start: %llu\t End: %llu \t \n",queued,submit,start,end);



		printf("Runtime %0.3f\n",(total_time/1000000.0));
		fprintf(file, "%0.3f\t", (total_time/1000000.0));




		clGetEventProfilingInfo(event_2, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued_2, NULL);
		clGetEventProfilingInfo(event_2, CL_PROFILING_COMMAND_SUBMIT,sizeof(submit), &submit_2, NULL);
		clGetEventProfilingInfo(event_2, CL_PROFILING_COMMAND_START,sizeof(start), &start_2, NULL);
		clGetEventProfilingInfo(event_2, CL_PROFILING_COMMAND_END,sizeof(end), &end_2, NULL);

		total_time_2 = end_2 - start_2;
		printf("total time = %f\n",total_time_2);
		printf("Queued %llu\t Submit: %llu\t Start: %llu\t End: %llu \t \n",queued_2,submit_2,start_2,end_2);



		printf("Runtime %0.3f\n",(total_time_2/1000000.0));
		fprintf(file_2, "%0.3f\t", (total_time_2/1000000.0));


		file_3 = fopen(realtime, "a+");
		if(file_3 == NULL)
		{
			printf("could not open file \n");
			exit(1);
		}

		fprintf(file_3, "%f\t", stopped_time*1000);

		FILE* another_file;
		another_file = fopen(summaryfile,"a+");
		if(another_file == NULL)
		{
			printf("could not open file \n");
			exit(1);
		}

		

		fprintf(another_file, "%0.3f\t", ((total_time_2 + total_time)/1000000.0));
		fclose(another_file);


		OCL_CHECK_ERROR(clReleaseMemObject(input));
		OCL_CHECK_ERROR(clReleaseMemObject(output));
		OCL_CHECK_ERROR(clReleaseMemObject(output_filter_1));
		OCL_CHECK_ERROR(clReleaseMemObject(settings));
		OCL_CHECK_ERROR(clReleaseKernel(filter_1));
		OCL_CHECK_ERROR(clReleaseKernel(filter_2));
		OCL_CHECK_ERROR(clReleaseProgram(program_1));
		OCL_CHECK_ERROR(clReleaseProgram(program_2));



		ocl_free(ocl);

		fclose(file);
		fclose(file_2);
		fclose(file_3);	
		fclose(file_4);
		fclose(file_5);	
		free(energy_time);





		free(filtersettings);
		free(input_vector);
		free(positions);



	}


}

