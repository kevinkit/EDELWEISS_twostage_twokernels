#ifndef WRITEBACK_H
#define WRITEBACK_H

#include <stdlib.h>
#include <stdio.h>

int writing_back(int filemode, char* filename, char* filename_e, char* filename_t, int* energy_time,int i)
{
    FILE* file;
    FILE* file_2;

	printf("\n\n\n");
    switch(filemode)
	{
		case 0: file = fopen(filename, "a+");
			if(file == NULL)
			{
				printf("Could not open %s file \n", filename);
				return -1;
			}
			fprintf(file, "%d\t%d\t", energy_time[2*i], energy_time[2*i+1] + 2*filter_length);
			fclose(file);
			break;
		case 1: file = fopen(filename_e,"a+");
			if(file == NULL)
			{
				printf("Could not open %s file \n", filename);
				return -1;
			}
			file_2 = fopen(filename_t,"a+");
			if(file == NULL)
			{
				printf("Could not open %s file \n", filename);
				return -1;
			}
			fprintf(file, "%d\t", energy_time[2*i]);
			fprintf(file_2, "%d\t", energy_time[2*i+1] + 2*filter_length);
			fclose(file);
			fclose(file_2);
			break;
		case 2: printf("energy[%i] = %d\n",i,energy_time[2*i]);							
			printf("time = %d\n", energy_time[2*i+1] + 2*filter_length);
			break;
		case 3: file = fopen(filename_e,"a+");
			if(file == NULL)
			{
				printf("Could not open %s file \n", filename);
				return -1;
			}
			file_2 = fopen(filename_t,"a+");
			if(file == NULL)
			{
				printf("Could not open %s file \n", filename);
				return -1;
			}
			fprintf(file, "%d\t", energy_time[2*i]);
			fprintf(file_2, "%d\t", energy_time[2*i+1] + 2*filter_length);
			fclose(file);
			fclose(file_2);
			printf("energy[%i] = %d\n",i,energy_time[2*i]);	
			printf("time = %d\n", energy_time[2*i+1] + 2*filter_length);
			break;
		case 4: file = fopen(filename, "a+");
			if(file == NULL)
			{
				printf("Could not open %s file \n", filename);
				return -1;
			}
			fprintf(file, "%d\t%d\t", energy_time[2*i], energy_time[2*i+1] + 2*filter_length);
			fclose(file);
			printf("energy[%i] = %d\n",i,energy_time[2*i]);	
			printf("time = %d\n", energy_time[2*i+1] + 2*filter_length);
			break;
		default: return 1;
	  
			
	}
	
      
      
}

#endif
