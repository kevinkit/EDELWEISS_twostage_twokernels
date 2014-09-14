#include <stdlib.h>
#include <stdio.h>
#include <limits.h>



//function will set random vectors at random points 
int rand_rects(int max, int min,int input_size, int zerogap,int rectgap, int endgap, int rect_size, int A, int* input_vector, int noise, int offset ,int percent, int* positions)
{
        //Checking if there are any rects set
	printf("Settings are: \n Max amount peaks %d \n Min amount peaks %d \n input_size %d \nzerogap: %d \n rectgap %d \n endgap %d \n rect_size %d \n Amplitude %d \n noise %d \n offset %d \n", max,min,input_size, zerogap, rectgap,endgap,rect_size,A,noise,offset);     

	int rec_half= rect_size/2;
	
	FILE* file;
	file = fopen("positions.txt", "a+");
	if(file == NULL)
	{
		return -1;
	}

	if(min == 0)
        {
                printf("Choose min larger than zero! \n");
                return -1;
        }
        if(rect_size%2 != 0)
        {
                printf("Choose an even rect size! \n");
                return -1;
        }

        int i = 0;
        int x = 0;
        int y = 0;
        int anz = 0;
	int var = 0;
        int b = 0;
        int real_noise = 0;
	int breaker;
	int deb;

        for(;;)
	{
		anz = (rand()%max);
                if(anz >= min && anz <= max)
		{
			break;
        	}
	}




        int* pos = (int*) malloc(sizeof(int) * (anz +1));

        for(i=1; i < anz+1; i++)
        {
                y = 0;
                for(;;)
                {
                        pos[i-1] = (rand()%input_size);
			if(pos[i-1] > (zerogap + rect_size) && (pos[i-1] < (input_size - rect_size - endgap)) && pos[i-1] < input_size)
                        {
				printf("first codns okay \n");
				for(x=1; x <= i; x++)
                                {
					printf("loop \n");	
					if((pos[i-1] > pos[x-1] + rectgap + rect_size) || pos[i-1] < pos[x-1] + rectgap + rect_size)
                                        {
                                         	printf("conditioning okay for %d need %d\n",x-1,y);        
						y++;
                                        }
					printf("y = %d, i = %d, anz = %d\n",y,i-1,anz);	
					if(y == i)
					{
						var = 1;
					}
                                }
                        }
			breaker++;

                        if(var == 1){
                                printf("The positon[%i] is %d\n",i-1,pos[i-1]);
				positions[i-1] = pos[i-1];
				var = 0;
				y = 0;
				break;
			}
			else
			{
				y = 0;
			}
			
			if(breaker > INT_MAX)
			{
				printf("gave up - change your settigns for signal creation \n");
				anz = i;
				FILE* file_2;
				file_2 = fopen("errorlog.txt", "w+");
				for(deb=0; deb < i; deb++)
				{
					fprintf(file_2, "Not a good positoning at %d\n", pos[deb]);
				}
				fclsoe(file_2);
				break;	
			}
                }
        }
        
	//Last check
	for(i=0; i < anz; i++)
	{
		if(pos[i] > (input_size - rec_half))
		{
			printf("Something went wrong \n");
			getchar();
			free(pos);
			return -1;
		}
		if(pos[i] < rec_half)
		{
			printf("something went wrong \n");
			getchar();
			free(pos);
			return -1;
		}
	}

	for(i=0; i < anz; i++)
	{	
		fprintf(file,"%d\t", pos[i]);
	}
        fclose(file);
                                                                                                       
        for(i=0; i < input_size;i++)
	{
		input_vector[i] = offset;
	}

	for(i=0; i < anz; i++)                                                                                                  
        {                                                                                                                       
                        for(x=0; x < rec_half; x++)                                                                          
                        {                                                                                                       
                                input_vector[x+pos[i]] = A;                                                                     
                        }                                                                                                       
                        for(x=0; x < rec_half; x++)                                                                          
                        {                                                                                                       
                                input_vector[x+pos[i] - rec_half] = A;                                                       
                        }                                                                                                       
        }                                                                                                                       
                                                                                                                                
                                                                                                                                
        if(percent == 1)                                                                                                     
        {                                                                                                                       
                b = 100/noise;                                                                                                  
                for(i = 0; i < input_size; i++)                                                                                 
                {                                                                                                               
                        input_vector[i] =  input_vector[i] + rand()%(A/b) - A/b;                                                                 
                }                                                        
        }                                                                
        else                                                             
        {                                                                
                for(i=0; i < input_size;i++)                             
                {                                                        
                        input_vector[i] = input_vector[i] + rand()%noise - noise;          
                }                                                        
        }                                                                
        
        return anz;                                                                 
	free(pos);
} 
