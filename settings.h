#ifndef SETTINGS_H
#define SETTINGS_h

#include <stdlib.h>
#include <stdio.h>
#include <glib.h>



/*Filtersettings*/
gint32 filter_length = 256;		//It is always better to choose a Filterlength in the power of 2 (2,4,8,16,32,64,128,256,512,1024...)
gint32 threshhold    = 9000;		//If the threshhold is very low you may find a lot of peaks which can cause a segmentation fault. Highre the safetyfactor if you have a low threshholid
gint32 input_length  = 40000;		//Set the Length - of the Input-Vector
gint32 expected      = 5;		//How many peaks are expected per sample



gint32 safetyfactor	     = 3;
gint32 notexpect_max	     = 3;	//Maximum amount of notexpected peaks you allow to find during the whole time

/*Modes*/
gint32 safemode = 1;			//If safemode is set the program will exit if there are more peaks found wich are not expected than in the variable notexpect_max are defined.
gint32 filemode = 2;			//0 -> write Output in one text files
					//1 -> write Output into two different files
					//2 -> write Output into the command line
					//3 -> write Output into the command line and in two different .txt file
					//4 -> wirte Output into the command line and in one .txt file
gint32 debugmode = 1;			//Creates an own random input vector with rects, gives you more information about memory allocation and more.
gint32 harddebug = 0;					

char filename[1024] = "energy_time.txt"; //Filename were energy & time will be written in
char filename_t[1024] = "time.txt";	 //Filename were just the time will be written in
char filename_e[1024] = "energy.txt";	 //Filename were just the energy will be written in

/*signal-settings for debug mode*/
gint32 peak_length = 1000;	//width of the rects
gint32 base  = 50000;			//baseline
gint32 peak  = 10000;			//heigth of the rects (base + peak)
gint32 noise = 20;			//Random-White noise which will be added



/*program - variable for loops and testing - do not change.*/
gint32 i = 0;
gint32 counter = 0;
gint32 notexpect = 0;

#endif
