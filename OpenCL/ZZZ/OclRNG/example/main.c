/*	Parallel Random Number Generator in OpenCL
	Copyright (C) 2011 Giorgos Arampatzis, Angelos Athanasopoulos

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>. */


#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <CL/cl.h>

#define MT_DATA_FILE "../mtParameters/mtDATA/mtDATA_521_32_30.bin"
#define STRING_SIZE 200

#define CLERR \
	if (clerr != CL_SUCCESS) {                              \
		printf("opencl error %d,\t file: %s   line: %d\n", clerr,__FILE__,__LINE__); \
		exit(-1);                                            \
	  }

typedef struct {
	cl_platform_id		platform_id;
	cl_device_id		device_id; /* compute device id */
	cl_context		context; /* compute context */
	cl_command_queue	commands; /* compute command queue */
	cl_program		program;
	cl_kernel		kernel;

	int nWI;	/* # of Work Items */
	int nWG;	/* # size of Work Group */
}dev_info;


/* same as mt_struct without the work vector */
typedef struct { 
    cl_uint aaa;
    cl_int mm,nn,rr,ww;
    cl_uint wmask,umask,lmask;
    cl_int shift0, shift1, shiftB, shiftC;
    cl_uint maskB, maskC;
    cl_int i;
}mt_struct_naked;

/****************************************************/

/* functions handling structs */

void InitSystemvars( int argc, char **argv, dev_info *di, int *length );
void readMTdata( mt_struct_naked *mts, int Nmts );
char * load_program_source(const char *filename);
void printMTnaked( mt_struct_naked *mtss, int count );

/****************************************************/

int
main(int argc, char **argv) {

	int device = 1, i, j, alength;
	int *seeds;
	unsigned int *output;
	uint32_t *states;
	size_t global; 
	size_t local;
	FILE *fp;

	const char *filename = "RNGkernels.cl";
	const char *KName2= "RNG";
	const char *KName= "initRNG";
	char foutput[100];
	char  buildOptions[STRING_SIZE];
	char *program_source;

	mt_struct_naked *mts;
	
	cl_int	clerr;
	dev_info di;
	cl_mem in1,in2,in3,in4; 
	cl_mem out1; 

	/* Set compute device */
	clerr = clGetPlatformIDs(1, &(di.platform_id),NULL); 
	CLERR
	
	clerr = clGetDeviceIDs(di.platform_id, device ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &(di.device_id), NULL);
	CLERR	

	di.context = clCreateContext(0, 1, &(di.device_id), NULL, NULL, &clerr);
	CLERR

	di.commands = clCreateCommandQueue(di.context, di.device_id, 0, &clerr);
	CLERR

	/* Set initial data */
	InitSystemvars(argc,argv,&di,&alength);

	/* Read PRNG data from file and set the initial seeds for the threads */
	mts = (mt_struct_naked*) malloc(di.nWI*sizeof(mt_struct_naked));
	readMTdata(mts,di.nWI);

	seeds = (int*)malloc(sizeof(int)*di.nWI);
	srand(time(NULL));
	for(i=0; i<di.nWI; i++) 
		seeds[i] = rand();

	/* Setting arrays output and states*/
	output = (unsigned int *) malloc(sizeof(unsigned int) * alength *di.nWI);	
	if(!output ) {
		printf("Out of memory!\n");
		exit(EXIT_FAILURE);
	}

        states = (uint32_t *) malloc(sizeof(uint32_t)*mts[0].nn*di.nWI);
	if(!states ) {
		printf("Out of memory!\n");
		exit(EXIT_FAILURE);
	}

	printf("Total # of samples = %d\n",alength*di.nWI);

	/*Pass array sizes as defines using -D flag */
	sprintf(buildOptions,"-D MT_NN=%d", mts[0].nn);

	program_source = load_program_source(filename);
	di.program = clCreateProgramWithSource(di.context, 1, (const char **) &program_source, NULL, &clerr);
	CLERR

	clerr = clBuildProgram(di.program, 0, NULL, buildOptions, NULL, NULL);
	CLERR

	/* Create buffer objects */ 
	in1  = clCreateBuffer(di.context, CL_MEM_READ_ONLY  ,sizeof( mt_struct_naked )*di.nWI, NULL, &clerr); CLERR
	in2  = clCreateBuffer(di.context, CL_MEM_READ_ONLY  ,sizeof( int )*di.nWI, NULL, &clerr); CLERR
	in3  = clCreateBuffer(di.context, CL_MEM_READ_ONLY  ,sizeof( int ), NULL, &clerr); CLERR
        in4  = clCreateBuffer(di.context, CL_MEM_READ_WRITE ,sizeof( uint32_t )*mts[0].nn*di.nWI, NULL, &clerr); CLERR
	out1 = clCreateBuffer(di.context, CL_MEM_WRITE_ONLY ,sizeof( int )*alength*di.nWI, NULL, &clerr); CLERR

	/* Enqueue buffers to the selected device */
	clerr = clEnqueueWriteBuffer(di.commands, in1 , CL_TRUE, 0,sizeof( mt_struct_naked )*di.nWI , mts , 0, NULL, NULL); CLERR
	clerr = clEnqueueWriteBuffer(di.commands, in2 , CL_TRUE, 0, sizeof(int) * di.nWI, seeds, 0, NULL, NULL); CLERR
	clerr = clEnqueueWriteBuffer(di.commands, in3 , CL_TRUE, 0, sizeof(int), &alength, 0, NULL, NULL); CLERR
        clerr = clEnqueueWriteBuffer(di.commands, in4 , CL_TRUE, 0, sizeof(uint32_t)*mts[0].nn*di.nWI,states , 0, NULL, NULL); CLERR
	clerr = clEnqueueWriteBuffer(di.commands, out1, CL_TRUE, 0, sizeof(unsigned int) * alength*di.nWI, output, 0, NULL, NULL); CLERR

	/*Set initRNG kernel and its arguments */
	di.kernel = clCreateKernel(di.program, KName, &clerr);
	CLERR

	clerr  = clSetKernelArg(di.kernel, 0, sizeof(cl_mem), &in1); CLERR
	clerr |= clSetKernelArg(di.kernel, 1, sizeof(cl_mem), &in4); CLERR
	clerr |= clSetKernelArg(di.kernel, 2, sizeof(cl_mem), &in2); CLERR

	global = di.nWI; local = di.nWG;
	clerr = clEnqueueNDRangeKernel(di.commands, di.kernel, 1, NULL, &global, &local, 0, NULL, NULL); CLERR
	clFinish(di.commands);

	clReleaseMemObject(in2);
        clReleaseProgram(di.program);

	/*Set RNG kernel and its arguments*/
	di.kernel = clCreateKernel(di.program, KName2, &clerr);
	CLERR

	clerr |= clSetKernelArg(di.kernel, 0, sizeof(cl_mem), &in1); CLERR
	clerr |= clSetKernelArg(di.kernel, 1, sizeof(cl_mem), &in4); CLERR
	clerr |= clSetKernelArg(di.kernel, 2, sizeof(cl_mem), &out1); CLERR
	clerr |= clSetKernelArg(di.kernel, 3, sizeof(cl_mem), &in3); CLERR

	/* Call kernel */
	global = di.nWI; local = di.nWG;
	clerr = clEnqueueNDRangeKernel(di.commands, di.kernel, 1, NULL, &global, &local, 0, NULL, NULL); CLERR

	/* Read ouput buffer */
	clerr = clEnqueueReadBuffer( di.commands, out1, CL_TRUE, 0, sizeof(unsigned int) * alength*di.nWI, output, 0, NULL, NULL );
	CLERR

	for(i=0;i<di.nWI;i++) {
		printf("\n\n\tStream %d\n------------------------\n",i+1);

		sprintf(foutput,"data/DATA_%d.bin",i+1);
		fp=fopen(foutput,"w");

		fprintf(fp,"type: d\ncount: %d\nnumbit: 32\n",alength);
		for(j=i*alength;j<(i+1)*alength;j++) {
			fprintf(fp,"%u\n",output[j]);
		}
		printf("Output is redirected on %s\n",foutput);
		printf("\n");
		fclose(fp);
	}

	/* free memory objects */
	clReleaseMemObject(in1);
	clReleaseMemObject(in3);
	clReleaseMemObject(in4);
	clReleaseMemObject(out1);
	clReleaseKernel(di.kernel);
	clReleaseProgram(di.program);

	/*Close connection with devices*/
	clReleaseCommandQueue(di.commands);
	clReleaseContext(di.context);

	return 0;
}

/****************************************************/

void InitSystemvars( int argc, char **argv, dev_info *di, int *length){

	size_t size;

	if(argc<2){
		printf("	Usage: ./rnggpu   nWI (#samples)\n");
		exit(EXIT_FAILURE);
	}

	di->nWI=atoi(argv[1]);
	if(0 < di->nWI) {
		clGetDeviceInfo( di->device_id ,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&size,NULL);
		if(di->nWI<size) 
			di->nWG = di->nWI;
		else {
			if(0==di->nWI%size)
				di->nWG = size;
			else {
				printf("The #threads has to be a multiple of %d\n",size);
				exit(EXIT_FAILURE);
			}	
		}	
	}
	else {
		printf("Wrong number of threads\n");
		exit(EXIT_FAILURE);
	}

	if(3 == argc)

		*length = atoi(argv[2]);
	else
		*length = 50;

}

/****************************************************/

void readMTdata( mt_struct_naked *mts, int Nmts){

	FILE *fp;	
	int fileN, i;
	mt_struct_naked lmts; 
	
	fp = fopen(MT_DATA_FILE,"rb");
	if(NULL==fp){
		printf("%s   MT data file not found\n",MT_DATA_FILE);
		exit(EXIT_FAILURE);
	}

	/*Read number of mt_struct_naked stored in file*/
	fread(&fileN,sizeof(int),1,fp);
	
	/*If not enough data, exit*/
	if(fileN<Nmts){
		printf("Number of data in file is less than # of threds. Run mtParams again.\n");
		exit(EXIT_FAILURE);
	}

	/*Read Nmts structs from file*/
	for(i=0;i<Nmts;i++){
		fread(mts+i,sizeof(mt_struct_naked),1,fp);
	}
	
	fclose(fp);

}

/****************************************************/

char * load_program_source(const char *filename) {

	struct stat statbuf;
	FILE *fh;
	char *source;
	fh = fopen(filename, "r");
	if (fh == 0)
		return 0;

	stat(filename, &statbuf);
	source = (char *) malloc(statbuf.st_size + 1);
	fread(source, statbuf.st_size, 1, fh);
	source[statbuf.st_size] = '\0';

	return source;

}
