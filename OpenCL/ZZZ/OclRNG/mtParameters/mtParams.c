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





/******************************************************************/
/**  mtParams.c  **************************************************/
/******************************************************************/
/**  Creates the Mersenne Twister parameters using the dcmt      **/
/**  library. The parameters seed, word length and the mersenne  **/
/**  exponent are defined as constants by the #define directive. **/
/**  The output is redirected to a .bin file in the mtDATA       **/
/**  directory that should be used in the OpenCL code.           **/
/******************************************************************/
/******************************************************************/

#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include"dc.h"

#define SEED 2011
#define WORD_LEN 32
#define MERS_EXP 521

#define DEBUG 0

/******************************************************************/

typedef struct {
    uint32_t aaa;
    int mm,nn,rr,ww;
    uint32_t wmask,umask,lmask;
    int shift0, shift1, shiftB, shiftC;
    uint32_t maskB, maskC;
    int i;
}mt_struct_naked;

/******************************************************************/
/**  Debug Functions  *********************************************/
/******************************************************************/

void printMT(mt_struct *, int );
void printMTnaked(mt_struct_naked *, int );
void printMTmat(mt_struct **, int );

/******************************************************************/

int main(int argc,  char **argv){

	FILE *fp;
	mt_struct **mtss;

	mt_struct_naked mts2;

	int i, N;
	
	char filename[100];
	
	if(argc<2){
		printf("	Usage: ./mtParams  numberOfThreads\n");
		return 0;
	}

	// # of MT parameters to be created
	N = atoi(argv[1]);
	sprintf(filename,"./mtDATA/mtDATA_%d_%d_%s.bin",MERS_EXP,WORD_LEN,argv[1]);
	puts(filename);

	// create N MT parameters
	printf("Creating %d parameters with seed %d\n",N,SEED);
	mtss = get_mt_parameters_st( WORD_LEN, MERS_EXP, 0, N-1, SEED, &N );


	// write data in file
	fp=fopen(filename,"wb");
	fwrite(&N,sizeof(int),1,fp);
	for(i=0;i<N;i++)
		fwrite(mtss[i],sizeof(mt_struct_naked),1,fp);
	fclose(fp);


#if DEBUG
	printMTmat(mtss,N);

	fp=fopen(filename,"rb");
	fread(&N,sizeof(int),1,fp);
	for(i=0;i<N;i++){
		fread(&mts2,sizeof(mt_struct_naked),1,fp);
		printMTnaked(&mts2,1);
	}

	fclose(fp);
#endif

return 0;
}

/******************************************************************/

void printMT(mt_struct *mtss, int count){
	int i,j;

	for(i=0; i<count; i++){
		printf("ID : %d\n",i);
	    printf("aaa = %"PRIx32"   ", mtss[i].aaa);
		printf("mm=%d    nn=%d       rr=%d      ww=%d   ",mtss[i].mm,mtss[i].nn,mtss[i].rr,mtss[i].ww);
		printf("s0=%d    s1=%d       sB=%d      sC=%d\n",mtss[i].shift0,mtss[i].shift1,mtss[i].shiftB,mtss[i].shiftC);
		printf("wm=%"PRIx32"    um=%"PRIX32"       lm=%"PRIx32"      mB=%"PRIx32"     mC=%"PRIx32"  ",
		mtss[i].wmask,mtss[i].umask,mtss[i].lmask, mtss[i].maskB, mtss[i].maskC);
		printf("i=%d\n",mtss[i].i);
		for(j=0;j<mtss[i].nn;j++)
			printf("%"PRIx32" \t",mtss[i].state[j]);
		printf("\n");
		printf("---------------------------------\n");
	}
}

/******************************************************************/

void printMTnaked(mt_struct_naked *mtss, int count){
	int i,j;

	for(i=0; i<count; i++){
		printf("ID : %d\n",i);
	    printf("aaa = %"PRIx32"   ", mtss[i].aaa);
		printf("mm=%d    nn=%d       rr=%d      ww=%d   ",mtss[i].mm,mtss[i].nn,mtss[i].rr,mtss[i].ww);
		printf("s0=%d    s1=%d       sB=%d      sC=%d\n",mtss[i].shift0,mtss[i].shift1,mtss[i].shiftB,mtss[i].shiftC);
		printf("wm=%"PRIx32"    um=%"PRIX32"       lm=%"PRIx32"      mB=%"PRIx32"     mC=%"PRIx32"  ",
		mtss[i].wmask,mtss[i].umask,mtss[i].lmask, mtss[i].maskB, mtss[i].maskC);
		printf("i=%d\n",mtss[i].i);
		printf("---------------------------------\n");
	}
}

/******************************************************************/

void printMTmat(mt_struct **mtss, int count){
	int i,j;

	for(i=0; i<count; i++){
		printf("ID : %d\n",i);
        printf("aaa = %"PRIx32"   ", mtss[i]->aaa);
		printf("mm=%d    nn=%d       rr=%d      ww=%d   ",mtss[i]->mm,mtss[i]->nn,mtss[i]->rr,mtss[i]->ww);
		printf("s0=%d    s1=%d       sB=%d      sC=%d\n",mtss[i]->shift0,mtss[i]->shift1,mtss[i]->shiftB,mtss[i]->shiftC);
		printf("wm=%"PRIx32"    um=%"PRIX32"       lm=%"PRIx32"      mB=%"PRIx32"     mC=%"PRIx32"  ",
		mtss[i]->wmask,mtss[i]->umask,mtss[i]->lmask, mtss[i]->maskB, mtss[i]->maskC);
		printf("i=%d\n",mtss[i]->i);
		for(j=0;j<mtss[i]->nn;j++)
			printf("%"PRIx32" \t",mtss[i]->state[j]);
		printf("\n");
		printf("---------------------------------\n");
	}
}
