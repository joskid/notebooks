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

#define uint32_t unsigned int

typedef struct {
    uint32_t aaa;
    int mm,nn,rr,ww;
    uint32_t wmask,umask,lmask;
    int shift0, shift1, shiftB, shiftC;
    uint32_t maskB, maskC;
    int i;
}mt_struct_naked;

//*******************************************************************************
//****     Pseudo Random Number Generator       *********************************
//*******************************************************************************

typedef struct {
    uint32_t aaa;
    int mm,nn,rr,ww;
    uint32_t wmask,umask,lmask;
    int shift0, shift1, shiftB, shiftC;
    uint32_t maskB, maskC;
    int i;
    uint32_t state[MT_NN];
}mt_struct;

void sgenrand_mt(uint32_t seed, mt_struct *mts) 
{
    int i;

    for (i=0; i<mts->nn; i++) {
	mts->state[i] = seed;
        seed = ( (1812433253U) * (seed  ^ (seed >> 30))) + i + 1;
    }
    mts->i = mts->nn;

    for (i=0; i<mts->nn; i++) 
	mts->state[i] &= mts->wmask;
}

uint32_t genrand_mt(mt_struct *mts) 
{
    uint32_t *st, uuu, lll, aa, x;
    int k,n,m,lim;

    if ( mts->i >= mts->nn ) {
	n = mts->nn; m = mts->mm;
	aa = mts->aaa;
	st = mts->state;
	uuu = mts->umask; lll = mts->lmask;

	lim = n - m;
	for (k=0; k<lim; k++) {
	    x = (st[k]&uuu)|(st[k+1]&lll);
	    st[k] = st[k+m] ^ (x>>1) ^ (x&1U ? aa : 0U);
	}
	lim = n - 1;
	for (; k<lim; k++) {
	    x = (st[k]&uuu)|(st[k+1]&lll);
	    st[k] = st[k+m-n] ^ (x>>1) ^ (x&1U ? aa : 0U);
	}
	x = (st[n-1]&uuu)|(st[0]&lll);
	st[n-1] = st[m-1] ^ (x>>1) ^ (x&1U ? aa : 0U);
	mts->i=0;
    }
		
    x = mts->state[mts->i];
    mts->i += 1;
    x ^= x >> mts->shift0;
    x ^= (x << mts->shiftB) & mts->maskB;
    x ^= (x << mts->shiftC) & mts->maskC;
    x ^= x >> mts->shift1;

    return x;
}


float floatrand(mt_struct *mts){
	return ( (float)genrand_mt(mts) ) / ((float) 0xFFFFFFFF) ;
}



//*******************************************************************************
//***********       Functions  ******************************************
//*******************************************************************************



// Copy the data from an mt_struct_naked to an mt_struct
// The NG uses mt_struct structs.
void copyNakedToMts(mt_struct *mts, mt_struct_naked mts_naked){

	mts->aaa = mts_naked.aaa;
	mts->mm  = mts_naked.mm;
	mts->nn  = mts_naked.nn;
	mts->rr  = mts_naked.rr;
	mts->ww  = mts_naked.ww;
	mts->wmask  = mts_naked.wmask;
	mts->umask  = mts_naked.umask;
	mts->lmask  = mts_naked.lmask;
	mts->maskB  = mts_naked.maskB;
	mts->maskC  = mts_naked.maskC;
	mts->shift0  = mts_naked.shift0;
	mts->shift1  = mts_naked.shift1;
	mts->shiftB  = mts_naked.shiftB;
	mts->shiftC  = mts_naked.shiftC;
	mts->i  = mts_naked.i;

}

void copyNakedToMtsWithState(mt_struct *mts, mt_struct_naked mts_naked, __global uint32_t *state) {

        int i;
        int gid = get_global_id(0);

        mts->aaa = mts_naked.aaa;
        mts->mm  = mts_naked.mm;
        mts->nn  = mts_naked.nn;
        mts->rr  = mts_naked.rr;
        mts->ww  = mts_naked.ww;
        mts->wmask  = mts_naked.wmask;
        mts->umask  = mts_naked.umask;
        mts->lmask  = mts_naked.lmask;
        mts->maskB  = mts_naked.maskB;
        mts->maskC  = mts_naked.maskC;
        mts->shift0  = mts_naked.shift0;
        mts->shift1  = mts_naked.shift1;
        mts->shiftB  = mts_naked.shiftB;
        mts->shiftC  = mts_naked.shiftC;
        mts->i  = mts_naked.i;
        for(i=0;i<MT_NN;i++)
                mts->state[i] = state[MT_NN*gid+i];

}

void SaveState(__global uint32_t *state, __global mt_struct_naked *mts_naked, mt_struct *mts) {

	int i;
	int gid = get_global_id(0);
	for(i=0;i<MT_NN;i++)
                 state[MT_NN*gid+i] = mts->state[i];

        mts_naked[gid].i = mts->i;
}

//*******************************************************************************
//***********       Kernel Functions    *****************************************
//*******************************************************************************

__kernel
void initRNG( __global mt_struct_naked *mts_naked,
              __global uint32_t *state,
              __constant int *seeds
            ) {

        int i;
        int gid = get_global_id(0);

        mt_struct mts;

        copyNakedToMts(&mts,mts_naked[gid]);
        sgenrand_mt( seeds[gid], &mts );

	SaveState(state,mts_naked,&mts);

}


__kernel
void RNG(	__global mt_struct_naked *mts_naked,
		__global uint32_t *state,
		__global int *output,
		__global int *alength) {

	int gid = get_global_id(0);
	int i;
	mt_struct mts;

	copyNakedToMtsWithState(&mts,mts_naked[gid],state);
	for(i=0;i<*alength;i++) 
		output[gid*(*alength)+i] = genrand_mt(&mts);

	SaveState(state,mts_naked,&mts);

}
