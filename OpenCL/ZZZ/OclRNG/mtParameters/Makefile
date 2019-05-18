#	Parallel Random Number Generator in OpenCL
#	Copyright (C) 2011 Giorgos Arampatzis, Angelos Athanasopoulos
#	
#	This program is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.

#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.

#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <http://www.gnu.org/licenses/>.

CC=/usr/bin/gcc

LIBS= -L../lib -ldcmt
INCLUDE=-I ../include

DFLAGS = -g
CFLAGS = -c $(DFLAGS) $(INCLUDE)
LFLAGS = $(DFLAGS) -o  

COMPILE = $(CC) $(CFLAGS)
LINK    = $(CC) $(LFLAGS)

RM = rm -f


all: mtParams


mtParams.o: mtParams.c
	$(COMPILE) mtParams.c

mtParams: mtParams.o
	$(LINK)   mtParams    mtParams.o $(LIBS)
	./mtParams 30

clean:
	$(RM) *.o
	$(RM) mtParams
