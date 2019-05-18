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

DIRS	= mtParameters/ lib/ example/

all: Lib mtParams rnggpu

Lib:
	$(MAKE) -C lib/

mtParams: 
	$(MAKE) -C mtParameters/

rnggpu:
	$(MAKE) -C example/

clean:
	-for d in $(DIRS); do (cd $$d; $(MAKE) clean ); done
