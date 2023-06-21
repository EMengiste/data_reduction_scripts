#!/bin/bash
#
rad=0.01

neper -V "final(type=ori):file(fin),initial(type=ori):file(ini),pathplast(type=ori):file(all)" \
        -space ipf                                     \
        -datainitialrad 0.02  -datainitialcol white    \
        -datapathplastrad 0.01 -datapathplastsymbol "square" -datapathplastcol    black   \
        -datafinalrad   0.025 -datafinalcol   grey    \
        -print ipf
exit 0