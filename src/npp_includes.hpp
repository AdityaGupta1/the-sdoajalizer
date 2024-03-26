#pragma once

#include <npp.h>
#include <nppi.h>

#include <iostream>
#define NPP_CHECK(S) { NppStatus eStatusNPP; \
        eStatusNPP = S; \
        if (eStatusNPP != NPP_SUCCESS) printf("brokey\n"); }
