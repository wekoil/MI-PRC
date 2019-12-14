//
// Created by majkl on 13.10.19.
//

#ifndef UNTITLED_POINT_H
#define UNTITLED_POINT_H

#include "constants.h"

struct point
{
    float * coords;
    int id;
    int clusterID;

    void alocate()
    {
        coords = new float[DIMENSIONS];
        clusterID = -1;
    }

    void dealocate()
    {
        delete [] coords;
    }

    float get(int x)
    {
        return coords[x];
    }

    float* get()
    {
        return coords;
    }

    void set(int x, float a)
    {
        coords[x] = a;
    }

    void set(float * a)
    {
        for (int i = 0; i < DIMENSIONS; i++)
            this->coords[i] = a[i];
    }
};


#endif //UNTITLED_POINT_H
