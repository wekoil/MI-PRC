//
// Created by majkl on 13.10.19.
//

#ifndef UNTITLED_POINT_H
#define UNTITLED_POINT_H

#include "constants.h"

struct point
{
    int dimensions;
    float coords[DIMENSIONS];
    int id;
    int clusterID;

    point(int id, float *& a)
    {
        this->id = id;
        clusterID = -1;
        this->dimensions = DIMENSIONS;
        for (int i = 0; i < DIMENSIONS; i++)
            this->coords[i] = a[i];
    }

    point(float * a)
    {
        this->dimensions = DIMENSIONS;
        for (int i = 0; i < DIMENSIONS; i++)
            this->coords[i] = a[i];
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

    void set(float *& a)
    {
        for (int i = 0; i < DIMENSIONS; i++)
            this->coords[i] = a[i];
    }
};


#endif //UNTITLED_POINT_H
