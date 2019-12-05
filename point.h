//
// Created by majkl on 13.10.19.
//

#ifndef UNTITLED_POINT_H
#define UNTITLED_POINT_H

#include "constants.h"

struct point
{
    int dimensions;
    float coordx, coordy;
    int id;
    int clusterID;

    point(int id, float *& coords) {
        this->id = id;
        clusterID = -1;
        this->dimensions = DIMENSIONS;
        coordx = coords[0];
        coordy = coords[1];

    }

    point(float x,float y) {
        coordx = x;
        coordy = y;

    }

    float get(int x)
    {
        if (x==0)
            return coordx;
        return coordy;
    }

    void set(int x, float a)
    {
        if (x==0)
            coordx = a;
        else
            coordy = a;
    }

    void set(float a, float b)
    {
        coordx = a;
        coordy = b;
    }
};


#endif //UNTITLED_POINT_H
