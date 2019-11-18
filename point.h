//
// Created by majkl on 13.10.19.
//

#ifndef UNTITLED_POINT_H
#define UNTITLED_POINT_H

#include "constants.h"

struct point
{
    int dimensions;
    double coordx, coordy;
    int id;
    int clusterID;

    point(int id, double *& coords) {
        this->id = id;
        clusterID = -1;
        this->dimensions = DIMENSIONS;
        coordx = coords[0];
        coordy = coords[1];

    }

    point(double x,double y) {
        coordx = x;
        coordy = y;

    }

    double get(int x)
    {
        if (x==0)
            return coordx;
        return coordy;
    }

    void set(int x, double a)
    {
        if (x==0)
            coordx = a;
        else
            coordy = a;
    }

    void set(double a, double b)
    {
        coordx = a;
        coordy = b;
    }
};


#endif //UNTITLED_POINT_H
