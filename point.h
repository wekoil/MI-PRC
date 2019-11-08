//
// Created by majkl on 13.10.19.
//

#ifndef UNTITLED_POINT_H
#define UNTITLED_POINT_H

#include "constants.h"

class point
{
private:
    int dimensions;
    double * coords;
    int id;
public:
    int clusterID;
    point(int id, double *& coords) {
        this->id = id;
        clusterID = -1;
        this->dimensions = DIMENSIONS;
        this->coords = new double[dimensions];
        for (int i = 0; i <dimensions; i++)
            this->coords[i] = coords[i];

    }

    double get(int x)
    {
        return coords[x];
    }

    void set(int x, double a)
    {
        coords[x] = a;
    }
};


#endif //UNTITLED_POINT_H
