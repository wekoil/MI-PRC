//
// Created by majkl on 13.10.19.
//

#ifndef UNTITLED_POINT_H
#define UNTITLED_POINT_H


class point
{
private:
    double x, y;
    int id;
public:
    int clusterID;
    point(int id, double x, double y) {
        this->id = id;
        this->x = x;
        this->y = y;
        clusterID = -1;
    }

    double getX()
    {
        return x;
    }

    double getY()
    {
        return y;
    }

    void setX(double a)
    {
        x = a;
    }

    void setY(double a)
    {
        y = a;
    }
};


#endif //UNTITLED_POINT_H
