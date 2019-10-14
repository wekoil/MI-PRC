//
// Created by majkl on 13.10.19.
//

#ifndef UNTITLED_CLUSTER_H
#define UNTITLED_CLUSTER_H
#include <vector>
#include "point.h"

class cluster
{
private:
    int id;
    std::vector<int> pointIDs;
    int centerPoint;
public:
    cluster(int id, int pointID)
    {
        this->id = id;
        this->pointIDs.push_back(pointID);
        centerPoint = pointID;
    }
    int getCenterPoint()
    {
        return centerPoint;
    }
    void addPoint(int p)
    {
        pointIDs.push_back(p);
    }
    std::vector<int>* getPoints()
    {
        return &this->pointIDs;
    }
    void makeNewCentroid(std::vector<point> *points)
    {
        double sumX = 0, sumY = 0;
        for (int i = 0; i < pointIDs.size(); i++)
        {
            sumX += points->at(pointIDs.at(i)).getX();
            sumY += points->at(pointIDs.at(i)).getY();
            if (pointIDs.at(i) != centerPoint)
                points->at(pointIDs.at(i)).clusterID = -1;
        }
        points->at(centerPoint).setX(sumX / pointIDs.size());
        points->at(centerPoint).setY(sumY / pointIDs.size());

        while (pointIDs.size() != 1)
            pointIDs.pop_back();
    }
};


#endif //UNTITLED_CLUSTER_H
