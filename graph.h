//
// Created by majkl on 13.10.19.
//


#ifndef UNTITLED_GRAPH_H
#define UNTITLED_GRAPH_H

#include <vector>
#include "point.h"
#include "cluster.h"
#include <iostream>

class graph
{
private:
    std::vector<point> points;
    std::vector<cluster> clusters, oldClusters;
    int counter;
public:
    void loadPoints(int x, int y) {

        this->points.push_back(point(counter++, x, y));
    }

    graph() {
        this->counter = 0;
    }

    std::vector<point> *getPoints() {
        return &this->points;
    }

    void initCluster(int id, int pointID) {
        clusters.push_back(cluster(id, pointID));
        points[pointID].clusterID = id;
    }

    int getCenterPointOfCluster(int clusterID)
    {
        return clusters.at(clusterID).getCenterPoint();
    }

    std::vector<cluster> * getClusters()
    {
        return &this->clusters;
    }

    void print()
    {
        for (int i = 0; i < clusters.size(); i++)
        {
            std::cout << "Cluster: " << i << std::endl;
            std::vector<int>* pointIDs = clusters.at(i).getPoints();
            for (int k = 0; k < pointIDs->size(); k++)
                std::cout << points.at(pointIDs->at(k)).getX() << " " << points.at(pointIDs->at(k)).getY() << std::endl;
        }
        std::cout << std::endl;
    }

    void makeNewCentroids()
    {
        oldClusters = clusters;
        for (int i = 0; i < clusters.size(); i++)
            clusters.at(i).makeNewCentroid(&this->points);
    }

    bool isSameAsOld()
    {
        for (int i = 0; i < clusters.size(); i++)
        {
            if (*clusters.at(i).getPoints() != *oldClusters.at(i).getPoints())
                return false;
        }
        return true;
    }
};


#endif //UNTITLED_GRAPH_H
