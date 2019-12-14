//
// Created by majkl on 13.10.19.
//


#ifndef UNTITLED_GRAPH_H
#define UNTITLED_GRAPH_H

#include <vector>
#include "point.h"
#include "cluster.h"
#include <iostream>
#include "constants.h"
#include <fstream>
#include <cstring>
#include <string>

class graph
{
private:
    std::vector<point> points;
    std::vector<cluster> clusters, oldClusters;
    int counter;
public:
    void loadPoints(float *& coords) {
        point x;
        x.alocate();
        x.set(coords);
        this->points.push_back(x);
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
                for (int j = 0; j < DIMENSIONS; j++)
                    std::cout << points.at(pointIDs->at(k)).get(j) << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    void printToFile(int * ids, float * coords)
    {
        std::ofstream myfile;
        
        // myfile.open ("output.txt", std::ofstream::out | std::ofstream::trunc);
        myfile.open (OUTPUT, std::ios_base::app);

        for (int dim = 0; dim < DIMENSIONS; dim++)
        {
            for (int i = 0; i < points.size(); i++)
                myfile << coords[dim * DIMENSIONS + i] << ", ";
            myfile << std::endl;
        }

        for (int i = 0; i < points.size(); i++)
            myfile << ids[i] << ", ";
        myfile << std::endl;


        myfile.close();
    }

    void printToFile()
    {
        std::ofstream myfile;
        
        // myfile.open ("output.txt", std::ofstream::out | std::ofstream::trunc);
        myfile.open (OUTPUT, std::ios_base::app);

        for (int dim = 0; dim < DIMENSIONS; dim++)
        {
            for (int i = 0; i < points.size(); i++)
                myfile << points[i].get(dim) << ", ";
            myfile << std::endl;
        }

        for (int i = 0; i < points.size(); i++)
            myfile << points[i].clusterID << ", ";
        myfile << std::endl;


        myfile.close();
    }

    void wipeFile()
    {
        std::ofstream myfile;
        
        myfile.open (OUTPUT, std::ofstream::out | std::ofstream::trunc);


        myfile.close();
    }

    void makeNewCentroids()
    {
        oldClusters = clusters;
// #pragma omp parallel for
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
