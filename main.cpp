#include <iostream>
#include "graph.h"
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cmath>
#include "constants.h"
#include <omp.h>


graph loadFile()
{
    int k;
    std::cin >> k;
    graph newGraph = graph();

    for (int i = 0; i < k; i++)
    {
        double * c = new double [DIMENSIONS];
        for (int k = 0; k < DIMENSIONS; k++)
        std::cin >> c[k];
        newGraph.loadPoints(c);
    }
    return newGraph;
}

graph randomInput()
{
    int k = 10000;
    graph newGraph = graph();

    for (int i = 0; i < k; i++)
    {
        double * c = new double[DIMENSIONS];

        for (int i = 0; i < DIMENSIONS; i++)
            c[i] = std::rand() % 200 - 100;

        newGraph.loadPoints(c);
    }
    return newGraph;
}

void initClusters(graph & g)
{
    // find different numbers for clusters
    srand(time(NULL));

    int a, b, c;
    a = std::rand() % g.getPoints()->size();
    b = std::rand() % g.getPoints()->size();
    while (a == b)
        b = std::rand() % g.getPoints()->size();
    c = std::rand() % g.getPoints()->size();
    while (a == c || b == c)
        c = std::rand() % g.getPoints()->size();


    int * cl = new int[CLUSTERS];
    cl[0] = std::rand() % g.getPoints()->size();
    for (int i = 1; i < CLUSTERS; i++)
    {
        cl[i] = std::rand() % g.getPoints()->size();
        for (int k = i - 1; k >= 0; k--)
            if (cl[i] == cl[k])
            {
                i--;
                break;
            }
    }

    for (unsigned int i = 0; i < CLUSTERS; i++)
        g.initCluster(i,cl[i]);

}

double getDistanceBetweenPoints(point & from, point & to)
{
    double sum = 0;
    for (int i = 0; i < DIMENSIONS; i++)
        sum += pow((from.get(i) - to.get(i)),2);
    return sqrt(sum);
}

void distributePointsIntoClusters(graph & g)
{
    std::vector<point> * points = g.getPoints();
    for (int i = 0; i < points->size(); i++)
    {
        if (points->at(i).clusterID != -1)
            continue;
        double min = getDistanceBetweenPoints(points->at(i), points->at(g.getCenterPointOfCluster(0)));
        int minClusterID = 0;

#pragma omp parallel for
        for (int center = 0; center < 3; center++)
        {
            double actual = getDistanceBetweenPoints(points->at(i), points->at(g.getCenterPointOfCluster(center)));
            if (actual < min)
            {
#pragma omp critical
                if (actual < min)
                {
                    min = actual;
                    minClusterID = center;
                }
            }
        }

        points->at(i).clusterID = minClusterID;
        g.getClusters()->at(minClusterID).addPoint(i);
    }
}

void kmeans(graph & g)
{
    initClusters(g);
    distributePointsIntoClusters(g);

    for (int i = 0; i < ITERATIONS; i++)
    {
        g.makeNewCentroids();
        distributePointsIntoClusters(g);

        if (g.isSameAsOld())
        {
            std::cout << "iter: " << i << std::endl;
            break;
        }

    }
    g.print();
}

int main() {

//    graph g = loadFile();
    graph g = randomInput();
    kmeans(g);
    return 0;
}