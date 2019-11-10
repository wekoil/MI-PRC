#include <iostream>
#include "graph.h"
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cmath>
#include "constants.h"
#include <omp.h>
#include <fstream>


graph loadFile()
{
    std::ifstream input(INPUT);
    int k;
    input >> k;
    graph newGraph = graph();

    for (int i = 0; i < k; i++)
    {
        double * c = new double [DIMENSIONS];
        for (int k = 0; k < DIMENSIONS; k++)
            input >> c[k];
        newGraph.loadPoints(c);
    }
    return newGraph;
}

graph randomInput()
{
    srand(time(NULL));
    int k = POINTS;
    graph newGraph = graph();

    std::ofstream myfile;
    myfile.open ("input.txt", std::ofstream::out | std::ofstream::trunc);
    myfile << POINTS << "\n";

    for (int i = 0; i < k; i++)
    {
        double * c = new double[DIMENSIONS];

        for (int i = 0; i < DIMENSIONS; i++)
        {
            c[i] = std::rand() % 20000 - 10000;

            myfile << c[i] << " ";
        }

        myfile << "\n";

        newGraph.loadPoints(c);
    }

    myfile.close();
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

#pragma omp parallel for schedule(static)
    for (int i = 0; i < points->size(); i++)
    {
        if (points->at(i).clusterID != -1)
            continue;
        double min = getDistanceBetweenPoints(points->at(i), points->at(g.getCenterPointOfCluster(0)));
        int minClusterID = 0;


        for (int center = 0; center < CLUSTERS; center++)
        {
            double actual = getDistanceBetweenPoints(points->at(i), points->at(g.getCenterPointOfCluster(center)));
            if (actual < min)
            {

                if (actual < min)
                {
                    min = actual;
                    minClusterID = center;
                }

            }
        }

        points->at(i).clusterID = minClusterID;

    }

    for (int i = 0; i < points->size(); i++)
        g.getClusters()->at(points->at(i).clusterID).addPoint(i);
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
    g.printToFile();
}

int main()   {

    //graph g = loadFile();
    clock_t begin = clock();

    graph g = randomInput();
    kmeans(g);

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << elapsed_secs << std::endl;
    return 0;
}