#include <iostream>
#include "graph.h"
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cmath>

const int CLUSTERS = 3;
const int ITERATIONS = 100;

graph loadFile()
{
    int k;
    std::cin >> k;
    graph newGraph = graph();

    for (int i = 0; i < k; i++)
    {
        int x, y;
        std::cin >> x >> y;
        newGraph.loadPoints(x, y);
    }
    return newGraph;
}

graph randomInput()
{
    int k = 1000;
    graph newGraph = graph();

    for (int i = 0; i < k; i++)
    {
        int x, y;
        x = std::rand() % 200;
        y = std::rand() % 200;
        newGraph.loadPoints(x-100, y-100);
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

    g.initCluster(1,a);
    g.initCluster(2,b);
    g.initCluster(3,c);
}

double getDistanceBetweenPoints(point & from, point & to)
{
    return sqrt(pow((from.getX() - to.getX()),2) + pow((from.getY() - to.getY()), 2));
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

        for (int center = 0; center < 3; center++)
        {
            double actual = getDistanceBetweenPoints(points->at(i), points->at(g.getCenterPointOfCluster(center)));
            if (actual < min)
            {
                min = actual;
                minClusterID = center;
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

    graph g = loadFile();
//    graph g = randomInput();
    kmeans(g);
    return 0;
}