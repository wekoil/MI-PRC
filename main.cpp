#include <iostream>
#include "graph.h"
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cmath>
#include "constants.h"
#include <omp.h>
#include <fstream>
#include <cuda_runtime_api.h>
#include <cuda.h>

static void HandleError( cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
      exit( EXIT_FAILURE );
  }
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

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

void initClusters(graph & g, point *& centroids)
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
    {
        g.initCluster(i,cl[i]);
        centroids[i] = point(g.getPoints()->at(cl[i]).coordx, g.getPoints()->at(cl[i]).coordy);
    }

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

__device__ float getDistanceBetweenPointsCUDA(point & from, point & to)
{
    double sum = 0;
    for (int i = 0; i < DIMENSIONS; i++)
        sum += pow((from.get(i) - to.get(i)),2);
    return sqrt(sum);
}

__global__ void computeGPU(point * points, point * centroids)
{

	// distribute into points to clusters
	double sumX[CLUSTERS], sumY[CLUSTERS];
    int count[CLUSTERS];

    for (int i = 0; i < CLUSTERS; i++)
    {
    	sumX[i] = 0.0;
    	sumY[i] = 0.0;
    	count[i] = 0;
    }

	for (int i = 0; i < POINTS; i++)
    {
        if (points[i].clusterID != -1)
        {
        	sumX[points[i].clusterID] += points[i].get(0);
        	sumY[points[i].clusterID] += points[i].get(1);
        	count[points[i].clusterID]++;
            continue;
        }
        double min = getDistanceBetweenPointsCUDA(points[i], centroids[0]);
        int minClusterID = 0;


        for (int center = 0; center < CLUSTERS; center++)
        {
            double actual = getDistanceBetweenPointsCUDA(points[i], centroids[i]);
            if (actual < min)
            {
                min = actual;
                minClusterID = center;
            }
        }
        points[i].clusterID = minClusterID;

        sumX[points[i].clusterID] += points[i].get(0);
    	sumY[points[i].clusterID] += points[i].get(1);
    	count[points[i].clusterID]++;
    }

    // compute new centroids
    bool flag = true;
    for (int i = 0; i < CLUSTERS; i++)
    {
    	if (centroids[i].get(0) != sumX[i]/count[i] || centroids[i].get(1) != sumY[i]/count[i])
    		flag = false;
    	centroids[i].set(sumX[i]/count[i], sumY[i]/count[i]);
    }
    
    if (flag)
    	return;

    for (int i = 0; i < POINTS; i++)
    	points[i].clusterID = -1;

} 

// void kmeans(graph & g)
// {
//     initClusters(g);
//     distributePointsIntoClusters(g);

//     for (int i = 0; i < ITERATIONS; i++)
//     {
//         g.makeNewCentroids();
//         distributePointsIntoClusters(g);

//         if (g.isSameAsOld())
//         {
//             std::cout << "iter: " << i << std::endl;
//             break;
//         }

//     }
//     g.printToFile();
// }

void kmeans(graph & g)
{
    

    void * cudaPoints, *cudaCentroids;

    point * points = (point *)malloc(POINTS * sizeof(point));
    point * centroids = (point *)malloc(CLUSTERS * sizeof(point));

    initClusters(g, centroids);

    for (int i = 0; i < POINTS; i++)
    	points[i] = g.getPoints()->at(i);

    for (int i = 0; i < CLUSTERS; i++)
    	centroids[i] = g.getPoints()->at(i);

    HANDLE_ERROR(cudaMalloc(&cudaPoints, POINTS * sizeof(point)));
    HANDLE_ERROR(cudaMalloc(&cudaCentroids, CLUSTERS * sizeof(point)));

    HANDLE_ERROR(cudaMemcpy(cudaPoints, points, POINTS * sizeof(point), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(cudaCentroids, centroids, CLUSTERS * sizeof(point), cudaMemcpyHostToDevice));


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
    

    HANDLE_ERROR(cudaMemcpy(points, cudaPoints, POINTS * sizeof(point), cudaMemcpyDeviceToHost));

    g.printToFile();

    free(points);

    cudaFree(cudaPoints);
    cudaFree(centroids);
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