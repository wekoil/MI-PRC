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
#include <cstdio>

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
            c[i] = std::rand() % (2*GRIDSIZE) - GRIDSIZE;

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

// #pragma omp parallel for schedule(static)
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

__device__ double getDistanceBetweenPointsCUDA(point & from, point & to)
{
    double sum = 0;
    // for (int i = 0; i < DIMENSIONS; i++)
        sum += pow((from.coordx - to.coordx),2);
        sum += pow((from.coordy - to.coordy),2);
    return sqrt(sum);
}

__global__ void computeGPU(point * points, point * centroids, double * sumX, double * sumY, int * count)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

	// distribute into points to clusters
    // #pragma unroll 4
	for (int i = index; i < POINTS; i += stride)
    {
        double min = getDistanceBetweenPointsCUDA(points[i], centroids[0]);
        int minClusterID = 0;

        // printf("%lf %lf\t", points[i].coordx, points[i].coordy);

        // printf("%lf ", min);


        for (int center = 1; center < CLUSTERS; center++)
        {
            double actual = getDistanceBetweenPointsCUDA(points[i], centroids[center]);
            // printf("%lf ", actual);
            if (actual < min)
            {
                min = actual;
                minClusterID = center;
            }
            
        }

        // printf("\t%d", minClusterID);

        // printf("\n");

        points[i].clusterID = minClusterID;

    }

} 

__global__ void computeNewCentroidsGPU(point * points, point * centroids, double * sumX, double * sumY, int * count)
{
    int center = threadIdx.x;
    // int stride = blockDim.x * gridDim.x;

    // distribute into points to clusters
    #pragma unroll 4
    for (int i = 0; i < POINTS; i++)
    {


        if (center == points[i].clusterID)
        {
            sumX[center] += points[i].coordx;
            sumY[center] += points[i].coordy;
            count[center]++;
        }

        // if (i == POINTS - 1)
        // {
            
        // }

        
    }

} 

void kmeans(graph & g)
{
    point * centroids = (point *)malloc(CLUSTERS * sizeof(point));
    initClusters(g, centroids);
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
        // g.printToFile();
    }
    
}

void cudaKmeans(graph & g)
{
    

    point * cudaPoints, *cudaCentroids;

    point * points = (point *)malloc(POINTS * sizeof(point));
    point * centroids = (point *)malloc(CLUSTERS * sizeof(point));

    initClusters(g, centroids);
    g.wipeFile();

    for (int i = 0; i < POINTS; i++)
    	points[i] = g.getPoints()->at(i);

    // for (int i = 0; i < CLUSTERS; i++)
    // {
    // 	// centroids[i] = g.getPoints()->at(g.getCenterPointOfCluster(i));
    //     std::cout << centroids[i].coordx << "\t" << centroids[i].coordy << std::endl;
    // }

    HANDLE_ERROR(cudaMalloc(&cudaPoints, POINTS * sizeof(struct point)));
    HANDLE_ERROR(cudaMalloc(&cudaCentroids, CLUSTERS * sizeof(struct point)));

    
    HANDLE_ERROR(cudaMemcpy(cudaPoints, points, POINTS * sizeof(point), cudaMemcpyHostToDevice));

    double * cudaSumX, *cudaSumY;
    int * cudaCount;

    HANDLE_ERROR(cudaMalloc(&cudaSumX, CLUSTERS * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&cudaSumY, CLUSTERS * sizeof(double)));
    HANDLE_ERROR(cudaMalloc(&cudaCount, CLUSTERS * sizeof(int)));

    int iter = 0;
    do
    {
        double sumX[CLUSTERS], sumY[CLUSTERS];
        int count[CLUSTERS];

        for (int i = 0; i < CLUSTERS; i++)
        {
            sumX[i] = 0.0;
            sumY[i] = 0.0;
            count[i] = 0;
        }
        HANDLE_ERROR(cudaMemcpy(cudaCentroids, centroids, CLUSTERS * sizeof(point), cudaMemcpyHostToDevice));
        

        HANDLE_ERROR(cudaMemcpy(cudaSumX, sumX,  CLUSTERS * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(cudaSumY, sumY,  CLUSTERS * sizeof(double), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(cudaCount, count, CLUSTERS * sizeof(int), cudaMemcpyHostToDevice));

        int blockSizeX = 128;//4*gsizex;
        int numBlocksX = (POINTS + blockSizeX - 1) / blockSizeX;

        computeGPU<<<numBlocksX, blockSizeX>>>((point*)cudaPoints, (point*)cudaCentroids, cudaSumX, cudaSumY, cudaCount);
        cudaDeviceSynchronize();


        computeNewCentroidsGPU<<<1, CLUSTERS>>>((point*)cudaPoints, (point*)cudaCentroids, cudaSumX, cudaSumY, cudaCount);

        cudaDeviceSynchronize();

        // HANDLE_ERROR(cudaMemcpy(points, cudaPoints, POINTS * sizeof(point), cudaMemcpyDeviceToHost));

        // for (int i = 0; i < POINTS; i++)
        //     g.getPoints()->at(i) = points[i];
        // g.printToFile();

        HANDLE_ERROR(cudaMemcpy(sumX, cudaSumX, CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(sumY, cudaSumY, CLUSTERS * sizeof(double), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(count, cudaCount, CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost));

        // std::cout << sumX[0] << std::endl;

        // HANDLE_ERROR(cudaMemcpy(centroids, cudaCentroids, CLUSTERS * sizeof(point), cudaMemcpyDeviceToHost));

        bool flag = true;
        for (int i = 0; i < CLUSTERS; i++)
        {
            if (centroids[i].coordx != sumX[i]/count[i] || centroids[i].coordy != sumY[i]/count[i])
                flag = false;
            centroids[i].coordx = sumX[i]/count[i];
            centroids[i].coordy = sumY[i]/count[i];
            // std::cout << sumX[i] << "\t" << sumY[i] << "\t" << count[i] << std::endl;
            // std::cout << centroids[i].coordx << "\t" << centroids[i].coordy << std::endl;
        }
        // std::cout << "\n\n";
        
        if (flag)
        {
            std::cout << "iter: " << iter << std::endl;
            break;
        }



    } while (iter++ < ITERATIONS);

    
    free(points);

    cudaFree(cudaPoints);
    cudaFree(centroids);
    cudaFree(cudaSumX);
    cudaFree(cudaSumY);
    cudaFree(cudaCount);
}

int main()   {

    //graph g = loadFile();

    graph g = randomInput();

    clock_t begin = clock();
    cudaKmeans(g);
    clock_t end = clock();

    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "cuda: " << elapsed_secs << std::endl;

    begin = clock();
    kmeans(g);
    end = clock();

    elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    std::cout << "cpu: " << elapsed_secs << std::endl;


    return 0;
}
