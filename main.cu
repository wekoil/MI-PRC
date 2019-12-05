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
        float * c = new float [DIMENSIONS];
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
        float * c = new float[DIMENSIONS];

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
        centroids[i] = point(g.getPoints()->at(cl[i]).get());
    }

}

float getDistanceBetweenPoints(point & from, point & to)
{
    float sum = 0;
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
        float min = getDistanceBetweenPoints(points->at(i), points->at(g.getCenterPointOfCluster(0)));
        int minClusterID = 0;


        for (int center = 0; center < CLUSTERS; center++)
        {
            float actual = getDistanceBetweenPoints(points->at(i), points->at(g.getCenterPointOfCluster(center)));
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
    float sum = 0;
    for (int i = 0; i < DIMENSIONS; i++)
        sum += pow((from.coords[i] - to.coords[i]),2);
    return sqrt(sum);
}

__global__ void computeGPU(point * points, point * centroids, float * sum, int * count, int *changes)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

	// distribute into points to clusters
    // #pragma unroll 4
	for (int i = index; i < POINTS; i += stride)
    {
        float min = getDistanceBetweenPointsCUDA(points[i], centroids[0]);
        int minClusterID = 0;

        // printf("%lf %lf\t", points[i].coordx, points[i].coordy);

        // printf("%lf ", min);


        for (int center = 1; center < CLUSTERS; center++)
        {
            float actual = getDistanceBetweenPointsCUDA(points[i], centroids[center]);
            // printf("%lf ", actual);
            if (actual < min)
            {
                min = actual;
                minClusterID = center;
            }
            
        }

        // printf("\t%d", minClusterID);

        // printf("\n");

        if (points[i].clusterID != minClusterID)
        {
            atomicAdd(&changes[0], 1);
            points[i].clusterID = minClusterID;
        }
            

        for (int dim = 0; dim < DIMENSIONS; dim++)
            atomicAdd(&sum[minClusterID * DIMENSIONS + dim], points[i].coords[dim]);
        atomicAdd(&count[minClusterID], 1);

    }

} 

__global__ void computeNewCentroidsGPU(point * points, point * centroids, float * sum, int * count)
{
    int center = threadIdx.x;
    // int stride = blockDim.x * gridDim.x;

    // distribute into points to clusters
    // #pragma unroll 4
    for (int i = 0; i < POINTS; i++)
    {


        if (center == points[i].clusterID)
        {
            for (int dim = 0; dim < DIMENSIONS; dim++)
                sum[center * DIMENSIONS + dim] += points[i].coords[dim];
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
    clock_t beginInit = clock();

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

    float * cudaSum;
    int * cudaCount;

    int changes[1];
    changes[0] = 0;
    int* cudaChanges;

    clock_t totalInit = clock() - beginInit;

    clock_t beginMalloc = clock();

    HANDLE_ERROR(cudaMalloc(&cudaPoints, POINTS * sizeof(struct point)));
    HANDLE_ERROR(cudaMalloc(&cudaCentroids, CLUSTERS * sizeof(struct point)));
    HANDLE_ERROR(cudaMalloc(&cudaSum, DIMENSIONS * CLUSTERS * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&cudaCount, CLUSTERS * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&cudaChanges, 1 * sizeof(int)));

    clock_t totalMalloc = clock() - beginMalloc;

    clock_t beginCopy = clock();

    
    HANDLE_ERROR(cudaMemcpy(cudaPoints, points, POINTS * sizeof(point), cudaMemcpyHostToDevice));

    clock_t totalCopy = clock() - beginCopy;

    clock_t totalCudaCompute = 0;
    
    clock_t totalCentroids = 0;

    int iter = 0;
    do
    {
        beginInit = clock();

        float sum[CLUSTERS * DIMENSIONS];
        int count[CLUSTERS];

        for (int i = 0; i < CLUSTERS; i++)
        {
            for (int dim = 0; dim < DIMENSIONS; dim++)
                sum[i * DIMENSIONS + dim] = 0.0;
            count[i] = 0;
        }

        

        totalInit += clock() - beginInit;

        beginCopy = clock();

        HANDLE_ERROR(cudaMemcpy(cudaCentroids, centroids, CLUSTERS * sizeof(point), cudaMemcpyHostToDevice));
        

        HANDLE_ERROR(cudaMemcpy(cudaSum, sum,  DIMENSIONS * CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(cudaCount, count, CLUSTERS * sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(cudaChanges, changes, sizeof(int), cudaMemcpyHostToDevice));

        totalCopy += clock() - beginCopy;

        clock_t cudeCompute = clock();

        int blockSizeX = 128;//4*gsizex;
        int numBlocksX = (POINTS + blockSizeX - 1) / blockSizeX;

        computeGPU<<<numBlocksX, blockSizeX>>>((point*)cudaPoints, (point*)cudaCentroids, cudaSum, cudaCount, cudaChanges);
        // cudaDeviceSynchronize();


        // computeNewCentroidsGPU<<<1, CLUSTERS>>>((point*)cudaPoints, (point*)cudaCentroids, cudaSum, cudaCount);

        cudaDeviceSynchronize();

        totalCudaCompute += clock() - cudeCompute;

        // HANDLE_ERROR(cudaMemcpy(points, cudaPoints, POINTS * sizeof(point), cudaMemcpyDeviceToHost));

        // for (int i = 0; i < POINTS; i++)
        //     g.getPoints()->at(i) = points[i];
        // g.printToFile();

        beginCopy = clock();

        HANDLE_ERROR(cudaMemcpy(sum, cudaSum, DIMENSIONS * CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(count, cudaCount, CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(changes, cudaChanges, sizeof(int), cudaMemcpyDeviceToHost));

        totalCopy += clock() - beginCopy;

        std::cout << "changes: " << changes[0] << " points: " << POINTS << " treshold: " << ITER_TRESHOLD << std::endl;

        if (float(changes[0]) / float(POINTS) < ITER_TRESHOLD)
        {
            std::cout << "iter: " << iter << std::endl;
            break;
        }

        changes[0] = 0;

        // std::cout << sumX[0] << std::endl;

        // HANDLE_ERROR(cudaMemcpy(centroids, cudaCentroids, CLUSTERS * sizeof(point), cudaMemcpyDeviceToHost));

        clock_t beginCentroids = clock();

        bool flag = true;
        for (int i = 0; i < CLUSTERS; i++)
        {
            for (int dim = 0; dim < DIMENSIONS; dim++)
            {
                if (flag && centroids[i].get(dim) != sum[i * DIMENSIONS + dim]/count[i])
                    flag = false;
                centroids[i].set(dim, sum[i * DIMENSIONS + dim]/count[i]);
            }
            // std::cout << sumX[i] << "\t" << sumY[i] << "\t" << count[i] << std::endl;
            // std::cout << centroids[i].coordx << "\t" << centroids[i].coordy << std::endl;
        }
        // std::cout << "\n\n";

        totalCentroids += clock() - beginCentroids;
        
        // if (flag)
        // {
        //     std::cout << "iter: " << iter << std::endl;
        //     break;
        // }



    } while (iter++ < ITERATIONS);

    // cudaKmeans(g);

    std::cout << "cuda init: " << float(totalInit) / CLOCKS_PER_SEC << std::endl;
    std::cout << "cuda malloc: " << float(totalMalloc) / CLOCKS_PER_SEC << std::endl;
    std::cout << "cuda copy: " << float(totalCopy) / CLOCKS_PER_SEC << std::endl;
    std::cout << "cuda computing: " << float(totalCudaCompute) / CLOCKS_PER_SEC << std::endl;
    std::cout << "cpu centroids conmputing: " << float(totalCentroids) / CLOCKS_PER_SEC << std::endl;

    
    free(points);

    cudaFree(cudaPoints);
    cudaFree(centroids);
    cudaFree(cudaSum);
    cudaFree(cudaCount);
    cudaFree(cudaChanges);
}

int main()   {

    //graph g = loadFile();

    graph g = randomInput();

    clock_t begin = clock();
    cudaKmeans(g);
    clock_t end = clock();

    float elapsed_secs = float(end - begin) / CLOCKS_PER_SEC;
    std::cout << "cuda total: " << elapsed_secs << std::endl;

    begin = clock();
    kmeans(g);
    end = clock();

    elapsed_secs = float(end - begin) / CLOCKS_PER_SEC;
    std::cout << "cpu: " << elapsed_secs << std::endl;


    return 0;
}
