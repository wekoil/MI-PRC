#include <iostream>
#include <cstdlib>
#include <ctime>
#include <climits>
#include <cmath>
#include "constants.h"
#include "point.h"
#include "cluster.h"
#include "graph.h"
#include <omp.h>
#include <fstream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdio>
#include <vector>

#include <cstring>
#include <string>

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
    // int k;
    input >> POINTS;
    graph newGraph = graph();

    for (int i = 0; i < POINTS; i++)
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

void initClusters(graph & g, float *& centroids)
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
        point x;
        x.alocate();
        x.set(g.getPoints()->at(cl[i]).get());
        for (unsigned int k = 0; k < DIMENSIONS; k++)
            centroids[i * DIMENSIONS + k ] = point(x).coords[k];
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
        // std::cout << i << std::endl;
        if (points->at(i).clusterID != -1)
            continue;
        float min = getDistanceBetweenPoints(points->at(i), points->at(g.getCenterPointOfCluster(0)));
        int minClusterID = 0;


        for (int center = 1; center < CLUSTERS; center++)
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

        // std::cout << points->at(i).clusterID << std::endl;

    }

    // std::cout << "venku\n";

    for (int i = 0; i < points->size(); i++)
    {
        // std::cout << points->at(i).clusterID << std::endl;
        g.getClusters()->at(points->at(i).clusterID).addPoint(i);
    }

    // std::cout << "ale ne uplne\n";
}

__device__ float getDistanceBetweenPointsCUDA(float * coords, float * centroids, int from, int to, int DIMENSIONS)
{
    float sum = 0;
    for (int i = 0; i < DIMENSIONS; i++)
        sum += pow((coords[from * DIMENSIONS + i] - centroids[to * DIMENSIONS + i]),2);
    return sqrt(sum);
}

__global__ void computeGPU(int POINTS, int CLUSTERS, int DIMENSIONS, int * ids, float * coords, float * centroids, float * sum, int * count, int *changes)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

	// distribute into points to clusters
    // #pragma unroll 4
	for (int i = index; i < POINTS; i += stride)
    {
        float min = getDistanceBetweenPointsCUDA(coords, centroids, i, 0, DIMENSIONS);
        int minClusterID = 0;


        // printf("x: %f, y: %f\n", coords[i * DIMENSIONS ], coords[i * DIMENSIONS + 1]);
        // printf("bod: %d, vzdalenost: %f od %f %f\n", i, min, centroids[0], centroids[1]);


        for (int center = 1; center < CLUSTERS; center++)
        {
            float actual = getDistanceBetweenPointsCUDA(coords, centroids, i, center, DIMENSIONS);
            // printf("vzdalenost: %f od %f %f\n", actual, centroids[center * DIMENSIONS], centroids[center * DIMENSIONS + 1]);
            if (actual < min)
            {
                min = actual;
                minClusterID = center;
            }
            
        }
        


        if (ids[i] != minClusterID)
        {
            atomicAdd(&changes[0], 1);
            ids[i] = minClusterID;
        }

        // printf("%d\n",ids[i]);
            

        for (int dim = 0; dim < DIMENSIONS; dim++)
            atomicAdd(&sum[minClusterID * DIMENSIONS + dim], coords[i * DIMENSIONS + dim]);
        atomicAdd(&count[minClusterID], 1);

    }

} 

void kmeans(graph & g)
{
    float * centroids = new float[DIMENSIONS * CLUSTERS];

    if (CPU_PRINT)
        g.wipeFile();

    initClusters(g, centroids);
    distributePointsIntoClusters(g);

    if (CPU_PRINT)
        g.printToFile();

    for (int i = 0; i < ITERATIONS; i++)
    {
        
        g.makeNewCentroids();
        distributePointsIntoClusters(g);

        if (CPU_PRINT)
            g.printToFile();

        if (g.isSameAsOld())
        {
            std::cout << "iter: " << i << std::endl;
            break;
        }
        // g.printToFile();
    }
    
}

bool isSameFloat(float a, float b)
{
    return (fabs(a * 0.99999) <= fabs(b) && fabs(a * 1.00001) >= fabs(b));
}

void cudaKmeans(graph & g)
{

    int *cudaPointsIDs, *pointsIDs, *cudaCount, *cudaChanges, changes[1];
    float *cudaPointsCoords, *pointsCoords, *cudaSum, *centroids, *cudaCentroids;

    centroids = new float[CLUSTERS * DIMENSIONS];

    initClusters(g, centroids);
    if (CUDA_PRINT)
        g.wipeFile();

    changes[0] = 0;


    pointsIDs = new int[POINTS];
    pointsCoords = new float[POINTS * DIMENSIONS];

    for (int i = 0; i < POINTS; i++)
    {
        pointsIDs[i] = g.getPoints()->at(i).id;
        for (int k = 0; k < DIMENSIONS; k++)
            pointsCoords[i * DIMENSIONS + k] = g.getPoints()->at(i).coords[k];

    }

    HANDLE_ERROR(cudaMalloc(&cudaPointsIDs, POINTS * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&cudaPointsCoords, POINTS * DIMENSIONS * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&cudaCentroids, CLUSTERS * DIMENSIONS * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&cudaSum, DIMENSIONS * CLUSTERS * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&cudaCount, CLUSTERS * sizeof(int)));
    HANDLE_ERROR(cudaMalloc(&cudaChanges, 1 * sizeof(int)));


    
    HANDLE_ERROR(cudaMemcpy(cudaPointsIDs, pointsIDs, POINTS * sizeof(int), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(cudaPointsCoords, pointsCoords, DIMENSIONS * POINTS * sizeof(float), cudaMemcpyHostToDevice));


    int iter = 0;
    do
    {
        float sum[CLUSTERS * DIMENSIONS];
        int count[CLUSTERS];

        for (int i = 0; i < CLUSTERS; i++)
        {
            for (int dim = 0; dim < DIMENSIONS; dim++)
                sum[i * DIMENSIONS + dim] = 0.0;
            count[i] = 0;
        }


        HANDLE_ERROR(cudaMemcpy(cudaCentroids, centroids, DIMENSIONS * CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
        

        HANDLE_ERROR(cudaMemcpy(cudaSum, sum,  DIMENSIONS * CLUSTERS * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(cudaCount, count, CLUSTERS * sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(cudaChanges, changes, sizeof(int), cudaMemcpyHostToDevice));


        int blockSizeX = 128;//4*gsizex;
        int numBlocksX = (POINTS + blockSizeX - 1) / blockSizeX;

        computeGPU<<<numBlocksX, blockSizeX>>>(POINTS, CLUSTERS, DIMENSIONS, cudaPointsIDs, cudaPointsCoords, cudaCentroids, cudaSum, cudaCount, cudaChanges);
        // cudaDeviceSynchronize();


        // computeNewCentroidsGPU<<<1, CLUSTERS>>>((point*)cudaPoints, (point*)cudaCentroids, cudaSum, cudaCount);

        cudaDeviceSynchronize();

        if (CUDA_PRINT)
        {
            HANDLE_ERROR(cudaMemcpy(pointsIDs, cudaPointsIDs, POINTS * sizeof(int), cudaMemcpyDeviceToHost));
            // HANDLE_ERROR(cudaMemcpy(pointsCoords, cudaPointsCoords, DIMENSIONS * POINTS * sizeof(float), cudaMemcpyDeviceToHost));

            // for (int i = 0; i < POINTS; i++)
            //     g.getPoints()->at(i).id = pointsIDs[i];
            // g.printToFile();
            g.printToFile(pointsIDs, pointsCoords);
        }


        HANDLE_ERROR(cudaMemcpy(sum, cudaSum, DIMENSIONS * CLUSTERS * sizeof(float), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(count, cudaCount, CLUSTERS * sizeof(int), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(changes, cudaChanges, sizeof(int), cudaMemcpyDeviceToHost));


        // std::cout << "changes: " << changes[0] << " points: " << POINTS << " treshold: " << ITER_TRESHOLD << std::endl;

        if (POINT_TRESHOLD && float(changes[0]) / float(POINTS) < ITER_TRESHOLD)
        {
            std::cout << "treshold iter: " << iter << std::endl;
            break;
        }

        changes[0] = 0;


        bool flag = true;
        for (int i = 0; i < CLUSTERS; i++)
        {
            for (int dim = 0; dim < DIMENSIONS; dim++)
            {
                if (flag && !isSameFloat(centroids[i * DIMENSIONS + dim], sum[i * DIMENSIONS + dim]/count[i]))
                    flag = false;
                centroids[i * DIMENSIONS + dim] = (sum[i * DIMENSIONS + dim]/count[i]);
                // std::cout << centroids[i * DIMENSIONS + dim] << '\t';
            }
            // std::cout << std::endl;
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


    
    // free(points);

    delete [] pointsIDs;
    delete [] pointsCoords;


    cudaFree(cudaPointsIDs);
    cudaFree(cudaPointsCoords);
    cudaFree(centroids);
    cudaFree(cudaSum);
    cudaFree(cudaCount);
    cudaFree(cudaChanges);
}

void parseArguments(int argc, char ** argv)
{
    if (argc == 1)
        return;
    CLUSTERS = atoi(argv[1]);
    ITERATIONS = atoi(argv[2]);

}

int main(int argc, char ** argv) 
{
    parseArguments(argc, argv);

    graph g = loadFile();

    // graph g = randomInput();

    clock_t begin = clock();
    cudaKmeans(g);
    clock_t end = clock();

    float elapsed_secs = float(end - begin) / CLOCKS_PER_SEC;
    std::cout << "cuda total: " << elapsed_secs << std::endl;

    begin = clock();
    //kmeans(g);
    end = clock();

    elapsed_secs = float(end - begin) / CLOCKS_PER_SEC;
    std::cout << "cpu: " << elapsed_secs << std::endl;


    return 0;
}
