# K means

## Kapitola 1

### Definice problému

Pro zadané body je naším úkolem jejich roztřídění do clusterů. Bod vždy náleží do clusteru, který má nejmenší euklidovskou vzdálenost do jeho středu. Poté co všechny body roztřídíme, aktualizujeme střed každého clusteru tak, aby byl středem všech bodů, které do něho náleží. Tímto končí jedna iterace. Iterace probíhají, dokud se nějaké body přesouvají do jiného clusteru nebo dokud počet iterací nepřekročil určitou mez.

### Popis sekvenčního algoritmu a jeho implementace

Pro sekvenční algoritmus si alokujeme body a několik clusterů (počet lze nakonfigurovat). Clustery se náhodně namapují na nějaký bod. V jednotlivých iteracích poté přiřazuji jednotlivé body vždy k nejbližšímu clusteru. Na konci jednotlivé iterace vypočítám nový střed clusteru. Pokud se střed od minulé iterace nepohnul, nemá cenu pokračovat a program končí. Max počet iterací se nechá opět nastavit.

## Kapitola 2 (Pro CUDA)

### Popis případných úprav algoritmu a jeho implementace verze pro grafickou kartu, včetně volby datových struktur

Verze pro GPU se ani moc nezměnila, jen se každý bod počítá ve svém vlastním vlákně, které běží paralelně. Nový střed clusterů se opět spočítá na CPU. Pro souřadnice je použit datový typ float. Pro GPU verzi jsem přidal další parametr - treshold. Program počítá, kolika bodům se změní cluster v jednotlivé iteraci, pokud je to méně než treshold program skončí. Tento přístup nám ušetří několik iterací, které se skoro vůbec nemění, jak je vidět na výstupech.

### Tabulkově a případně graficky zpracované naměřené hodnoty časové složitosti měřených instancí běhu programu s popisem instancí dat, přepočet výkonnosti programu na MIPS nebo MFlops, včetně zrychlení pro různé ex. konfigurace, porovnání s CPU verzí.

Pokud není uvedeno jinak je výpočet pro 3 clustery. Časy jsou uvedené ve vteřinách.

Pro dimenzi 2

| Points        | CPU    | GPU   |
| ------------- |:------:| -----:|
| 10            | 0      | 1.15  |
| 100           | 0      | 1.13  |
| 1000          | 0.01   | 1.12  |
| 10000         | 0.07   | 1.08  |
| 100000        | 1.34   | 1.17  |
| 1000000       | 35.57  | 1.52  |

Pro dimenzi 10

| Points        | CPU    | GPU   |
| ------------- |:------:| -----:|
| 10            | 0      | 1.06  |
| 100           | 0      | 1.18  |
| 1000          | 0      | 1.17  |
| 10000         | 0.14   | 1.17  |
| 100000        | 2.16   | 1.06  |
| 1000000       | 40.8   | 1.59  |

Pro 100000 bodu

| Dimenze       | CPU    | GPU   |
| ------------- |:------:| -----:|
| 2             | 2.8    | 1.22  |
| 5             | 2.57   | 1.12  |
| 10            | 2.58   | 1.18  |
| 20            | 2.7    | 1.16  |
| 30            | 3.58   | 1.14  |
| 40            | 2.8    | 1.17  |
| 50            | 2.48   | 1.18  |

Pro 100000 bodu a 5 dimenzí

| Clusterů      | CPU    | GPU   |
| ------------- |:------:| -----:|
| 2             | 1.32   | 1.16  |
| 5             | 2.43   | 1.11  |
| 10            | 5.49   | 1.14  |
| 20            | 28.19  | 1.18  |
| 30            | 43.41  | 1.27  |
| 40            | 58.62  | 1.21  |
| 50            | 110.05 | 1.27  |

Pro 10 clusterů a 5 dimenzí

| Points        | GPU    |
| ------------- |:------:|
| 1000000       | 1.57   |
| 10000000      | 9.54   |
| 100000000     | 170.19 |

Vše měřené na Tesla K40c

### Vše doplnit výpisem hodnot z profileru včetně komentáře.

iter: 54
cuda total: 0.44
cpu: 0
==22805== Profiling application: ./a.out
==22805== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.33%  31.254ms        55  568.26us  565.03us  613.19us  computeGPU(int, int, int, int*, float*, float*, float*, int*, int*)
                    0.99%  314.24us       222  1.4150us     864ns  77.664us  [CUDA memcpy HtoD]
                    0.68%  216.35us       165  1.3110us  1.2800us  2.3040us  [CUDA memcpy DtoH]
      API calls:   91.61%  474.84ms         6  79.141ms  8.7070us  474.27ms  cudaMalloc
                    6.08%  31.510ms        55  572.91us  568.78us  615.89us  cudaDeviceSynchronize
                    1.33%  6.8904ms       387  17.804us  10.505us  385.53us  cudaMemcpy
                    0.46%  2.3678ms       288  8.2210us     284ns  491.63us  cuDeviceGetAttribute
                    0.26%  1.3633ms        55  24.787us  19.049us  252.63us  cudaLaunchKernel
                    0.11%  577.86us         3  192.62us  94.012us  362.61us  cuDeviceTotalMem
                    0.10%  526.32us         6  87.720us  3.4740us  263.94us  cudaFree
                    0.04%  215.29us         3  71.763us  60.780us  93.243us  cuDeviceGetName
                    0.00%  15.504us         3  5.1680us  3.1030us  8.7980us  cuDeviceGetPCIBusId
                    0.00%  6.5230us         6  1.0870us     343ns  2.3090us  cuDeviceGet
                    0.00%  2.9690us         3     989ns     327ns  1.7290us  cuDeviceGetCount
                    0.00%  1.2960us         3     432ns     358ns     510ns  cuDeviceGetUuid

### Analýza a hodnocení vlastností dané implementace programu.

## Kapitola 3

### Závěr

Pro algoritmy, které se dají velice dobře paralelizovat a nemají kritické sekce, jako je například tento, je výpočet na GPU o mnoho rychlejší. Počet dimenzí nemá na složitost výpočtu žádný vliv.

### (případně) Literatura