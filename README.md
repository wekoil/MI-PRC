# K means

## Kapitola 1

### Definice problému

Pro zadané body je naším úkolem jejich roztřídění do clusterů. Bod vždy náleží do clusteru, který má nejmenší euklidovskou vzdálenost do jeho středu. Poté co všechny body roztřídíme, aktualizujeme střed každého clusteru tak, aby byl středem všech bodů, které do něho náleží. Tímto končí jedna iterace. Iterace probíhají, dokud se nějaké body přesouvají do jiného clusteru nebo dokud počet iterací nepřekročil určitou mez.

### Popis sekvenčního algoritmu a jeho implementace

Pro sekvenční algoritmus si alokujeme body a několik clusterů (počet lze nakonfigurovat). Clustery se náhodně namapují na nějaký bod. V jednotlivých iteracích poté přiřazuji jednotlivé body vždy k nejbližšímu clusteru. Na konci jednotlivé iterace vypočítám nový střed clusteru. Pokud se střed od minulé iterace nepohnul, nemá cenu pokračovat a program končí. Max počet iterací se nechá opět nastavit.

## Kapitola 2 (Pro CUDA)

### Popis případných úprav algoritmu a jeho implementace verze pro grafickou kartu, včetně volby datových struktur

Verze pro GPU se ani moc nezměnila, jen se každý bod počítá ve svém vlastním vlákně, které běží paralelně. Nový střed clusterů se opět spočítá na CPU. Pro souřadnice je použit datový typ float. Pro GPU verzi jsem přidal další parametr - treshold. Program počítá, kolika bodům se změní cluster v jednotlivé iteraci, pokud je to méně než treshold program skončí. Tento přístup nám ušetří několik iterací, které se skoro vůbec nemění.

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

Počet iterací s/bez tresholdu

| Points    | S   | Bez |
| --------- |:---:| ---:|
| 10        | 1   |  1  |
| 100       | 4   |  4  |
| 1000      | 8   |  8  |
| 10000     | 45  |  25 |
| 100000    | 47  |  27 |
| 1000000   | 60  |  17 |
| 10000000  | 93  |  12 |
| 100000000 | 159 |  26 |

Treshold byl nastaven tak, aby výpočet utnul, pokud se změnilo pouze 1 promile bodů.

Vše měřené na Tesla K40c

### Vše doplnit výpisem hodnot z profileru včetně komentáře.

Vše níže uvedené je pro 2 dimenze a 3 clustery.

Výpis z profileru pro 10 bodů

<pre>
iter: 1
cuda total: 0.38
==23330== Profiling application: ./a.out
==23330== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   43.78%  13.856us         2  6.9280us  6.2080us  7.6480us  computeGPU(int, int, int, int*, float*, float*, float*, int*, int*)
                   30.03%  9.5040us        10     950ns     864ns  1.4720us  [CUDA memcpy HtoD]
                   26.19%  8.2890us         6  1.3810us  1.2800us  1.5360us  [CUDA memcpy DtoH]
      API calls:   98.52%  256.96ms         6  42.827ms  7.6370us  256.92ms  cudaMalloc
                    0.90%  2.3564ms       288  8.1810us     297ns  327.65us  cuDeviceGetAttribute
                    0.24%  625.69us         3  208.56us  99.494us  396.93us  cuDeviceTotalMem
                    0.11%  276.66us        16  17.291us  10.649us  50.152us  cudaMemcpy
                    0.10%  254.64us         2  127.32us  24.973us  229.67us  cudaLaunchKernel
                    0.08%  214.57us         3  71.522us  64.240us  85.132us  cuDeviceGetName
                    0.03%  68.899us         6  11.483us  2.3110us  25.996us  cudaFree
                    0.01%  25.760us         2  12.880us  9.7390us  16.021us  cudaDeviceSynchronize
                    0.01%  18.710us         3  6.2360us  3.7080us  9.0220us  cuDeviceGetPCIBusId
                    0.00%  4.9890us         6     831ns     353ns  1.5750us  cuDeviceGet
                    0.00%  3.2880us         3  1.0960us     445ns  2.0700us  cuDeviceGetCount
                    0.00%  1.3870us         3     462ns     375ns     552ns  cuDeviceGetUuid
</pre>

Výpis z profileru pro 1000000 bodů

<pre>
iter: 69
cuda total: 0.83
==23217== Profiling application: ./a.out
==23217== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.85%  389.27ms        70  5.5609ms  5.5389ms  6.4137ms  computeGPU(int, int, int, int*, float*, float*, float*, int*, int*)
                    1.08%  4.2386ms       282  15.030us     928ns  2.7386ms  [CUDA memcpy HtoD]
                    0.08%  298.59us       210  1.4210us  1.3440us  1.9200us  [CUDA memcpy DtoH]
      API calls:   59.44%  389.59ms        70  5.5655ms  5.5387ms  6.4174ms  cudaDeviceSynchronize
                   37.85%  248.08ms         6  41.347ms  7.6630us  247.51ms  cudaMalloc
                    1.89%  12.396ms       492  25.195us  9.6170us  3.1568ms  cudaMemcpy
                    0.31%  2.0456ms       288  7.1020us     263ns  297.00us  cuDeviceGetAttribute
                    0.26%  1.6817ms        70  24.024us  19.480us  251.58us  cudaLaunchKernel
                    0.13%  838.31us         6  139.72us  3.9330us  522.24us  cudaFree
                    0.08%  545.65us         3  181.88us  90.401us  338.76us  cuDeviceTotalMem
                    0.03%  191.78us         3  63.925us  58.658us  73.076us  cuDeviceGetName
                    0.00%  19.736us         3  6.5780us  3.9800us  9.5790us  cuDeviceGetPCIBusId
                    0.00%  4.2660us         6     711ns     318ns  1.3200us  cuDeviceGet
                    0.00%  3.1310us         3  1.0430us     370ns  2.0500us  cuDeviceGetCount
                    0.00%  1.1160us         3     372ns     322ns     460ns  cuDeviceGetUuid
</pre>

Je zajímavé, že pokud spustím program přes profiler, běží kratší dobu, než při spuštění z příkazové řádky. Pro malý počet bodů trvá dlouho alokování paměti, při velkém počtu bodů se čeká i na synchronizaci vláken.

### Analýza a hodnocení vlastností dané implementace programu.

Z výsledků měření je dobře vidět, že s většujícím se počtem bodů nebo počtem clusterů se více vyplatí výpočet na GPU. Počet dimenzí má na složitost výpočtu minimální vliv.

## Kapitola 3

### Závěr

Pro algoritmy, které se dají velice dobře paralelizovat a nemají kritické sekce, jako je například tento, je výpočet na GPU pro velký počet bodů (alespoň 1 milion) o mnoho rychlejší než na klasickém CPU. Pro malý počet bodů se vyplatí používat pouze CPU verzi, jelikož se nevyplatí alokovat a kopírovat vstup na GPU a implementace je jednodušší.