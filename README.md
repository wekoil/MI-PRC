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

### Analýza a hodnocení vlastností dané implementace programu.

## Kapitola 3

### Závěr

Pro algoritmy, které se dají velice dobře paralelizovat a nemají kritické sekce, jako je například tento, je výpočet na GPU o mnoho rychlejší. Počet dimenzí nemá na složitost výpočtu žádný vliv.

### (případně) Literatura