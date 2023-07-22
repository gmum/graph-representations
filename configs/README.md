# Configuration files

Directories `data` and `representation` contain configuration files required to reproduce results of the study.
Directory `models` contains an exemplary configuration file and a script to generate all possible hyperparameter configurations.

All configuration files are parsed by methods implemented in `graphrepr.config`.

## Dataset configuration files

These configuration files specify preprocessing, cost function, and data file paths for each dataset. **The paths should be updated to ensure they point to the correct data file destinations.**

ESOL (scaffold) and QM9 datasets are provided in the repository while the remaining datasets (including splits) are publicly available and can be obtained as described in Section "Availability of data and materials".


## Model configuration files

File `1-model.cfg` is an example of a configuration of model architecture.
To create all possible configurations of architectures, enter `models` directory and use this command:

```bash
python make_configs.py
```

The list of the 100 architectures used in the study is the following:

`2754-model.cfg 361-model.cfg 613-model.cfg 1790-model.cfg 2095-model.cfg 1202-model.cfg 2177-model.cfg 14-model.cfg 1171-model.cfg 2072-model.cfg 1828-model.cfg 2260-model.cfg 941-model.cfg 1920-model.cfg 739-model.cfg 1265-model.cfg 2042-model.cfg 1119-model.cfg 1168-model.cfg 2751-model.cfg 1637-model.cfg 1850-model.cfg 982-model.cfg 1736-model.cfg 287-model.cfg 2356-model.cfg 823-model.cfg 527-model.cfg 914-model.cfg 608-model.cfg 1037-model.cfg 2261-model.cfg 1018-model.cfg 996-model.cfg 1771-model.cfg 879-model.cfg 284-model.cfg 2277-model.cfg 976-model.cfg 1294-model.cfg 2306-model.cfg 387-model.cfg 726-model.cfg 397-model.cfg 3036-model.cfg 130-model.cfg 3240-model.cfg 3004-model.cfg 119-model.cfg 1105-model.cfg 429-model.cfg 2876-model.cfg 3180-model.cfg 240-model.cfg 168-model.cfg 247-model.cfg 2448-model.cfg 2875-model.cfg 2292-model.cfg 2243-model.cfg 111-model.cfg 741-model.cfg 3034-model.cfg 1652-model.cfg 1278-model.cfg 2567-model.cfg 1380-model.cfg 359-model.cfg 1916-model.cfg 2472-model.cfg 2195-model.cfg 1743-model.cfg 1865-model.cfg 1816-model.cfg 2940-model.cfg 528-model.cfg 3008-model.cfg 3024-model.cfg 626-model.cfg 3140-model.cfg 2811-model.cfg 2701-model.cfg 3237-model.cfg 517-model.cfg 1409-model.cfg 2643-model.cfg 447-model.cfg 1940-model.cfg 70-model.cfg 406-model.cfg 2179-model.cfg 1141-model.cfg 1990-model.cfg 2806-model.cfg 1524-model.cfg 917-model.cfg 75-model.cfg 2973-model.cfg 3228-model.cfg 1249-model.cfg`

## Representation configuration files

These configuration files define representations used in the study.
