# Urban change detection and dating using Sentinel time series data: Exploring the potential of a robust urban extractor

# To do:
- 2021-05-06 Rework bad data handling. Single sensor data should be used in case data from either sensor is corrupted.
(tried this one but doesn't work!)
- 2021-05-06 Add masks to SpaceNet7_S1S2_Dataset and ad support for masks
- 2021-05-06 Fix OSCD Multitemporal Dataset Labels (size does not correspond to satellite images)



# Added changes:
- 2021-05-14 Reworked data preprocessing
- 2021-05-14 Added support for spacenet7_s1s2_dataset_v2 and switching between datasets is not possible
(but broke OSCD dataset)
- 2021-05-14 Added time series length plot