# Accelerating Atmospheric Turbulence Simulation via Learned Phase-to-Space Transform

This repository contains the code for the following paper:

Zhiyuan Mao, Nicholas Chimitt, and Stanley H. Chan, ‘‘Accelerating Atmospheric Turbulence Simulation via Learned Phase-to-Space Transform’’, accepted to ICCV 2021

The soruce of Oiringal Repository: https://github.itap.purdue.edu/StanleyChanGroup/TurbulenceSim_P2S

Arxiv: 
https://arxiv.org/abs/2107.11627

How to use: 
  - Check the demo.py file
  - The generation of correlation matrix for point spread functions is slow. You can download the correlation matrices prepared by us and put them under the data folder. The correlation matrices are available at: 
  
    https://drive.google.com/file/d/1W5yyXiPQSKssIwIdsIMofBN7eZ9nuApx/view?usp=sharing

The code with simulator_old.py is tested with the following environment: 
  - Python      3.6
  - Pytorch     1.4.0
  - Numpy	    1.19.2
  - Scipy       1.6.0
  - Matplotlib  3.3.6
  
**Update: we modify the code to work with pytorch 1.10 and beyond. Please use the latest simulator.py. Nothing else has changed. For pytorch 1.4, use simulator_old.py. The difference is due to the pytorch's update on the fft module.**

If you find our work helpful in your research, please consider cite our paper

```
@InProceedings{Mao_2021_ICCV,
author = {Zhiyuan Mao and Nicholas Chimitt and Stanley H. Chan},
title = {Accelerating Atmospheric Turbulence Simulation via Learned Phase-to-Space Transform},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2021}
} 
```
  
Please also check out our other work on Atmospheric Turbulence: 
  - Project page
  
    https://engineering.purdue.edu/ChanGroup/project_turbulence.html
  
  - Code for image reconstruction through atmospheric turbulence: 
  
    https://github.itap.purdue.edu/StanleyChanGroup/TurbRecon_TCI
    
    
> *This software is made available for evaluation purposes only and has features which are patent pending.*

## LICENSE

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
