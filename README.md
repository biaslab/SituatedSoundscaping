# Towards Situated Soundscaping in Hearing Aids

This repository provides the files and code developen by Bart van Erp for his graduation project.
The goal of the project is to develop a framework, which allows users to develop their own noise reduction or speech enhancement algorithms by recording a short fragment of the sound they want to alter. By training the signal models of the constituent sounds in the observed mixture, these sounds can be separated through informed source separated, which is implemented as probabilistic inference through message passing on factor graphs. The separated sounds allow the user to perform source-specific filtering and allows for on the spot ("situated") soundscaping.

The files in this repository are structured as followed:
```
    01-data - Provides the sound fragments used during the experiments.
    02-functions - Provides functions written in Julia to make the workflow easier.
    03-extensions - Provides extensions to the ForneyLab toolbox in order to allow for the probabilistic phase vocoder (PPV) and Gaussian scale mixture models (GSMM) to work properly.
    04a-AR_modelling - Provides (experimental) code for modelling and source separation using auto-regressive models.
    04b-PPV_modelling - Provides (experimental) code for modelling and source separation using probabilistic phase vocoders.
    04c-GSMM_modelling - Provides (experimental) code for modelling and source separation using Gaussian scale mixture models.
    04d-GSMM_realtime - Provides a realtime (approximate) implementation of the GSMM, where source separation processes 1 second of mixture data into approximately 200ms.
    05-thesis - Provides the final files for all three individual models.
```