The **CMIP6-MedPlus** dataset provides daily **precipitation** and **mean, minimum, and maximum temperature** downscaled to **0.25° spatial resolution** over an extended Mediterranean domain for the period **1986–2100**.

The dataset is derived from a statistical downscaling of an ensemble of **CMIP6 Global Climate Models (GCMs)** participating in the **ScenarioMIP** experiment. For each model, two **Shared Socioeconomic Pathways (SSPs)** are considered:

- **SSP1-2.6**
- **SSP3-7.0**

---

## Downscaling methodology

The CMIP6-MedPlus climate archive was developed using a dedicated **two-stage statistical downscaling framework**:

1. **CNN-based regridding**  
   A Convolutional Neural Network (CNN) was trained to spatially regrid coarse-resolution CMIP6 climate fields to 0.25°. The training was performed using **ERA5** data as the high-resolution reference, allowing the model to learn fine-scale spatial patterns and variability.

2. **Bias adjustment via Quantile Delta Mapping (QDM)**  
   The CNN-downscaled outputs were subsequently bias-adjusted using the **Quantile Delta Mapping (QDM)** method. QDM corrects systematic biases while preserving the modeled climate change signal in the distribution of the variables. **ERA5** was used as the reference dataset.

---

## Repository contents

This repository contains the **Python code** used to generate the CMIP6-MedPlus dataset. Specifically, for each variable, the repository includes:

- A script implementing the **CNN-based regridding**
- A script implementing the **QDM bias correction** procedure


---

## Data availability

The resulting CMIP6-MedPlus dataset is **freely available** on **Zenodo**:

https://doi.org/10.5281/zenodo.17898529

---

## Funding

This work was supported by OurMED PRIMA Program project funded by the European Union's Horizon 2020 research and innovation under grant agreement No. 2222

More information is available at:  
https://www.ourmed.eu/

---

## License

Please refer to the Zenodo record for licensing information associated with the dataset.