# synthcity_experiments
Experimental study of similarity and quality metrics of synthetic tabular data obtained from deep generative models.

In this work, the [synthcity](https://github.com/vanderschaarlab/synthcity/tree/main) library was used, which provides the ability to evaluate the quality of generated synthetic data using metrics, and also allows you to add your own models for generating synthetics using the interface 

## 🚀 GPU environment setup

To repeat experiments on generating synthetic data for models that require GPU calculations, it is recommended to create a container on a GPU that supports CUDA based on the Dockerfile located in the synth_test folder as follows:

```bash
docker build -t <image_name> .
```

Docker will automatically install all the necessary libraries from the file [requirements.txt](https://github.com/Samp0rt/synthcity_experiments/blob/main/synth_tests/requirements.txt) while building image.

## 🏭 Generating data
### CPU
To generate data using models that support training only on CPU run *fit_and_generate.py* from CPU_train folder using:

```bash
python3.9 fit_and_generate.py
```
This code will create folder [generated_data](https://github.com/Samp0rt/synthcity_experiments/tree/main/generated_data) with folders for every dataset in data folder.

### GPU

If case of using image from the Dockerfile you need to run *fit_and_generate.py* manually from GPU_train folder using:

```bash
python3.9 -m fit_and_generate.py
```
The results will be saved exactly like for CPU models in generated_data folder.

## ⚡ Benchmark evaluation

To evaluate synthcity benchmark simply run *benchmark.py*:

```bash
python3.9 benchmark.py
```
All the results will be saved in csv format in [benchmark_results](https://github.com/Samp0rt/synthcity_experiments/tree/main/benchmark_results) folder.

## 📓 Metrics analysis 

The analysis of metrics and models for which these metrics were calculated is in the file [metrics_analysis.ipynb](). There, all metrics are examined in detail and the most sensitive ones are selected, as well as the models that cope with generation best of all are selected.

❗ Note: some parts of code in *metrics_analysis.ipynb* use information about datasets properties from *df_properties.csv* file.
