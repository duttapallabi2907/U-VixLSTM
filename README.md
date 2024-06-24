## UVixLSTM
Pallabi Dutta, [Soham Bose](https://github.com/shb2908), [Swalpa Kumar Roy](https://swalpa.github.io/), and [Sushmita Mitra](https://www.isical.ac.in/~sushmita/)


**Abstract**: The advancement of developing efficient medical image segmentation has evolved from initial dependence on Convolutional Neural Networks (CNNs) to the present investigation of hybrid models that combine CNNs with Vision Transformers. Furthermore, there is an increasing focus on creating architectures that are both high-performing in medical image segmentation tasks and computationally efficient to be deployed on systems with limited resources. Although transformers have several advantages like capturing global dependencies in the input data, they face challenges such as high computational and memory complexity. This paper investigates the integration of CNNs and Vision Extended Long Short-Term Memory (Vision-xLSTM) models by introducing a novel approach called **_UVixLSTM_**. The Vision-xLSTM blocks captures temporal and global relationships within the patches extracted from the CNN feature maps. The convolutional feature reconstruction path upsamples the output volume from the Vision-xLSTM blocks to produce the segmentation output. Our primary objective is to propose that Vision-xLSTM forms a reliable backbone for medical image segmentation tasks, offering excellent segmentation performance and reduced computational complexity. **_UVixLSTM_** exhibits superior performance compared to state-of-the-art networks on the publicly-available Synapse dataset.

![Architecture](uvixlstm.jpg)

## Datasets

The BTCV dataset can be accessed from [https://doi.org/10.7303/syn3193805](https://doi.org/10.7303/syn3193805)

## Note:
Install necessary Python packages using:
```
pip install requirements.txt

```
structure of model directory:
```
model
   |----UvisionLstm.py
   |----VisionLSTM.py

```
## Citation
If you like our work, please consider giving it a star ⭐ and cite us

          @article{dutta2024segmentation,
        	title={Are Vision xLSTM Embedded UNet More Reliable in Medical 3D Image Segmentation?},
		author={Dutta, Pallabi and Bose, Soham and Roy, Swalpa Kumar and Mitra, Sushmita},
		journal={arXiv},
		pp.={1-9},
		year={2024}
		}

## Acknowledgement

Part of this code is implementation from [https://github.com/nx-ai/vision-lstm]
