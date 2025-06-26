# FAIR-Pruner: FDR-based Automatic Identification and Removal

## Project Overview

**FAIR-Pruner** is an automated pruning tool based on FDR (False Discovery Rate), designed to prune redundant parts of deep learning models to improve computational efficiency while maintaining high accuracy. This tool can process a variety of common datasets such as **CIFAR-10** and **ImageNet2012 1K**. The project demonstrates how pruning algorithms can be applied to deep learning models.

![Original model](image/original_model.pdf)

![Pruned model](image/tiny_model.pdf)
## Requirements

- Python 3.7.7
- PyTorch 1.13.1
- torchvision 0.14.1+cu117
- scipy 1.10.1
- numpy
- pickle


## Dataset
This project uses the CIFAR-10 dataset for training and testing the model. The CIFAR-10 dataset contains 60,000 32x32 color images, divided into 10 classes, with 6,000 images per class. Out of the total dataset, 50,000 images are used for training, and 10,000 images are used for testing.

You can also use other datasets such as ImageNet2012 1K. Feel free to modify the code to fit the dataset you're working with.

# Preparatory work

## Install our FAIR-Pruner library
For example, install FAIR-pruner version 0.4. `FAIR_Pruner-0.4-py3-none-any.whl` is in the dist folder.
```{bash}
pip install dist/FAIR_Pruner-0.4-py3-none-any.whl
```
## Install Dataset and Model
To facilitate the demonstration of the pruning process, I provide `Mini_CIFAR10_640case.pkl` a small dataset with 640 pieces belonging to the CIFAR10 dataset and `CIFAR10_vgg16.pht` a VGG16 model trained on the CIFAR10 dataset.

# Pruning Example

## Import our method
```
from FAIR_Pruner import *
```
## Preset the necessary parameters
```
model_path = r'../CIFAR10_vgg16.pht'
data_path = r'../Mini_CIFAR10_640case.pkl'
results_save_path = r'../test_res.pkl'
tiny_model_save_path = r'../test_tiny_model.pht'
the_list_of_layers_to_prune = [2,4,7,9,12,14,16,19,21,23,26,28,30,35,38,41]
the_list_of_layers_to_compute_Distance = [3,5,8,10,13,15,17,20,22,24,27,29,31,36,39]
loss_function = nn.CrossEntropyLoss() # The loss function used when training the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_num = 10  # Determined by the prediction problem itself, the labels of the CFIAR10 dataset have ten categories
```
## Start pruning

### Calculate the Reconstruction Error and Distance
```
results = FAIR_Pruner_get_results(model_path, data_path, results_save_path,
            the_list_of_layers_to_prune,the_list_of_layers_to_compute_Distance,
            loss_function = loss_function, device, class_num,
            the_batch_for_compute_distance = 16, max_sample_for_compute_distance = 1e+10)
```
### Determine the number of neurons that should be prune off in each layer based on the FDR level
```
k_list = get_k_list(results, the_list_of_layers_to_prune, FDR_level = 0.05)
```
### Define a pruned Tiny network class
```
class Tiny_model_class(nn.Module):
    def __init__(self):
        super(Tiny_model_class, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64 - k_list[0], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64 - k_list[0], 64 - k_list[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64 - k_list[1], 128 - k_list[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128 - k_list[2], 128 - k_list[3], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128 - k_list[3], 256 - k_list[4], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 - k_list[4], 256 - k_list[5], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256 - k_list[5], 256 - k_list[6], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256 - k_list[6], 512 - k_list[7], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 - k_list[7], 512 - k_list[8], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 - k_list[8], 512 - k_list[9], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512 - k_list[9], 512 - k_list[10], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 - k_list[10], 512 - k_list[11], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512 - k_list[11], 512 - k_list[12], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear((512 - k_list[12]) * 7 * 7, 4096 - k_list[13]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096 - k_list[13], 4096 - k_list[14]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096 - k_list[14], 1000 - k_list[15])
        )
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

tiny_model = Tiny_model_class()
```
### Copy the parameters of the original model to the small network model and save tiny_model
```
tiny_model = Generate_model_after_pruning(tiny_model,model_path,
                             tiny_model_save_path,
                             results,k_list,
                             the_list_of_layers_to_prune)
```

# Final Thoughts
- The model saved after pruning is the final pruned version that can be used for further training or evaluation.
- You can experiment with different FDR levels to see how pruning affects the model’s performance.
- Make sure to adjust the layer pruning configurations (the_list_of_layers_to_prune, the_list_of_layers_to_compute_Distance) according to your model architecture.
