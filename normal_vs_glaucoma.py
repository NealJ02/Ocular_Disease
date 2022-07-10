import os
import pandas as pd
import pdb
import shutil
import torchvision
import wandb
import torch
from sklearn.metrics import precision_recall_fscore_support

torch.manual_seed(2022)

wandb.init(project="NormalVSCat", entity="nealj06", name="normal_vs_cat_1_noweightdecay")
wandb.config = {
  "learning_rate": 0.001,
  "epochs": 1,
  "batch_size": 8
}


#get data from data folder
dataset = torchvision.datasets.ImageFolder(
    root = 'data_subset',
    transform = torchvision.transforms.ToTensor()
)

trainlen = int(0.8*len(dataset))
trainset, testset = torch.utils.data.random_split(dataset, 
                    [trainlen, len(dataset)-trainlen])

category_count = {}
for category, index in dataset.class_to_idx.items():
	category_count[index] = sum([x[1] == index for x in dataset.imgs])

weights = []
for img in dataset.imgs:
	weights.append(max(category_count.values())/category_count[img[1]])

train_weights = [weights[i] for i in trainset.indices]
train_sampler = torch.utils.data.WeightedRandomSampler(train_weights, len(train_weights))

train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=8, sampler=train_sampler)

test_weights = [weights[i] for i in testset.indices]
test_sampler = torch.utils.data.WeightedRandomSampler(test_weights, len(test_weights))

test_dataloader = torch.utils.data.DataLoader(testset, batch_size=8, sampler = test_sampler)

#pdb.set_trace()
#make and train model

cnn_neural_network = torchvision.models.resnet18(pretrained=True)
cnn_neural_network.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
cost_function = torch.nn.CrossEntropyLoss()

gradient_descent = torch.optim.Adam(cnn_neural_network.parameters(), lr=0.001)

cnn_neural_network.train()
count = 0;
for data in train_dataloader:
    input_images, output_categories = data
    gradient_descent.zero_grad()
    model_predictions = cnn_neural_network(input_images)
    model_predictions_discrete = torch.argmax(model_predictions, dim=1)
    precision, recall, f1score, support = precision_recall_fscore_support(output_categories, model_predictions_discrete,
                                            labels=(0,1), zero_division=0)
    for category_class in dataset.class_to_idx.keys():
        category_index = dataset.class_to_idx[category_class]
        category_f1score = f1score[category_index]
        wandb.log({"train f1score "+category_class : f1score[category_index]})

    cost = cost_function(model_predictions, output_categories)
    wandb.log({"train loss": cost})
    wandb.watch(cnn_neural_network)
    cost.backward()
    gradient_descent.step()

    if count % 3 == 0:
        cnn_neural_network.eval()
        input_images_test, output_categories_test = iter(test_dataloader).next()
        model_predictions_test = cnn_neural_network(input_images_test)
        model_predictions_test_discrete = torch.argmax(model_predictions_test, dim=1)
        cost_test = cost_function(model_predictions_test, output_categories_test)
        wandb.log({"test loss": cost_test})  
        cnn_neural_network.train()
        precision, recall, f1score, support = precision_recall_fscore_support(output_categories_test, model_predictions_test_discrete,
                                            labels=(0,1), zero_division=0)
        for category_class in dataset.class_to_idx.keys():
            category_index = dataset.class_to_idx[category_class]
            category_f1score = f1score[category_index]
            wandb.log({"test f1score "+category_class : f1score[category_index]})
    count += 1