import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
NUM_EPOCH = 10
from torch.utils.data import DataLoader
#uncomment fot using GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        #Super: You call the constructure of your parent class
        super(ResNet50_CIFAR, self).__init__()
        # Initialize ResNet 50 with ImageNet weights
        # ResNet50 up to its average pooling layer
        # using the pretrained model (ResNet50)
        # it automotically downloads it if you don't have it
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        #nn.sequential: list or dict of layers
        backbone = nn.Sequential(*modules)
        # Create new layers
        self.backbone = nn.Sequential(*modules)
        # A small network with 32 hidden nodes
        # The input dimension is 2048
        # 2048 input, 32 output
        # nn.Linear(input dimension, number of hidden nodes)
        self.fc1 = nn.Linear(2048, 32)
        self.dropout = nn.Dropout(p=0.5)
        # 10 outputs
        self.fc2 = nn.Linear(32, 10)
#forward function gets the data, img and passes the data thorough the network
    def forward(self, img):
        # Get the flattened vector from the backbone of resnet50
        out = self.backbone(img)
        # processing the vector with the added new layers
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)

def train():
    ## Define the training dataloader
    # This part transforms the data
    # trnsforms are common image transforms
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5),
                                                         (0.5, 0.5, 0.5))])
    #1)transform turns it to a tensor
    #data set gets access to 1 dataset, However, dataloader combines all that stuff to get batches of data+multiprocessing
    trainset = datasets.CIFAR10('./data', download=True, transform=transform)
    #2)create a dataloader
    #dataloader is for covering a batch of data which basically gets the data set, and can return batches of dataset
    #gives you multi processing, each image is calculated in parallel
    #shuffling the data with training data set makes sense
    trainloader = DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
#    trainloader.to(device)
    #Added by Nadia
#uncomment this for using GPU
#    net.to(device)
#    inputs, labels = inputs.to(device), labels.to(device)
    #Added by Nadia
    ## Create model, objective function and optimizer
    # adding to model to GPU
    model = ResNet50_CIFAR().to(device)
    criterion = nn.CrossEntropyLoss()           #loss
    optimizer = optim.SGD(list(model.fc1.parameters()) + list(model.fc2.parameters()),
                           lr=0.001, momentum=0.9)
    ## Do the training
    for epoch in range(NUM_EPOCH):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
        #each data is one batch
            # get the inputs
            # data will move to GPU if the current device is GPU
#            data.to(device)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # at each step, you have to set the gradient of the optimizer to 0
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # pass the input data through the model (forward pass)
            outputs = model(inputs)
            # get the prediction
            loss = criterion(outputs, labels)
            # now we do the backward pass
            loss.backward()
            # optimize our step (updates the parameters of the model)
            optimizer.step()
            #next step is the eval mode, and then pass the data through the model and get the prediction

            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 20))
                running_loss = 0.0
    print('Finished Training')
    torch.save(model.state_dict(), 'mytraining.pt')

if __name__ == '__main__':
    train()

#should split the data in to training and validation set
#validation data:
#model.eval()
#with torch.no_grad():
#    y_ = model(x)
