import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision
NUM_EPOCH = 10
from torch.utils.data import DataLoader
import numpy as np
#from html import HTML
#import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

class ResNet50_CIFAR(nn.Module):
    def __init__(self):
        #Super: You call the constructure of your parent class
        super(ResNet50_CIFAR, self).__init__()
        ResNet50 = models.resnet50(pretrained=True)
        modules = list(ResNet50.children())[:-1]
        backbone = nn.Sequential(*modules)
        self.backbone = nn.Sequential(*modules)

        self.fc1 = nn.Linear(2048, 32)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(32, 10)
    def forward(self, img):
        out = self.backbone(img)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        return self.fc2(out)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def test():
        transform = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

        testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
        model = ResNet50_CIFAR()
        #model1 = model1.to('cpu')
        model.load_state_dict(torch.load('model', map_location=device))
        model.eval()

        final_images = []
        final_output = []
        final_predicted = []
        final_labels = []
        number = 1
        for i, data in enumerate(testloader, 0):
            if number > 100:
                break
    #    dataiter = iter(testloader)
            image, label = data
            output = model(image)
            _, predicted = torch.max(output.data, 1)
#            score = m(output).detach().numpy()
            final_images += list(image)
            final_output += list(output)
            final_predicted += list(predicted)
            final_labels += list(label)
            number += 1
            print number
        m = nn.Softmax()
        results = []
        for i in range(len(final_images)):
            image_path = 'images/' + str(i + 1) + '.png'
            torchvision.utils.save_image(final_images[i], image_path)
#            score = m(final_output[i]).detach().numpy()
            score = softmax(final_output[i].detach().numpy())
            results += [(image_path, score, classes[final_predicted[i]])]
#        print results
        html(results)
def html(results):
    with open('prediction.html', 'w') as file:
        file.write("""
        <html>
            <head>
                <title>Tests</title>
                <style>
                table {
                    border-collapse: collapse;
                    width: 100%;
                    text-align: center;
                }
                tr:nth-child(even){background-color: #f2f2f2}
                th {
                    background-color: #4CAF50;
                    color: white;
                    padding: 8px;
                }
                </style>

            </head>
            <body>
            <table border='1'>
            <thead>
            	<tr>
            		<th width='75px'><h3>image</h3></th>
            		<th width='75px'><h3>prediction</h3></th>
            		<th width='75px'><h3>plane</h3></th>
            		<th width='75px'><h3>car</h3></th>
            		<th width='75px'><h3>bird</h3></th>
            		<th width='75px'><h3>cat</h3></th>
            		<th width='75px'><h3>deer</h3></th>
            		<th width='75px'><h3>dog</h3></th>
            		<th width='75px'><h3>frog</h3></th>
            		<th width='75px'><h3>horse</h3></th>
            		<th width='75px'><h3>ship</h3></th>
            		<th width='75px'><h3>truck</h3></th>
        		</tr>
            </thead>
        """)

        for (image_path, score, predicted_label) in results:
            	file.write("<tr>\n")
            	file.write("<td width='75px'>\n<img src=\"%s\" height=75 width=75>\n</td>\n" % (image_path))
            	file.write("<td width='75px' style='color:%s'>\n<h2>%s</h2>\n</td>\n" % ('blue', predicted_label))
            	for s in score:
                    file.write("<td width='70px'>\n%.8f\n</td>\n" % (s))
            	file.write("</tr>\n")
        file.write("</table></body>\n</html>")
#htmlcode = HTML.table(results)
#print (htmlcode)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

if __name__ == '__main__':
    test()
