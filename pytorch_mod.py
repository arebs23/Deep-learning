import torch.nn as nn
import torch.functional as F
import torch


class MyneuralNet(nn.Module):
    def __init__(self,in_c, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d
 
        self.fc1 = nn.Linear(32 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, n_classes)

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)


        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        
        return x


###################################

"""Using Sequential blocks stack and merge layers"""


class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, 32, 
                                                  kernel_size=3, 
                                                   stride = 1, padding = 1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride = 1, padding = 1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.decoder = nn.Sequential(nn.Linear(32 * 28 * 28, 1024),
                                     nn.Sigmoid(),
                                     nn.Linear(1024, n_classes))
                                      
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x

                        

###################################

"""Using Sequential blocks stack and merge layers"""

def conv_block(in_c, out_la, *args, **kwargs):
    return nn.Sequential(nn.Conv2d(in_c, out_la, *args, **kwargs),
                         nn.BatchNorm2d(out_la),
                         nn.ReLU())


def deconv_block(in_c, out_la):
    return nn.Sequential(nn.Linear(in_c, out_la),
                         nn.Sigmoid()
                                  )


class MyCNNClassifier1(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.conv1 = conv_block(in_c, 32, kernel_size=3, stride = 1, padding = 1)
        self.conv2 = conv_block(32, 64, kernel_size=3, stride = 1, padding = 1)


        self.decoder = nn.Sequential(nn.Linear(32 * 28 * 28, 1024),
                                     nn.Sigmoid(),
                                     nn.Linear(1024, n_classes))

    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.decoder(x)


class Encoder(nn.Module):
    def __init__(self, en_sizes):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[conv_block(in_c, out_la,kernel_size=3, 
                                                      stride = 1, padding = 1) 
                                    for in_c, out_la in zip(en_sizes, en_sizes[1:])])


    def forward(self, x):
        x = self.conv_blocks(x)
        return x

class Decoder(nn.Module):
    def __init__(self, de_sizes, n_classes):
        super().__init__()
        self.deconv_blocks = nn.Sequential(*[deconv_block(in_c, out_la)
                              for in_c, out_la in zip(de_sizes, de_sizes[1:])])
        self.last = nn.Linear(de_sizes[-1], n_classes)


    def forward(self, x):
        x = self.deconv_blocks(x)
        x = self.last(x)
        return x



class CNNClassifier(nn.Module):
    def __init__(self, in_c, en_sizes, de_sizes, n_classes):
        super().__init__()
        self.en_sizes = [in_c, *en_sizes]
        self.de_sizes = [32 * 28 * 28, *de_sizes]
        self.encoder = Encoder(en_sizes)
        self.decoder = Decoder(de_sizes, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        x = self.decoder(x)
        return x


















def main():
   model = CNNClassifier(1, [32,64], [1024, 512], 10)
   print(model)


if __name__ == '__main__':
    main()