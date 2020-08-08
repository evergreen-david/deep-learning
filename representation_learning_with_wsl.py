from PIL import Image
from torchvision import transforms

import torch
model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
# model.eval()

class ImageDataset(Dataset):
    def __init__(self):
        self.path = ['Male/asdfsadf.jpg',
                     'Female/afdf.jpg'
                     ...
                    ]

    def __len__(self):
        len(self.path)


    def __getitem__(self, index):
        filename = self.path[index]
        input_image = Image.open(filename)

        preprocessed_image = self.preprocess(input_image)
        print(preprocessed_image)

        label = torch.tensor(  self.labler(filename)  )

        return preprocessed_image, label


    def labler(self, path):
        cat2index = {'Male':0, 'Femail':1}
        return cat2index[ path.split('/')[0] ] 


    def preprocess(self, image):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess


dataloader = DataLoader(dataset=ImageDataset(), batch_size=50, shuffle=True)


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        self.linear = nn.Linear( 512 * 7* 7, 2 ) 

        for parameter in self.base.parameters():
            parameter.requires_grad = False


    def forward(self, inputs):
        representations = self.base( inputs )
        flattened = representations.view( -1, 512*7*7 ) 
        logits = self.linear( flattened ) 
        return logits


net = ImageClassifier()
optimizer = optim.SGD(net.parameters(), lr = 0.01)
loss_fn = nn.CrossEntropyLoss()


for i in range(100):
    for x, y in dataloader:
        predictions = net(x)
        loss = loss_fn(predictions, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(loss)
