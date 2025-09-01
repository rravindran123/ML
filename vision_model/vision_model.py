#import pytorch
import torch
import torch.nn as nn

#import torchvision
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

#import mathplot
import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

#set up the training data
train_data = datasets.FashionMNIST(
    root = "data",
    train= True,
    download = True,
    transform = ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

#set up the testing data
test_data = datasets.FashionMNIST(
    root = "data",
    train= False,
    download = True,
    transform = ToTensor(), # images come as PIL format, we want to turn into Torch tensors
)

image, label = train_data[0]
print(image, label)

print(image.shape)
print(len(train_data.data), len(train_data.targets), len(test_data.data), len(test_data.targets))

class_names = train_data.classes
print(class_names)

#visualizing the data
# image, label = train_data[0]
# print(f"Image shape: {image.shape}")
# plt.imshow(image.squeeze()) # image shape is [1, 28, 28] (colour channels, height, width)
# #plt.title(label)
# plt.show()

torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows, cols = 4,4

# for i in range(1, rows*cols + 1):
#     random_idx = torch.randint(0, len(train_data), size=[1]).item()
#     img, label = train_data[random_idx]
#     fig.add_subplot(rows, cols, i)
#     plt.imshow(img.squeeze(), cmap="grey")
#     plt.title(class_names[label]) 
#     plt.axis(False)
#plt.show()

from torch.utils.data import DataLoader
# set the batch size
BATCH_SIZE = 32

train_dataloader = DataLoader(train_data, batch_size= BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle=False)

print(f"Dataloaders : {train_dataloader, test_dataloader}")
print(f"Length of train dataloader: {len(train_dataloader)}, batch size: {BATCH_SIZE}")

train_features_batch, train_labels_batch = next(iter(train_dataloader))
print(f"{train_features_batch.shape, train_labels_batch.shape}")


# create the model
flatten_model = nn.Flatten()

x = train_features_batch[0]

#Flatten the sample
output = flatten_model(x)

print(f"output shape: {output.shape}")

class fashion_minst_model_v0(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape, out_features=hidden_units),
            nn.Linear(in_features = hidden_units, out_features=output_shape)
        )
    def forward(self, x):
        return self.layer_stack(x)

class fashion_minst_model_v1(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features = hidden_units, out_features=output_shape),
            nn.ReLU()
        )
    def forward(self, x):
        return self.layer_stack(x)

class fashion_minst_model_v2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = input_shape,
                      out_channels = hidden_units,
                      kernel_size =3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    def forward(self, x:torch.Tensor):
        x= self.block1(x)
      #  print(x.shape)
        x= self.block2(x)
      #  print(x.shape)
        x= self.classifier(x)
        return x

torch.manual_seed(42)
device="mps"
# model_2 = fashion_minst_model_v2(input_shape=1, 
#     hidden_units=10, 
#     output_shape=len(class_names)).to(device)
# print(model_2)

# check for the device
def get_device():
    """
    Get the device to be used for training.
    
    Returns:
        torch.device: The device to be used (CPU or GPU).
    """
    # check for GPU, or mac's accelerator
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def print_time(start:float, end:float, device:torch.device=None):
    total_time = end-start
    response= {"Training time": total_time}
    return response

def eval_model(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module, accuracy_f, device):
    
    """Returns a dictionary containing the results of model predicting on data_loader.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """
    loss, acc=0.0, 0.0

    model.eval()
    with torch.inference_mode():
       for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            # forward pass
            test_pred = model(x)

            # accumulate scalar loss
            loss += loss_fn(test_pred, y).item()

            # accumulate accuracy (assuming accuracy_f returns a float or int)
            acc += accuracy_f(y_true=y, y_pred=test_pred.argmax(dim=1))

    #average the test_loss
    loss /=len(data_loader)

    acc /= len(data_loader)

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss,
            "model_acc": acc}
       
#creating a training loop and training a model on batches of data

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = "mps"):
    
    train_loss, train_acc=0.0,0.0
    model.to(device)
    for batch, (x, y) in enumerate(data_loader):
        #send data to GPU
        x,y = x.to(device), y.to(device)

        #forward pass
        # 1. forward pass
        y_pred = model(x)

        # calculate loss
        loss = loss_fn(y_pred, y)
        train_loss +=loss.item()
        train_acc += accuracy_fn(y_true =y, y_pred=y_pred.argmax(dim=1))

        #optimizer zero grad
        optimizer.zero_grad()

        #backward loss
        loss.backward()

        #optimizer step
        optimizer.step()
    
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = "mps"):
    test_loss , test_acc = 0, 0

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for batch, (x, y) in enumerate(data_loader):

            #send to device
            x, y = x.to(device), y.to(device)

            #forward pass
            y_pred = model(x)

            #calculate loss
            test_loss += loss_fn(y_pred, y).item()
            test_acc += accuracy_fn(y_true = y, y_pred= y_pred.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

def make_predictions(model: torch.nn.Module, data:list, device:torch.device= "mps"):

    pred_probs = []

    model.eval()

    with torch.inference_mode():
        
        for sample in data:
            #prepare sample
            sample = sample.to(device)
            sample = torch.unsqueeze(sample, dim=0)

            #forward pass 
            pred_logit = model(sample)

            #print(f"shape of pred logit : {pred_logit.shape}")

            #get prediction probability (logit -> prediction probability)
            pred_prob= torch.softmax(pred_logit.squeeze(), dim=0)

            #get pred_prob  off GPU for further calculations
            pred_probs.append(pred_prob.cpu())

    #stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)


    





