from vision_data import *
from timeit import default_timer as timer   
from tqdm.auto import tqdm

#import tqdm for progress bar
torch.manual_seed(42)
train_time_start_time = timer()

#number of epochs
epochs=10

device = get_device()
print("device:", device ,"being used for training")

#model_0= fashion_minst_model_v0(input_shape=784, hidden_units=10, output_shape=len(class_names))
model_0= fashion_minst_model_v1(input_shape=784, hidden_units=10, output_shape=len(class_names))
model_0.to(device)

print(model_0)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

#create training and testing loops
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n...")

    train_step(model=model_0, data_loader=train_dataloader,loss_fn =loss_fn,
               optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)

    test_step(model=model_0, data_loader=test_dataloader,loss_fn =loss_fn,
            accuracy_fn=accuracy_fn, device=device)
       
#calcuate training time
train_time_end_time = timer()
total_train_time_model_0 = print_time(start=train_time_start_time, 
                                           end=train_time_end_time,
                                           device=str(next(model_0.parameters()).device))

print(total_train_time_model_0)

# Calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0, data_loader=test_dataloader, loss_fn=loss_fn, accuracy_f=accuracy_fn, device=device)
print(model_0_results)
