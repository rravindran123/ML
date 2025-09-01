from vision_model import *
from timeit import default_timer as timer   
from tqdm.auto import tqdm
import pandas as pd
import random

#import tqdm for progress bar
torch.manual_seed(42)
train_time_start_time = timer()

#number of epochs
epochs=10

device = get_device()
print("device:", device ,"being used for training")

# models = ['model_0', 'model_2']


# model_0= fashion_minst_model_v0(input_shape=784, hidden_units=10, output_shape=len(class_names))
# model_2= fashion_minst_model_v2(input_shape=1, hidden_units=10, output_shape=len(class_names))

models =[
    fashion_minst_model_v0(input_shape=784, hidden_units=10, output_shape=len(class_names)),
    fashion_minst_model_v1(input_shape=784, hidden_units=10, output_shape=len(class_names)),
    fashion_minst_model_v2(input_shape=1, hidden_units=10, output_shape=len(class_names))
]
results=[]
training_time=[]

for m in models:
    m.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=m.parameters(), lr=0.1)

    #create training and testing loops
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n...")

        train_step(model=m, data_loader=train_dataloader,loss_fn =loss_fn,
                optimizer=optimizer, accuracy_fn=accuracy_fn, device=device)

        test_step(model=m, data_loader=test_dataloader,loss_fn =loss_fn,
                accuracy_fn=accuracy_fn, device=device)
        
    #calcuate training time
    train_time_end_time = timer()
    total_train_time_model_0 = print_time(start=train_time_start_time, 
                                            end=train_time_end_time,
                                            device=str(next(m.parameters()).device))

    training_time.append(total_train_time_model_0)

    # Calculate model 0 results on test dataset
    results.append(eval_model(model=m, data_loader=test_dataloader, loss_fn=loss_fn, 
                              accuracy_f=accuracy_fn, device=device))
for output in results:
    print(f"{output}\n")

compare_results = pd.DataFrame(results)
training_df = pd.DataFrame(training_time)

print(compare_results)
print(training_time)

#compare_results["trainging_time"]=training_time
compare_results= pd.concat([compare_results, training_df], axis=1)
print(compare_results)

# Visualize our model results
# compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
# plt.xlabel("accuracy (%)")
# plt.ylabel("model")
# plt.show()

test_samples =[]
test_labels =[]

for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")

pred_prob = make_predictions(model=models[2], data=test_samples)
print(pred_prob[:2])

pred_class = pred_prob.argmax(dim=1)
print(pred_class)
print(test_labels)