import torch
import torch.nn as nn
import random
from vision_model import *
from vision_train import models, MODEL_SAVE_PATH
from tqdm.auto import tqdm

# models =[
#     fashion_minst_model_v0(input_shape=784, hidden_units=10, output_shape=len(class_names)),
#     fashion_minst_model_v1(input_shape=784, hidden_units=10, output_shape=len(class_names)),
#     fashion_minst_model_v2(input_shape=1, hidden_units=10, output_shape=len(class_names))
# ]

def inference():
    test_samples =[]
    test_labels =[]

    for sample, label in random.sample(list(test_data), k=9):
        test_samples.append(sample)
        test_labels.append(label)

    print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")

    m = models[2]
    m.load_state_dict(torch.load(f=MODEL_SAVE_PATH[2]))
    m = m.to(device)
    m.eval()


    pred_prob = make_predictions(m, data=test_samples)
    print(pred_prob[:2])

    pred_class = pred_prob.argmax(dim=1)
    print(pred_class)
    print(test_labels)

    # Plot predictions
    # plt.figure(figsize=(9,9))
    # nrows =3
    # ncols =3

    # for i, sample in enumerate(test_samples):
        
    #     #create a subplot
    #     plt.subplot(nrows, ncols, i+1)

    #     #plot the target image
    #     plt.imshow(sample.squeeze(), cmap="grey")

    #     #find the predictions label
    #     pred_label = class_names[pred_class[i]]

    #     #get the truth label
    #     truth_label = class_names[test_labels[i]]

    #     #create the title of the text of the plot
    #     title_text = f"Pred{pred_label} | Truth:{truth_label}"

    #     #check the equality and change  title color accordingly
    #     if pred_label == truth_label:
    #         plt.title(title_text, fontsize=10, c='g')
    #     else:
    #         plt.title(title_text, fontsize =10, c='r')
        
    #     plt.axis(False)

    # plt.show()

    y_preds=[]
    #models[2].eval()

    with torch.inference_mode():
        for x, y in tqdm(test_dataloader, desc="making predictions"):

            x, y = x.to(device), y.to(device)
            #do the forward pass
            y_logit = m(x)

            #turn predictions for logits-> prediction probabilities -> prediction labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

            #put predictions on cpu for evaluation
            y_preds.append(y_pred.cpu())

    #Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)


    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend verison should be 0.19.0 or higher"

    from torchmetrics import ConfusionMatrix
    from mlxtend.plotting import plot_confusion_matrix

    # 2. Setup confusion matrix instance and compare predictions to targets
    confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
    confmat_tensor = confmat(preds=y_pred_tensor,
                            target=test_data.targets)

    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
        class_names=class_names, # turn the row and column labels into class names
        figsize=(10, 7)
    )

    plt.show()


    # test whether the truth and the pred values are different
    for sample, label in random.sample(list(test_data), k=100):
        test_samples.append(sample)
        test_labels.append(label)

    print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")

    pred_prob = make_predictions(model=m, data=test_samples)
    print(pred_prob[:2])

    pred_class = pred_prob.argmax(dim=1)
    # print(pred_class)
    # print(test_labels)

    count=0
    for p_l, c_l in zip(pred_class, test_labels):
        if p_l != c_l:
            print(f"predicted labels:{class_names[p_l]} different from test label {class_names[c_l]}")
            count+=1
        
    print(f"{count}% predictons are wrong")


if __name__ == "__main__":
    inference()