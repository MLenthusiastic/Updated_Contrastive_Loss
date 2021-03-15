import json
import random
import argparse
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch.optim as optim
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorboard_utils import CustomSummaryWriter

parser = argparse.ArgumentParser(description='Simple Contrastive Loss Davis')
parser.add_argument('--data_expansion_factor', type=float, default=1.0)
parser.add_argument('--constractive_loss_margin', type=float, default=0.2)
parser.add_argument('--test_classes', nargs='+', type=str, default='car ' 'person ' 'motorcycle')
parser.add_argument('--model', type=str, default='shufflenet')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--embedding_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--dist_metric', type=str, default='euclidean')
parser.add_argument('--num_epochs', type=int, default=2)
parser.add_argument('--resize', type=int, default=512)
parser.add_argument('--train_label_count', type=float, default=0.7)
parser.add_argument('--train_data_ratio', type=float, default=0.8)
parser.add_argument('--min_objects_per_class', type=int, default=20)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--no_of_samples_per_class', type=int, default=200)
parser.add_argument('--no_of_classes', type=int, default=50)
parser.add_argument('--projector_img_size', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('-run_name', default=f'run_{time.time()}', type=str)
parser.add_argument('-sequence_name', default=f'seq_default', type=str)
args, unknown = parser.parse_known_args()

print(args.run_name, args.sequence_name)

DEVICE = args.device
if not torch.cuda.is_available():
    DEVICE = 'cpu'

base_path = 'Davis_2017_Full_1080P'
json_path = 'all_data_fp100.json'
all_imgs_path = './Padded_Images_1000/'

all_classes_with_count = {}
class_info = {}

final_classes_with_count = {}
class_with_object_img_files = {}

train_data = {}
eval_data = {}
test_data = {}

class SiameseDavis(Dataset):

    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.data_labels = list(self.data_dict.keys())
        self.data_transforms = transforms.Compose([
            transforms.Resize(args.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.pair_item_count = {}

        positive_pairs = []  # 0 same
        negative_pairs = []  # 1 different

        ## datastructure [ img1, img2, target, label1, label2 ]
        for label in self.data_labels:
            data_for_label = self.data_dict.get(label)
            for i in range(int(len(data_for_label) * args.data_expansion_factor)):
                pairs = random.sample(data_for_label, 2)
                self.add_to_pair_item_count([label])
                final_pairs = [pairs[0], pairs[1], 0, label, label]
                positive_pairs.append(final_pairs)

        ## datastructure [ img1, img2, target, label1, label2 ]
        for idx in range(len(positive_pairs)):
            random_labels = random.sample(self.data_labels, 2)
            data_for_label = self.data_dict.get(random_labels[0])
            data_for_non_label = self.data_dict.get(random_labels[1])
            pair_0 = random.choice(data_for_label)
            pair_1 = random.choice(data_for_non_label)
            self.add_to_pair_item_count([random_labels[0], random_labels[1]])
            pairs_neg = [pair_0, pair_1, 1, random_labels[0], random_labels[1]]
            negative_pairs.append(pairs_neg)

        self.data_pairs = positive_pairs + negative_pairs

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, index):
        item = self.data_pairs[index]
        img1_t = self.data_transforms(Image.open(all_imgs_path+item[0]))
        img2_t = self.data_transforms(Image.open(all_imgs_path+item[1]))
        target = item[2]
        label1 = item[3]
        label2 = item[4]
        return img1_t, img2_t, target, label1, label2

    def add_to_pair_item_count(self, item_list):
     count = 0
     if len(item_list) == 1:
         count = 2
     if len(item_list) == 2:
         count = 1
     for item in item_list:
        if item not in list(self.pair_item_count.keys()):
            self.pair_item_count[item] = count
        else:
            current_item_count = self.pair_item_count.get(item)
            self.pair_item_count[item] = current_item_count+count

    def get_pairs_item_count(self):
        return self.pair_item_count

    def get_total_items_count(self):
        total_count = 0
        for label in list(self.data_dict.keys()):
            total_count += len(self.data_dict.get(label))
        return total_count



class SimaseNet(nn.Module):
    def __init__(self):
        super(SimaseNet, self).__init__()

        if str(args.model) == 'resnet50':
            self.model = torchvision.models.resnet50(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, args.embedding_size)
        elif str(args.model) == 'resnet101':
            self.model = torchvision.models.resnet101(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, args.embedding_size)
        elif str(args.model) == 'densenet121':
            self.model = torchvision.models.densenet121(pretrained=True)
            self.model.classifier = nn.Linear(1024, args.embedding_size)
        elif str(args.model) == 'vgg16':
            self.model = torchvision.models.vgg16(pretrained=True)
            self.model.classifier[6] = nn.Linear(4096, args.embedding_size)
        elif str(args.model) == 'inception_v3':
            self.model = torchvision.models.inception_v3(pretrained=True)
            self.model.AuxLogits.fc = nn.Linear(768, args.embedding_size)
            self.model.fc = nn.Linear(2048, args.embedding_size)
        elif str(args.model) == 'googlenet':
            self.model = torchvision.models.googlenet(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, args.embedding_size)
        elif str(args.model) == 'shufflenet':
            self.model = torchvision.models.shufflenet_v2_x0_5(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, args.embedding_size)
        elif str(args.model) == 'sqeezenet':
            self.model = torchvision.models.squeezenet1_1(pretrained=True)
            self.model.classifier[1] = nn.Conv2d(512, args.embedding_size, kernel_size=(1,1), stride=(1,1))

    def forward(self, x):
        out_x = self.model.forward(x)
        if args.model == 'inception_v3':
            l2_length = torch.norm(out_x[0], p=2, dim=1, keepdim=True)
            z = out_x[0] / l2_length
        else:
            l2_length = torch.norm(out_x.detach(), p=2, dim=1, keepdim=True)
            z = out_x / l2_length
        return z


class ContrastiveLoss(nn.Module):

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, target):
        if args.dist_metric == 'euclidean':
            distance = F.pairwise_distance(output[0], output[1])
        if args.dist_metric == 'cosine':
            distance = F.cosine_similarity(output[0], output[1])

        loss = 0.5 * (1 - target.float()) * torch.pow(distance, 2) + \
               0.5 * target.float() * torch.pow(torch.clamp(self.margin - distance, min=0.00), 2)

        return loss.mean()

def transform_image_for_projector(img_tensor):
    x_np = img_tensor.to('cpu').data.numpy()
    x_np = x_np.swapaxes(0, 1)
    x_np = x_np.swapaxes(1, 2)
    img = cv2.resize(x_np, (args.projector_img_size, args.projector_img_size))
    img_tp = np.transpose(img, (2, 1, 0))
    return img_tp

def add_to_confusion_matrix(keys_ascending, conf_matrix):
    fig = plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.xticks([i for i in range(len(keys_ascending))], keys_ascending)
    plt.yticks([i for i in range(len(keys_ascending))], keys_ascending)

    for x in range(len(keys_ascending)):
        for y in range(len(keys_ascending)):
            plt.annotate(
                str(conf_matrix[x, y]), xy=(x, y),
                horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='white'
            )
    plt.xlabel('True')
    plt.ylabel('Predicted')
    #plt.show()

    return fig

def add_to_confusion_matrix_acc(keys_ascending, conf_matrix):
    fig = plt.figure()
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.xticks([i for i in range(len(keys_ascending))], keys_ascending)
    plt.yticks([i for i in range(len(keys_ascending))], keys_ascending)

    for x in range(len(keys_ascending)):
        for y in range(len(keys_ascending)):
            plt.annotate(
                str(round(100 * conf_matrix[x, y] / np.sum(conf_matrix[x]), 1)), xy=(x, y),
                horizontalalignment='center',
                verticalalignment='center',
                backgroundcolor='white'
            )
    plt.xlabel('True')
    plt.ylabel('Predicted')
    #plt.show()

    return fig


def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def precision_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_precisions = 0
    for label in range(rows):
        sum_of_precisions += precision(label, confusion_matrix)
    return sum_of_precisions / rows


def recall_macro_average(confusion_matrix):
    rows, columns = confusion_matrix.shape
    sum_of_recalls = 0
    for label in range(columns):
        sum_of_recalls += recall(label, confusion_matrix)
    return sum_of_recalls / columns


def f1(confusion_matrix):
    precision = precision_macro_average(confusion_matrix)
    recall = recall_macro_average(confusion_matrix)
    f1 = 2*(precision*recall)/(precision+recall)
    return f1


with open(json_path) as json_file:
    data = json.load(json_file)
    all_classes_with_count = data['no_objects_per_class']
    class_info = data['class_info']

#Filtering Classes which has lower than min_objects_per_class
for class_ in list(all_classes_with_count.keys()):
    if all_classes_with_count.get(class_) > args.min_objects_per_class:
        final_classes_with_count[class_] = all_classes_with_count.get(class_)

def get_all_count_in_dataset():
    count_all = 0
    for key in list(final_classes_with_count.keys()):
        count_all += final_classes_with_count.get(key)
    return  count_all


#Adding object image files per class
for class_ in list(final_classes_with_count.keys()):
    class_context_ = class_info.get(class_)
    keys_class_context = list(class_context_.keys())
    object_img_per_class = []
    for key_class_context in keys_class_context:
        object_img_per_class.extend(class_context_.get(key_class_context))
    class_with_object_img_files[class_] = object_img_per_class

#make split by classes
test_labels = args.test_classes.split(' ')
all_classes = list(final_classes_with_count.keys())
train_labels = list(set(all_classes) - set(test_labels))

print('Test Classes', test_labels)
print('Train Classes', train_labels)

#Create Datasets
for set_ in [train_labels, test_labels]:
    if set_ == train_labels:
        for class_ in set_:
            object_img_list = class_with_object_img_files.get(class_)
            random.shuffle(object_img_list)
            img_count = len(object_img_list)
            train_set_count = int(img_count*args.train_data_ratio)
            train_data[class_] = object_img_list[0:train_set_count]
            eval_data[class_] = list(set(object_img_list) - set(train_data[class_]))
    else:
        for class_ in set_:
            object_img_list = class_with_object_img_files.get(class_)
            test_data[class_] = object_img_list

train_dataset = SiameseDavis(train_data)
eval_dataset = SiameseDavis(eval_data)
test_dataset = SiameseDavis(test_data)

print("train dataset", train_dataset.get_pairs_item_count(), train_dataset.get_total_items_count(), train_dataset.__len__())
print("eval dataset", eval_dataset.get_pairs_item_count(), eval_dataset.get_total_items_count(), eval_dataset.__len__())
print("test dataset", test_dataset.get_pairs_item_count(), test_dataset.get_total_items_count(), test_dataset.__len__())

train_pairs_item_dict = train_dataset.get_pairs_item_count()
train_total_items_count = train_dataset.get_total_items_count()
eval_pairs_item_dict = eval_dataset.get_pairs_item_count()
eval_total_items_count = eval_dataset.get_total_items_count()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

#tb_writer = tensorboardX.SummaryWriter()
tb_writer = CustomSummaryWriter(
    logdir=f'{args.sequence_name}/{args.run_name}'
)

model = SimaseNet()

criterion = ContrastiveLoss(margin=args.constractive_loss_margin)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
model = model.to(DEVICE)

train_losses = []
eval_losses = []
epochs = []

train_accuracy = []
test_accuracy = []
best_train_accuracy = 0.0
train_precision = 0.00
train_recall = 0.00
train_f1 = 0.00

best_test_accuracy = 0.0
test_precision = 0.00
test_recall = 0.00
test_f1 = 0.00

train_class_count = len(train_labels)
test_class_count = len(test_labels)

def get_adjusted_loss_value(loss, label_1, label_2, is_train):
    coef_value = 0
    item_dict = {}
    item_total = 0
    eps = 1e-8

    if is_train:
        item_dict = train_pairs_item_dict
        item_total = train_total_items_count
    else:
        item_dict = eval_pairs_item_dict
        item_total = eval_total_items_count

    #coefs_for_each_pairs = 1 - ((count_pair_item_A + count_pair_item_B) / total_items_in_dataset) + eps
    #updated_loss = loss*(1-coef_value+eps)

    for class_1, class_2 in zip(label_1, label_2):
        count_class_1 = item_dict.get(class_1)
        count_class_2 = item_dict.get(class_2)
        total_items = item_total
        coef_value += (count_class_1 + count_class_2) / total_items

    updated_loss = loss * (1 - coef_value + eps)
    return updated_loss


for epoch in range(1, args.num_epochs + 1):

    epochs.append(epoch)

    ## Training
    for dataloader in [train_loader, eval_loader]:

        start_time = time.time()
        losses = []

        classes_dict = {}
        projector_labels = []
        projector_imgs = []
        projector_embeddings = []

        model.train()
        torch.set_grad_enabled(True)

        out1_t = torch.Tensor()
        out2_t = torch.Tensor()
        target_arr = np.array([])

        for batch in dataloader:

            img_1_t, img_2_t, target, label_1, label_2 = batch

            img_1_t = img_1_t.to(DEVICE)
            img_2_t = img_2_t.to(DEVICE)
            target = target.to(DEVICE)
            out_1 = model(img_1_t)
            out_2 = model(img_2_t)
            out = [out_1, out_2]
            loss = criterion(out, target)

            if dataloader == train_loader:

                #adjust loss based on the number classes
                loss_up = get_adjusted_loss_value(loss, label_1, label_2, is_train=True)
                optimizer.zero_grad()
                loss_up.backward()
                optimizer.step()

            if dataloader == eval_loader:
                # adjust loss based on the number classes
                loss_up = get_adjusted_loss_value(loss, label_1, label_2, is_train=False)

            losses.append(loss_up.item())

            ##Adding Tensorboard Projector
            labels_total = [label_1, label_2]
            imgs_total = [img_1_t, img_2_t]
            outs_total = [out_1, out_2]
            for label_batch, img_batch, out_batch in zip(labels_total, imgs_total, outs_total): # 2,4, 4
                for label, img, out in zip(label_batch, img_batch, out_batch):
                    if label not in list(classes_dict.keys()):
                        classes_dict[label] = 1
                        projector_labels.append(label)
                        projector_embeddings.append(out.detach().cpu())
                        projector_imgs.append(transform_image_for_projector(img.detach().cpu()))
                    else:
                        current_count = classes_dict.get(label)
                        if current_count < args.no_of_samples_per_class:
                            classes_dict[label] = current_count + 1
                            projector_labels.append(label)
                            projector_embeddings.append(out.detach().cpu())
                            projector_imgs.append(transform_image_for_projector(img.detach().cpu()))

        if dataloader == train_loader:
            train_losses.append(np.mean(losses))
            train_time_end = time.time()
            print('epoch', epoch, 'train_loss', np.mean(losses), ' elapsed time',
                      (train_time_end - start_time), ' seconds')
            tb_writer.add_scalars(tag_scalar_dict={'Train': np.mean(losses)}, global_step=epoch,
                                      main_tag='Loss')

            tb_writer.add_embedding(
                mat=torch.FloatTensor(np.stack(projector_embeddings)),
                label_img=torch.FloatTensor(np.stack(projector_imgs)),
                metadata=projector_labels,
                global_step=epoch, tag=f'train_emb_{epoch}')
            tb_writer.flush()

        if dataloader == eval_loader:
            eval_losses.append(np.mean(losses))
            eval_time_end = time.time()
            print('epoch', epoch, 'eval_loss', np.mean(losses), ' elapsed time',
                      (eval_time_end - start_time), ' seconds')
            tb_writer.add_scalars(tag_scalar_dict={'Eval': np.mean(losses)}, global_step=epoch,
                                      main_tag='Loss')

            tb_writer.add_embedding(
                mat=torch.FloatTensor(np.stack(projector_embeddings)),
                label_img=torch.FloatTensor(np.stack(projector_imgs)),
                metadata=projector_labels,
                global_step=epoch, tag=f'eval_emb_{epoch}')
            tb_writer.flush()

    torch.save(model, 'cm_up_shuffle_net_b'+str(args.run_name)+'.pt')
    model = torch.load('cm_up_shuffle_net_b'+str(args.run_name)+'.pt')

    model.eval()
    torch.set_grad_enabled(False)

    t_start_time = time.time()

    #Testing
    for dataloader in [train_loader, test_loader]:

        embbedding_dict = {}
        embbedding_center_dict = {}
        batch_accuracy_list = []

        classes_dict = {}
        projector_labels = []
        projector_imgs = []
        projector_embeddings = []

        if dataloader == train_loader:
            conf_matrix = np.zeros((train_class_count, train_class_count))
            labels_ = train_labels
            labels_.sort()
        else:
            conf_matrix = np.zeros((test_class_count, test_class_count))
            labels_ = test_labels
            labels_.sort()

        count_dict = {}
        if dataloader == test_loader:
            for batch in dataloader:
                img1, img2, target, label1, label2 = batch
                for l1, l2 in zip(label1,label2):
                    if l1 not in list(count_dict.keys()):
                        count_dict[l1]= 1
                    elif l1 in list(count_dict.keys()):
                        count_dict[l1] = count_dict.get(l1)+1

                    if l2 not in list(count_dict.keys()):
                        count_dict[l2]= 1
                    elif l2 in list(count_dict.keys()):
                        count_dict[l2] = count_dict.get(l2)+1

        ##center calculation
        for batch in dataloader:
            img1, img2, target, label1, label2 = batch

            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)

            out1 = model(img1)
            out2 = model(img2)

            ##collecting all embedding for each label
            for index, (label1_np, label2_np) in enumerate(zip(label1, label2)):
                if label1_np in embbedding_dict.keys():
                    existing_emb_for_label = embbedding_dict.get(label1_np)
                    existing_emb_for_label.append(out1[index].data[:])
                else:
                    embbedding_dict[label1_np] = [out1[index].data[:]]
                if label2_np in embbedding_dict.keys():
                    existing_emb_for_label = embbedding_dict.get(label2_np)
                    existing_emb_for_label.append(out2[index].data[:])
                else:
                    embbedding_dict[label2_np] = [out2[index].data[:]]

        ##calculate center for each label
        for label in embbedding_dict.keys():
            embeddings = embbedding_dict.get(label)
            embbedding_center_dict[label] = torch.mean(torch.stack(embeddings), dim=0)

        ##End of center calculation for whole dataset
        keys_ascending = list(embbedding_center_dict.keys())
        keys_ascending.sort()

        all_centers_ascending = [embbedding_center_dict[label] for label in keys_ascending]

        ##Calculation of accuracy
        for batch in dataloader:
            img1, img2, target, label1, label2 = batch

            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)

            out1 = model(img1)
            out2 = model(img2)
            batch_accuracy = 0
            for single_out1, single_out2, single_target, label_1, label_2 in zip(out1, out2, target, label1, label2):
                dists_single_out1 = torch.pairwise_distance(torch.stack(all_centers_ascending),
                                                                torch.stack(len(all_centers_ascending) * [single_out1]),
                                                                p=2)
                closest_idx_1 = torch.argmin(dists_single_out1)
                actual_1 = labels_.index(label_1)
                predict_1 = labels_.index(keys_ascending[closest_idx_1])
                conf_matrix[actual_1, predict_1] += 1

                dists_single_out2 = torch.pairwise_distance(torch.stack(all_centers_ascending),
                                                                torch.stack(len(all_centers_ascending) * [single_out2]),
                                                                p=2)
                closest_idx_2 = torch.argmin(dists_single_out2)
                actual_2 = labels_.index(label_2)
                predict_2 = labels_.index(keys_ascending[closest_idx_2])
                conf_matrix[actual_2, predict_2] += 1

                if torch.eq(closest_idx_1, closest_idx_2) and single_target.item() == 0:
                    batch_accuracy += 1
                elif not torch.eq(closest_idx_1, closest_idx_2) and single_target.item() == 1:
                    batch_accuracy += 1

            batch_accuracy_list.append(batch_accuracy / args.batch_size)

            if dataloader == test_loader:
                ##Adding Tensorboard Projector
                labels_total = [label1, label2]
                imgs_total = [img1, img2]
                outs_total = [out1, out2]

                for label_batch, img_batch, out_batch in zip(labels_total, imgs_total, outs_total):
                    for label, img, out in zip(label_batch, img_batch, out_batch):
                        if label not in list(classes_dict.keys()):
                            classes_dict[label] = 1
                            projector_labels.append(label)
                            projector_embeddings.append(out.detach().cpu())
                            projector_imgs.append(transform_image_for_projector(img.detach().cpu()))
                        else:
                            current_count = classes_dict.get(label)
                            if current_count < args.no_of_samples_per_class:
                                classes_dict[label] = current_count + 1
                                projector_labels.append(label)
                                projector_embeddings.append(out.detach().cpu())
                                projector_imgs.append(transform_image_for_projector(img.detach().cpu()))

        # epoch accuracy
        if dataloader == train_loader:
            test_end_time = time.time()
            print('train epoch acc', epoch, np.mean(batch_accuracy_list) * 100, ' elapsed time',
                                  (test_end_time - t_start_time), ' seconds')
            train_accuracy.append(np.mean(batch_accuracy_list) * 100)
            tb_writer.add_scalars(tag_scalar_dict={'Train': (np.mean(batch_accuracy_list) * 100)},
                                                  global_step=epoch,
                                                  main_tag='Accuracy')
            best_train_accuracy = np.mean(batch_accuracy_list) * 100

            # Confusion Matrix
            #fig = add_to_confusion_matrix(labels_, conf_matrix)
            fig = add_to_confusion_matrix_acc(labels_, conf_matrix)
            tb_writer.add_figure(
                tag='train_conf_matrix',
                figure=fig,
                global_step=epoch
            )

            train_precision = precision_macro_average(conf_matrix)
            train_recall = recall_macro_average(conf_matrix)
            train_f1 = f1(conf_matrix)

            # Metrics
            tb_writer.add_scalars(tag_scalar_dict={'Train': train_precision},
                                  global_step=epoch,
                                  main_tag='Precision')
            tb_writer.add_scalars(tag_scalar_dict={'Train': train_recall},
                                  global_step=epoch,
                                  main_tag='Recall')
            tb_writer.add_scalars(tag_scalar_dict={'Train': train_f1},
                                  global_step=epoch,
                                  main_tag='F1_Score')


        if dataloader == test_loader:
            test_end_time = time.time()
            print('test epoch acc', epoch, np.mean(batch_accuracy_list) * 100, ' elapsed time',
                                  (test_end_time - t_start_time), ' seconds')
            test_accuracy.append(np.mean(batch_accuracy_list) * 100)
            tb_writer.add_scalars(tag_scalar_dict={'Test': (np.mean(batch_accuracy_list) * 100)},
                                                  global_step=epoch,
                                                  main_tag='Accuracy')
            best_test_accuracy = np.mean(batch_accuracy_list) * 100

            # Confusion Matrix
            fig = add_to_confusion_matrix_acc(labels_, conf_matrix)
            tb_writer.add_figure(
                tag='test_conf_matrix',
                figure=fig,
                global_step=epoch
            )

            test_precision = precision_macro_average(conf_matrix)
            test_recall = recall_macro_average(conf_matrix)
            test_f1 = f1(conf_matrix)
            #Metric
            tb_writer.add_scalars(tag_scalar_dict={'Test': test_precision},
                                  global_step=epoch,
                                  main_tag='Precision')
            tb_writer.add_scalars(tag_scalar_dict={'Test': test_recall},
                                  global_step=epoch,
                                  main_tag='Recall')
            tb_writer.add_scalars(tag_scalar_dict={'Test': test_f1},
                                  global_step=epoch,
                                  main_tag='F1_Score')

            tb_writer.add_embedding(
                mat=torch.FloatTensor(np.stack(projector_embeddings)),
                label_img=torch.FloatTensor(np.stack(projector_imgs)),
                metadata=projector_labels,
                global_step=epoch, tag=f'test_emb_{epoch}')
            tb_writer.flush()

tb_writer.add_hparams(hparam_dict={'lr': args.learning_rate,
                                   'batch_size': args.batch_size,
                                   'cons_margin': args.constractive_loss_margin,
                                   'embedding_size': args.embedding_size,
                                   'model': args.model,
                                   'weight_file': args.run_name},
                       metric_dict={'train_accuracy': best_train_accuracy,
                                    'test_accuracy': best_test_accuracy,
                                    'train_precision':train_precision,
                                    'test_precision':test_precision,
                                    'train_recall':train_recall,
                                    'test_recall':test_recall,
                                    'train_f1':train_f1,
                                    'test_f1':test_f1},
                       name=args.run_name
                      )
tb_writer.flush()




