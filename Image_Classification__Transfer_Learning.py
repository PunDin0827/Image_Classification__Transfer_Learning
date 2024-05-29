import wget
import zipfile
import torch
import torchvision
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from torch import nn
from tqdm.auto import tqdm


# ���J�v���ø����Y zip �ɮ�
import requests
url = "https://firebasestorage.googleapis.com/v0/b/grandmacan-2dae4.appspot.com/o/ML_data%2Fone_piece_full.zip?alt=media&token=937656fd-f5c1-44f5-b174-1e2d590b8ef3"
with open("one_piece_full.zip", "wb") as f:
  req = requests.get(url)
  f.write(req.content)
with zipfile.ZipFile("one_piece_full.zip", "r") as zip_file:
  zip_file.extractall("one_piece_full")


# �N�ɮץ]�˦� Dataset
class ImageDataset(Dataset):
  def __init__(self, root, train, transform=None):

    if train:
      image_root = Path(root) / "train"
    else:
      image_root = Path(root) / "test"  # �P�_Ū�����O�V�m����ƩάO���ն����

    with open(Path(root) / "classnames.txt", "r") as f:
      lines = f.readlines()
      self.classes = [line.strip() for line in lines]  # ���o���O�ƥH�Υh��\n

    self.paths = [i for i in image_root.rglob("*") if i.is_file()]  # ���o�ɮ׸��|
    self.transform = transform

  def __getitem__(self, index):
    img = Image.open(self.paths[index]).convert("RGB")   # Ū���v��
    class_name = self.paths[index].parent.name
    class_idx = self.classes.index(class_name)  # ���o���O����

    if self.transform:
      return self.transform(img), class_idx
    else:
      return img, class_idx  # �إ߼v���ഫ�ݩ�


  def __len__(self):
    return len(self.paths)  # ���o����


# �w�q�ǽT�v�p����
def accuracy_fn(y_pred, y_true):
  correct_num = (y_pred==y_true).sum()
  acc = correct_num / len(y_true) * 100
  return acc


# �V�m�B�J
def train_step(dataloader, model, cost_fn, optimizer, accuracy_fn):
  train_cost = 0
  train_acc = 0
  for batch, (x, y) in enumerate(dataloader):
  

    model.train()  # �]�m�ҫ����V�m�Ҧ�

    y_pred = model(x)  

    cost = cost_fn(y_pred, y)   # �p��l��

    train_cost += cost
    train_acc += accuracy_fn(y_pred.argmax(dim=1), y)  # �p��ǽT�v

    optimizer.zero_grad()  # ����k�s

    cost.backward()  # �p����

    optimizer.step()  # ��s�Ѽ�

  train_cost /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  print(f"\nTrain Cost: {train_cost:.4f}, Train Acc: {train_acc:.2f}")

# ���ըB�J
def test_step(dataloader, model, cost_fn, accuracy_fn):
  test_cost = 0
  test_acc = 0
  model.eval() # �]�m�ҫ��������Ҧ�
  with torch.inference_mode():
    for x, y in dataloader:
     

      test_pred = model(x)

      test_cost += cost_fn(test_pred, y)  # �p��l��
      test_acc += accuracy_fn(test_pred.argmax(dim=1), y)  # �p��ǽT�v

    test_cost /= len(test_dataloader)
    test_acc /= len(test_dataloader)

  print(f"Test Cost: {test_cost:.4f}, Test Acc: {test_acc:.2f} \n")
  
 
# ���J�ҫ��M�v��
weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT  # ���J�ҫ��v��
model_2 = torchvision.models.efficientnet_b1(weights=weights)  # �M���v��


# ���o��ҫ��V�m�e������ഫ�ʧ@
efficientnet_b1_transforms = weights.transforms()


# �إ߰V�m�M���� Dataset
train_dataset = ImageDataset(root="one_piece_full",
              train=True,
              transform=efficientnet_b1_transforms
)

test_dataset = ImageDataset(root="one_piece_full",
              train=False,
              transform=efficientnet_b1_transforms
)


# �إ� Dataloader
BATCH_SIZE = 16

train_dataloader = DataLoader(dataset=train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
)

test_dataloader = DataLoader(dataset=test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
)
len(train_dataloader), len(test_dataloader)


# �[��ҫ���T
summary(model=model_2,
    input_size=(16, 3, 64, 64),
    col_names=["input_size", "output_size", "num_params", "trainable"],  
    row_settings=["var_names"]
)


# �N��X�令18�����O
model_2.classifier = nn.Linear(in_features=1280, out_features=18, bias=True)


# �����Ѽưl�ܱ��,���v������s
for param in model_2.features.parameters():
  param.requires_grad=False


# �w�q�l����Ƥγ̨Τƨ��
cost_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_2.parameters(), lr=0.001)

# �}�l�V�m
epochs = 10

for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n-------")

  train_step(train_dataloader, model_2, cost_fn, optimizer, accuracy_fn)

  test_step(test_dataloader, model_2, cost_fn, accuracy_fn)