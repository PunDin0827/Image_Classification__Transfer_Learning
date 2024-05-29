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


# 載入影像並解壓縮 zip 檔案
import requests
url = "https://firebasestorage.googleapis.com/v0/b/grandmacan-2dae4.appspot.com/o/ML_data%2Fone_piece_full.zip?alt=media&token=937656fd-f5c1-44f5-b174-1e2d590b8ef3"
with open("one_piece_full.zip", "wb") as f:
  req = requests.get(url)
  f.write(req.content)
with zipfile.ZipFile("one_piece_full.zip", "r") as zip_file:
  zip_file.extractall("one_piece_full")


# 將檔案包裝成 Dataset
class ImageDataset(Dataset):
  def __init__(self, root, train, transform=None):

    if train:
      image_root = Path(root) / "train"
    else:
      image_root = Path(root) / "test"  # 判斷讀取的是訓練集資料或是測試集資料

    with open(Path(root) / "classnames.txt", "r") as f:
      lines = f.readlines()
      self.classes = [line.strip() for line in lines]  # 取得類別數以及去除\n

    self.paths = [i for i in image_root.rglob("*") if i.is_file()]  # 取得檔案路徑
    self.transform = transform

  def __getitem__(self, index):
    img = Image.open(self.paths[index]).convert("RGB")   # 讀取影像
    class_name = self.paths[index].parent.name
    class_idx = self.classes.index(class_name)  # 取得類別索引

    if self.transform:
      return self.transform(img), class_idx
    else:
      return img, class_idx  # 建立影像轉換屬性


  def __len__(self):
    return len(self.paths)  # 取得長度


# 定義準確率計算函數
def accuracy_fn(y_pred, y_true):
  correct_num = (y_pred==y_true).sum()
  acc = correct_num / len(y_true) * 100
  return acc


# 訓練步驟
def train_step(dataloader, model, cost_fn, optimizer, accuracy_fn):
  train_cost = 0
  train_acc = 0
  for batch, (x, y) in enumerate(dataloader):
  

    model.train()  # 設置模型為訓練模式

    y_pred = model(x)  

    cost = cost_fn(y_pred, y)   # 計算損失

    train_cost += cost
    train_acc += accuracy_fn(y_pred.argmax(dim=1), y)  # 計算準確率

    optimizer.zero_grad()  # 梯度歸零

    cost.backward()  # 計算梯度

    optimizer.step()  # 更新參數

  train_cost /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  print(f"\nTrain Cost: {train_cost:.4f}, Train Acc: {train_acc:.2f}")

# 測試步驟
def test_step(dataloader, model, cost_fn, accuracy_fn):
  test_cost = 0
  test_acc = 0
  model.eval() # 設置模型為評估模式
  with torch.inference_mode():
    for x, y in dataloader:
     

      test_pred = model(x)

      test_cost += cost_fn(test_pred, y)  # 計算損失
      test_acc += accuracy_fn(test_pred.argmax(dim=1), y)  # 計算準確率

    test_cost /= len(test_dataloader)
    test_acc /= len(test_dataloader)

  print(f"Test Cost: {test_cost:.4f}, Test Acc: {test_acc:.2f} \n")
  
 
# 載入模型和權重
weights = torchvision.models.EfficientNet_B1_Weights.DEFAULT  # 載入模型權重
model_2 = torchvision.models.efficientnet_b1(weights=weights)  # 套用權重


# 取得原模型訓練前的資料轉換動作
efficientnet_b1_transforms = weights.transforms()


# 建立訓練和測試 Dataset
train_dataset = ImageDataset(root="one_piece_full",
              train=True,
              transform=efficientnet_b1_transforms
)

test_dataset = ImageDataset(root="one_piece_full",
              train=False,
              transform=efficientnet_b1_transforms
)


# 建立 Dataloader
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


# 觀察模型資訊
summary(model=model_2,
    input_size=(16, 3, 64, 64),
    col_names=["input_size", "output_size", "num_params", "trainable"],  
    row_settings=["var_names"]
)


# 將輸出改成18個類別
model_2.classifier = nn.Linear(in_features=1280, out_features=18, bias=True)


# 取消參數追蹤梯度,使權重不更新
for param in model_2.features.parameters():
  param.requires_grad=False


# 定義損失函數及最佳化函數
cost_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_2.parameters(), lr=0.001)

# 開始訓練
epochs = 10

for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n-------")

  train_step(train_dataloader, model_2, cost_fn, optimizer, accuracy_fn)

  test_step(test_dataloader, model_2, cost_fn, accuracy_fn)