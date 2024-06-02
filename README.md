# 使用 PyTorch 進行圖像分類   

使用PyTorch和EfficientNet_B1模型進行影像分類。  
資料集內包含測試集和訓練集，資料集中包含了18個不同類別的影像。使用訓練集來訓練模型，並使用測試集來評估模型的性能。

## 概述  
本項目包括以下步驟：  
1. 用 requests 提出下載請求，並從 zip 文件中提取圖像。  
2. 創建自定義數據集類來加載圖像及其對應的標籤。  
3. 載入預訓練模型權重。  
4. 訓練並評估模型。   
5. 報告成果。  

## 數據集  
train/：訓練集影像資料夾  
test/：測試集影像資料夾  
classnames.txt：類別名稱列表  

## 模型架構  
模型：EfficientNet_B1  
輸入影像尺寸：64x64  
輸出類別數量：18  
使用交叉熵損失函數進行訓練  
使用Adam優化器進行參數優化   

## 使用方法  
運行腳本以提取數據集並訓練模型：  

## 訓練與評估  
訓練和評估過程包括：  

1. 載入和預處理數據集。  
2. 載入預訓練的模型權重。  
3. 在訓練數據集上訓練模型。    
4. 在測試數據集上評估模型。    
5. 在腳本中調整訓練次數和其他超參數。  

## 結果
訓練後，模型會在測試集上進行性能評估。每個訓練次數都會顯示訓練和測試的準確率及損失。最終結果將顯示模型在測試數據集上的準確度。  
