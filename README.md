# filter-feature-selection
## 程式簡介
### 簡述
* 使用 Sklearn 套件實作 Feature Selection 中的 **Filter methods【過濾器法】**

## Feature Selection
* 又稱為 variable selection、attribution selection 或 subset selection

* 指從資料集中選出最重要、最相關的特徵來給機器學習建立模型，大部分時候，這樣做可以增加機器學習的效能

* Feature selection 不等於 Dimensionality Reduction

### WHY
機器學習的實際應用中，特徵數量往往較多，可能存在不相關的特徵，特徵之間也可能存在相互依賴，容易導致：

* 特徵個數越多，分析特徵、訓練模型所需的時間也就越長。

* 特徵個數越多，容易引起【維度災難】
#### 維度災難
* 特徵維度超過一定界限後，分類器的效能隨著特徵維度的增加反而下降，如下圖
  ![image](https://user-images.githubusercontent.com/93152909/145701552-148a6354-f79c-4310-b047-619353903b76.png)
  > 原因往往是因為這些高維度特徵中含有「無關特徵」和「冗餘特徵」

* 無關特徵  
該特徵所提供的資訊對於當前學習任務無用，如對於「學生成績」而言，「學號」則是無關特徵。
* 冗餘特徵  
該特徵所包含的資訊能從其他特徵推演出來，如「面積」特徵，能從「長」和「寬」得出，則它是冗餘特徵。
### HOW
一般 Feature selection 的演算法分為三類：

* **Filter methods【此篇介紹】**

* Wrapper methods

* Embedding methods

## Filter methods【過濾器法】

* 使用統計變量來評估特徵或特徵子集的特性，再根據「挑選標準」，挑選符合標準的特徵

* 選擇特徵子集時，只考慮特徵的特性，不考慮將來要使用哪一種模型進行學習

* 常見的統計變量，例如：相關性、離散程度等等

* 概念圖  

  ![image](https://user-images.githubusercontent.com/93152909/146445500-85c3360f-1188-4d1d-8d2d-dd6e6be70442.png)

* 優點
  * 選出的特徵子集合可以被用在任何機器學習演算法
  
  * 不會耗費大量電腦資源

* 相關統計變量【僅列出有使用 sklearn 實作的】

  * 變異數 - Variance
  
  * 卡方檢定 - Chi-square
  
  > 除上述兩種，還有很多統計變量可用，可以上網另查
  
### 挑選標準
* sklearn 中常用的有以下 2 種：

  * 選擇前 k% 高分的特徵
    ```python
    from sklearn.feature_selection import SelectPercentile
    ```
    
  * 選擇 k 個最高分的特徵
    ```python
    from sklearn.feature_selection import SelectKBest 
    ```
* k為一超參數，過低會刪除與模型相關且有效的特徵；過高會保留過多無用特徵，須不斷進行調整

* 可使用學習曲線驗證法得出好的k值，但計算成本龐大
  
### 變異數 - Variance
```python
   from sklearn.feature_selection import VarianceThreshold
```

* 計算每個 「特徵」 的變異數
   
### 卡方檢定 - Chi-square
```python
   from sklearn.feature_selection import chi2
```

* 計算每個 「非負數特徵」 和 「標籤」 之間的卡方統計量

* 「非負數特徵」是**回歸變數**或**類別變數**皆可；但「標籤」必須是**類別變數**

* chi2 返回 卡方值 及 P值 兩個統計量

* 尋找卡方值很大且P值>=0.05 or 0.01 的特徵【表示與標籤相關的特徵】

* 卡方檢驗法不能計算負數；可是用以下兩種預處理方法變為正數
  * MinMaxScalar
  * StandardScalar
  
> 傳統計算卡方值的方式中，特徵(x) 與 標籤(y) 必須都是**類別變數**，因為需要使用頻率來構建列聯表，但 sklearn 計算卡方值的方式與傳統不同，因此也可以處理特徵(x)是**回歸變數**的情況。  
> 參考：[Feature selection_ Chi-square test, F-test and mutual information - Programmer All](https://www.programmerall.com/article/5467105157/)

### 【傳統】特徵(x)與標籤(y)關係之統計變量適用表    

![image](https://user-images.githubusercontent.com/93152909/146684361-e11cbfd4-8107-4dad-bb38-b2ba29df0d60.png)
    
## 參考
* [特徵選擇(feature selection) - IT閱讀](https://www.itread01.com/content/1547263108.html)
* [特徵選擇常用演算法綜述 - IT閱讀](https://www.itread01.com/content/1550470354.html)
* [Day28 - Feature Selection -- 1. Filter methods(過濾器法) - iT 邦幫忙__一起幫忙解決難題，拯救 IT 人的一天](https://ithelp.ithome.com.tw/articles/10245037)
* [Feature Selection 特徵選取 -Filter(ㄧ) Python sklearn 實作 _ by Jasmine＿ _ Medium](https://jasmine880809.medium.com/feature-selection-%E7%89%B9%E5%BE%B5%E9%81%B8%E5%8F%96-filter-%E3%84%A7-python-sklearn-%E5%AF%A6%E4%BD%9C-2797b941c6a9)
* [機器學習之特徵選擇（Feature Selection）_實用技巧_程式人生](https://www.796t.com/article.php?id=173751)
* [特徵工程之特徵選擇概念. 特徵選擇是特徵工程裡的一個重要問題，其目標是尋找最優特徵子集。特徵選擇能剔除不相… _ by Ryan Lu _ AI反斗城 _ Medium](https://medium.com/ai%E5%8F%8D%E6%96%97%E5%9F%8E/%E7%89%B9%E5%BE%B5%E5%B7%A5%E7%A8%8B%E4%B9%8B%E7%89%B9%E5%BE%B5%E9%81%B8%E6%93%87%E6%A6%82%E5%BF%B5-ca11745db63c)
* [機器學習-特徵工程-特徵選擇(feature_selection)-過濾法 _ Taroballz StudyNotes](http://www.taroballz.com/2019/06/12/ML_feature_selection_filter_method/)
* [](https://iter01.com/502884.html)
* [](https://towardsdatascience.com/using-the-chi-squared-test-for-feature-selection-with-implementation-b15a4dad93f1)
* [](https://towardsdatascience.com/mistakes-in-applying-univariate-feature-selection-methods-34c43ce8b93d)
* [](https://medium.com/analytics-vidhya/feature-selection-73bc12a9b39e#dd6f)

> 備份於 Reference 資料夾中
