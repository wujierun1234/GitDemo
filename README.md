
# 代码核心功能说明
## classify代码截图
<img src="https://github.com/wujierun1234/GitDemo/blob/043ae0e5bf3ea162dc833501b0fbf770e54037d2/classify.png" width="500" alt="代码截图">

## 1.get_words() 函数：

```python
def get_words(filename):
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            line = cut(line)
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words
all_words = []
```
#### 该函数从一个文本文件中读取文本并进行处理
#### 读取文本：通过 open() 打开文件并按行读取。
#### 去除无效字符：使用正则表达式 re.sub() 来去除常见的无效字符（如数字、标点符号等）。
#### 分词：利用 jieba.cut() 对每一行文本进行分词处理。

## 2.get_top_words() 函数：

```python
def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    # 遍历邮件建立词库
    for filename in filename_list:
        all_words.append(get_words(filename))
    # itertools.chain()把all_words内的所有列表组合成一个列表
    # collections.Counter()统计词个数
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]
top_words = get_top_words(100)
```
#### 该函数用于构建一个包含文本数据中出现次数最多的词的词汇表：
#### 读取文件：通过文件名列表 filename_list 来读取多个文件（邮件文件）。
#### 提取所有词汇：对于每个文件，调用 get_words() 来提取词汇。
#### 统计词频：通过 itertools.chain(*all_words) 将多个文件的词汇合并为一个列表，然后使用 collections.Counter 来统计各个词的频率。

## 3.优化特征选择方法
### 提取高频词特征
```python
count_features, count_feature_names = feature_extraction(documents, method='count')
print("高频词特征矩阵：")
print(count_features.toarray())
print("特征名：", count_feature_names)
```
#### 使用 CountVectorizer 提取文本的高频词特征。
#### toarray() 将稀疏矩阵转换为数组，便于查看。
#### get_feature_names_out() 返回提取的特征名称（即所有被识别的词）。

### 提取TF-IDF特征
```python
tfidf_features, tfidf_feature_names = feature_extraction(documents, method='tfidf')
print("\nTF-IDF特征矩阵：")
print(tfidf_features.toarray())
print("特征名：", tfidf_feature_names)
```
#### 使用 TfidfVectorizer 提取文本的 TF-IDF 特征。
#### 和高频词特征一样，使用 toarray() 和 get_feature_names_out() 查看特征矩阵和特征名称。
<img src="https://github.com/LZY6888/LZY1/blob/main/image/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20181002.png" width="500" alt="代码截图">

## 4.样本平衡处理
<img src="https://github.com/LZY6888/LZY1/blob/main/image/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20181123.png" width="500" alt="代码截图">

## 5.增加模型评估指标
<img src="https://github.com/LZY6888/LZY1/blob/main/image/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202025-04-08%20181137.png" width="500" alt="代码截图">
