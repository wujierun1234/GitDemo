
# 代码核心功能说明
## classify代码截图
<img src="https://github.com/wujierun1234/GitDemo/blob/043ae0e5bf3ea162dc833501b0fbf770e54037d2/classify.png" width="500" alt="代码截图">

## 1.文本预处理函数：

```python
def get_text(filename):
    """读取文本并过滤无效字符"""
    with open(filename, 'r', encoding='utf-8') as fr:
        text = fr.read()
        text = re.sub(r'[.【】0-9、——。，！~\*]', '', text)
    return text
```
#### 该函数负责读取文本文件并进行初步清洗：
#### 使用open()函数读取整个文件内容
#### 通过正则表达式re.sub()去除数字、标点等特殊字符
#### 返回清洗后的纯文本内容

## 2.分词处理函数：

```python
def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    text = get_text(filename)
    line = cut(text)
    line = filter(lambda word: len(word) > 1, line)
    words.extend(line)
    return words
```
#### 该函数该函数对文本进行精细处理：
#### 调用get_text()获取清洗后的文本
#### 使用jieba.cut()进行中文分词
#### 通过filter()过滤掉单字词
#### 返回处理后的词语列表用于构建一个包含文本数据中出现次数最多的词的词汇表：

## 3.特征提取方法
### 高频词特征提取
```python
def get_top_words(top_num=100):
    global all_words
    filename_list = [f'邮件_files/{i}.txt' for i in range(151)]
    all_words = []
    
    for filename in filename_list:
        all_words.append(get_words(filename))
    
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]
```
### TF-IDF特征提取
```python
def get_tfidf_features(top_num=100):
    filename_list = [f'邮件_files/{i}.txt' for i in range(151)]
    corpus = [get_text(filename) for filename in filename_list]
    
    def jieba_tokenizer(text):
        return [word for word in cut(text) if len(word) > 1]
    
    tfidf = TfidfVectorizer(tokenizer=jieba_tokenizer, max_features=top_num)
    tfidf_matrix = tfidf.fit_transform(corpus)
    return tfidf, tfidf.toarray()
```
#### 特征提取对比：
#### 高频词：统计词频最高的N个词
#### TF-IDF：考虑词频和逆文档频率
#### 两种方法

## 分类器实现
```python
class SpamClassifier:
    def __init__(self, feature_method='frequency', top_num=100):
        self.feature_method = feature_method
        self.top_num = top_num
        self.model = MultinomialNB()
    def train(self):
        if self.feature_method == 'frequency':
            self.top_words = get_top_words(self.top_num)
            # ...特征向量构建...
        else:
            self.tfidf_vectorizer, X = get_tfidf_features(self.top_num)
        
        y = np.array([1]*127 + [0]*24)
        self.model.fit(X, y)
    
    def predict(self, filename):
        # ...特征提取...
        result = self.model.predict(current_vector.reshape(1, -1))
        return '垃圾邮件' if result == 1 else '普通邮件'
```
#### 分类器特点：
#### 支持两种特征提取方式切换
#### 使用多项式朴素贝叶斯分类器
#### 训练数据中前127封为垃圾邮件(标记1)，后24封为普通邮件(标记0)

# 特征模式切换方法
## 1.高频词模式：
```python
# 初始化高频词分类器
freq_classifier = SpamClassifier(feature_method='frequency')
freq_classifier.train()

# 预测示例
print(freq_classifier.predict('邮件_files/151.txt'))
```
## 2.TF-IDF模式：
```python
# 初始化TF-IDF分类器
tfidf_classifier = SpamClassifier(feature_method='tfidf')
tfidf_classifier.train()

# 预测示例
print(tfidf_classifier.predict('邮件_files/151.txt'))
```
## 3.参数调整：
### 可通过top_num参数调整特征数量：
```python
# 使用200个特征词
classifier = SpamClassifier(feature_method='frequency', top_num=200)
classifier.train()
```
