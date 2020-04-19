# 爬虫Beautiful suop

### 基本元素

Beautiful Soup库是解析、遍历、维护“标签树”的功能库

引用：form bs4 import BeautifulSoup

| 解析器           | 使用方法                        | 条件                 |
| ---------------- | ------------------------------- | -------------------- |
| bs4的HTML解析器  | BeautifulSoup(mk,'html.parser') | 安装bs4库            |
| lxml的HTML解析器 | BeautifulSoup(mk,'lxml')        | pip install  ixml    |
| lxml的XML解析器  | BeautifulSoup(mk,'xml')         | pip install lxml     |
| html5lib的解析器 | BeautifulSoup(mk,'html5lib')    | pip install html5lib |

五种基本元素

| 基本元素        | 说明                                                       |
| --------------- | ---------------------------------------------------------- |
| Tag             | 标签，最基本的信息组值单元，分别用<> and </>表明开头和结尾 |
| Name            | 标签的名字，\<p>...\</p>的名字是‘p'，格式：\<tag>.name     |
| Attributes      | 标签的属性，字典形式组值，格式：\<tag>.attrs               |
| NavigableString | 标签内非属性字符串，\<>...</>中字符串，格式：\<tag>.string |
| Comment         | 标签内字符串的注释部分，一种特殊的Comment类型              |

### 标签树的下行遍历

| 属性        | 说明                                                    |
| ----------- | ------------------------------------------------------- |
| .contents   | 子节点的列表，将\<tag>所有儿子节点存入列表              |
| .children   | 子节点的迭代类型，与.contents类似，用于循环遍历儿子节点 |
| .descendant | 子孙节点的迭代类型，包含所有子孙节点，用于循环遍历      |

上行遍历

| 属性     | 说明                                         |
| -------- | -------------------------------------------- |
| .parent  | 节点的父亲标签                               |
| .parents | 节点先辈标签的迭代类型，用于循环遍历先辈节点 |

```python
for child in soup.body.children:
	print(child)
for child in soup.body.descendant:
	print(child)			遍历子孙节点	
```

标签书的上行遍历

```python
for parent in soup.a.parents:
    if parent is None:
        print(parent)
    else:
        print(parent.name)
```

| 属性               | 说明                                                 |
| ------------------ | ---------------------------------------------------- |
| .next_sibling      | 返回按照HTML文本顺序的下一个平行节点标签             |
| .previous_sibling  | 返回按照HTML文本顺序的上一个平行节点标签             |
| .next_siblings     | 迭代类型，返回按照HTML文本顺序的后续所有平行节点标签 |
| .previous_siblings | 迭代类型，返回按照HTML文本顺序的前序所有平行节点标签 |

平行遍历

```python
for sibling in soup.a.next_siblings:
	print(sibling)#遍历后续节点
for sibling in soup.a.previous_siblings:
	print(sibling)#前序
```

### 友好显示

```python
print(soup.prettify())
```

# 信息标记

HTML通过预定义的<>...</>标签形式组值不同类型的信息

信息标记的三种形式：xml,json,yaml

x 最早的通用信息标记语言，可扩展性好，但繁琐，Internet上的信息交互与传递

j信息有类型，适合程序处理，比x简洁，移动应用云端和节点的信息通讯，无注释

y信息无类型，文本信息比例最高，可读性好，各种系统的配置文件，有注释易读

### 信息提取的一般方法

方法一：完整解析信息的标记形式，在提取关键信息

需要标记解释器，如bs4的标签书遍历

优点：信息解析准确

缺点：提取过程繁琐，速度慢

方法二：无视标记形式，直接搜索关键信息/

搜索，对信息文本用查找函数即可

优点：提取过程简介，速度快

缺点：对提取结果缺乏准确性与信息内容相关

方法三：融合方法

需要标记解释器及文本查找函数

# 内容查找

<>.find_all(**name**,**attrs**,**recursive**,**string,******kwalrgs**)

返回一个列表类型，存储查找结果。

**name**：对标签名称的检索字符串

**attrs**:对标签属性值的检索字符串，可标注属性检索

**recursive**:是否对子孙全部检索，默认True

**string**：<>...</>中字符串区域的检索字符串

```python
#打印所有标签
for tag in soup.find_all(True):
    print(tag.name)
#打印有关b的所有标签
import re
for tag in soup.find_all(re.compile('b')):
    print(tag.name)
```

\<tag>(...)等价于\<tag>.find_all(...)

soup(...)等价于soup.find_all(...)

| 方法                       | 说明                                                    |
| -------------------------- | ------------------------------------------------------- |
| <>.find()                  | 搜索且只返回一个结果，字符串类型，同.find_all()参数     |
| <>.find_parents()          | 在先辈节点中搜索，返回列表类型，同.find_all()参数       |
| <>.find_parent()           | 在先辈节点中返回一个结果，字符串类型，同.find()参数     |
| <>.find_next_siblings()    | 在后续平行节点中搜索，返回列表类型，同.find_all()参数   |
| <>,find_next_sibling()     | 在后续平行节点中返回一个结果，字符串类型，同.find()参数 |
| <>.find_previous_siblngs() | 在前序平行节点中搜索，返回列表类型，同.find_all()参数   |
| <>.find_previous_sibling() | 在前序平行节点中返回一个结果，字符串类型，同.find()参数 |

查找内容的主要方法

# 实例中国大学排名

代码

采用中文字符空格填充chr(12288)



```python
import requests
import bs4
from bs4 import BeautifulSoup

def getHTMLText(url):
    try:
        r = requests.get(url,timeout = 30)
        r.raise_for_status()
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return ''

def fillUnivList(ulist,html):
    soup = BeautifulSoup(html,"html.parser")
    for tr in soup.find('tbody').children:
        if isinstance(tr,bs4.element.Tag):
            tds = tr('td')
            ulist.append([tds[0].string,tds[1].string,tds[2].string,tds[3].string])

def printunivList(ulist,num):
    tplp = "{0:^10}\t{1:{3}^10}\t{2:^10}"
    print(tplp.format("排名","学校","成绩",chr(12288)))
    for i in range(num):
        u = ulist[i]
        print(tplp.format(u[0],u[1],u[3],chr(12288)))
              
def main():
    uinfo = []
    url = "http://www.zuihaodaxue.com/zuihaodaxuepaiming2019.html"
    html = getHTMLText(url)
    fillUnivList(uinfo,html)
    printunivList(uinfo,20)
              
main()
```

