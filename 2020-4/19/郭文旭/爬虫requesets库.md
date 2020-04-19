# python网络爬虫

## 1.Requsets库中的get（）方法

获得网页方法

r = requests.get(url)

构造一个向服务器请求资源的Request对象

返回一个包含服务器资源的Response对象

request.get(url,params = None,**kwargs)

url: 拟获取页面的url链接

params：url中的额外参数，字典或字节流格式，可选

**kwargs：12个控制访问的参数

| 属性                | 说明                                             |
| ------------------- | ------------------------------------------------ |
| r.status_code       | HTTP请求的返回状态，200表示连接成功，404表示失败 |
| r.text              | HTTP响应内容的字符串形式，即，url对应的页面内容  |
| r.encoding          | 从HTTP header中猜测的相应内容编码方式            |
| r.apparent_encoding | 从内容中分析出的响应内容编码方式（备选编码方式） |
| r.content           | HTTP响应内容的二进制形式                         |

### 流程

1.先r.status_code判断

2.成功：r.text \ r.encoding \ r.apparent_encoding \ r.content

   失败：其他原因出错将产生异常

| 属性                | 说明                                             |
| ------------------- | ------------------------------------------------ |
| r.encoding          | 从HTTP header中猜测的响应内容编码方式            |
| r.apparent_encoding | 从内容中分析出的响应内容编码方式（备选编码方式） |

r.encoding:如果header中不存在charset，则认为编码为SO-8859-1

r.apparent_encoding:根据网页内容分析出的编码方式

## 2.爬取网页通用代码框架

| 异常                      | 说明                                    |
| ------------------------- | --------------------------------------- |
| requests.CommectionError  | 网络连接错误，如DNS查询失败、拒绝连接等 |
| requests.HTTPError        | HTTP错误异常                            |
| requests.URLRequired      | URL缺失异常                             |
| requests.TooManyRedirects | 超过最大重定向次数，产生重定向异常      |
| requests.ConnectTimeout   | 连接远程服务器超时异常                  |
| requests.Timeout          | 请求URL超时，产生超时异常               |

| 异常                 | 说明                                   |
| -------------------- | -------------------------------------- |
| r.raise_for_status() | 如果不是200，产生异常request.HTTPError |

### 爬取网页的通用代码

```python
import requests

def getHTMLText(url):
	try:
        r = request.get(url, timeout = 30)
        r.raise_for_status()	#如果状态不是200，错误
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return "产生异常"
    
if _name_ == "_main_":
    url = "http://www.baidu.com"
    print(getHTMLText(url))
```

## 3.HTTP协议

超文本传输协议

| HTTP协议方法 | Requests库方法    | 功能 |
| ------------ | ----------------- | ---- |
| GET          | requests.get()    | 一致 |
| HEAD         | requests.head()   | 一致 |
| POST         | raquests.post()   | 一致 |
| PUT          | requests.put()    | 一致 |
| PATCH        | requests.patch()  | 一致 |
| DELETE       | requests.delete() | 一致 |

## 4.Requests库解析

requests.request(method,url,**kwargs)

mathod:请求方式，对应get/put/post等操作

url：拟获取页面的url链接

#### **kwargs：

控制访问的参数，共13个

**params**：字典或字节序列，作为参数增加到url之中

**data**：字典、字节序列或文件对象，作为request的内容

**json**：json格式的数据，作为request的内容

**headers**：字典，HTTP定制头

cookies：字典或CookieJar，request中的cookie

auth：元组，支持HTTP认证功能

files：字典类型，传输文件

timeout：设定超时时间，秒为单位

proxies：字典类型，设定访问代理服务器，可以增加登录认证

allow_redirects:True/False，默认为True，重定向开关

stream：True/False，默认为True，获取内容立即下载开关

verify：Ture/False，默认为True，认证SSL证书开关

cert：本地SSL证书路径

#### 函数

requests.get(url,params = None,**kwargs)

url：拟获取页面的url链接

params:url中的额外参数，字典或字节流格式

requests.head(url,**kwargs)

requests.post(url,data = None,json = None,**kwags)

requests.put(ual,data = None,**kwargs)

requests.patch(ual,data = None,**kwargs)

requests.delete(url,**kwargs)

# 网络爬虫第二课

### 对应尺寸选择不同的爬虫库

小规模，数据量小，爬取速度不敏感，requests库，针对网页，占用大多数爬虫

中规模，数据规模较大，爬取速度敏感，scrapy库，爬取网站，爬取系列网站

大规模，搜索引擎是爬取速度的关键，爬取全网，定制开发

网络爬虫有法律风险

服务器获取数据有风险，隐私泄露的风险

网络爬虫可能具备一定突破简单访问控制的能力，获得被保护数据从而泄露个人隐私。

### robots协议

告知爬虫网站的爬取策略，要求爬虫遵守。（道德限制）

Robots Exclusion Standard:网络爬虫排除标准

作用：网站告知爬虫哪些页面可以抓取，那些不行。

形式：网站根目录下的robots.txt文件。

遵守方式：约束性。可以不遵守，但存在法律风险。

协议基本语法

#注释，*代表所有，/代表根目录

User-agent: *

Disallow: /

### 实际爬取演习

亚马逊，改头名称为'user-agent':'Mozilla/5.0'

提取网站信息

```python
import requests
url = "https://www.amazon.cn/dp/B01MYWGSG2/ref=sr_1_1?dchild=1&keywords=Champion&p_n_global_store_origin_marketplace=1827360071%7C1844252071%7C1879515071%7C1901313071&qid=1586787265&sr=8-1"
try:
    kv = {'user-agent':'Mozilla/5.0'}#常用的欺骗网页头
    r = requests.get(url,headers = kv)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    print(r.text)
except:
    print("爬取失败")
```

百度提交关键词获得搜索结果

百度关键词接口：

http://www.baidu.com/s?wd=keyword

360关键词接口：

http://www.so.com/s?q=keyword

替换keyword就可以向搜索引擎提交关键词

```python
import requests
keyword = "python"
try:
    kv = {'wd':keyword}
    r = requests.get("https://www.baidu.com/",params = kv)
    print(r.request.url)
    r.raise_for_status()
    print(len(r.text))
except:
    print("paqushibai")
```

爬取图片下载

```python
import requests
import os
url = "https://dss2.bdstatic.com/70cFvnSh_Q1YnxGkpoWK1HF6hhy/it/u=1091978681,1253128401&fm=26&gp=0.jpg"
root = "D://paqic//"
path = root + url.split('/')[-1]#文件地址为根目录加上截取图片名称
try:
    if not os.path.exists(root):#判断根目录是否存在
        os.mkdir(root)
    if not os.path.exists(path):#判断文件是否存在
        r = requests.get(url)
        with open(path,'wb') as f:
            f.write(r.content) #将返回的二进制数据写到文件中
            f.close()
            print("success")
    else:
        print("alreaded")
except:
    print("fail")
```

ip地址查询

```python
import requests
url = "https://www.ip138.com/ip.asp?ip="
try:
    r = requests.get(url + '202.204.80.112')
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    print(r.text[-500:])
except:
    print("fail")
```



