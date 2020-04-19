# Scrapy1.3

功能强大的网络爬虫框架

爬虫框架：是实现爬虫功能的一个软件结构和功能组件集合。

半成品，帮助用户实现专业网络爬虫

Spider

- 解析Downloader返回的响应
- 产生爬取项
- 产生额外的爬取请求

Item Pipelines

- 以流水线方式处理Spider产生的爬取项
- 有一组操作顺序组成的，类似流水线，每个操作是一个Item Pipeline类型
- 可能操作包括：清理、检验、和查重爬取项中的HTML数据、将数据储存到数据库

Spider Middleware

目的：对请求和爬取项的再处理

功能：修改、丢弃、新增请求或爬取项

| requests                 | Scrapy                     |
| ------------------------ | -------------------------- |
| 页面级爬虫               | 网站级爬虫                 |
| 功能库                   | 框架                       |
| 并发性考虑不足，性能较差 | 并发性好，性能较高         |
| 重点在于页面下载         | 重点在于爬虫结构           |
| 定制灵活                 | 一般定制灵活，深度定制困难 |
| 上手简单                 | 稍难                       |

Scarapy命令行

> scrapy\<command>[options]\[args]

| 命令         | 说明              | 格式                                     |
| ------------ | ----------------- | ---------------------------------------- |
| startproject | 创建一个新工程    | scrapy startproject\<name>[dir]          |
| genspider    | 创建一个爬虫      | scrapy genspider[options]\<name>\<domin> |
| settings     | 获得爬虫配置信息  | scrapy settings[options]                 |
| crawl        | 运行一个爬虫      | scrapy crawl\<spider>                    |
| list         | 列出工程所有爬虫  | scrapy list                              |
| shell        | 启动URL调试命令行 | scrapy shell [url]                       |

### 生成文件

```
切换自己所存盘位置
cd ‘文件夹名称’
生成框架
scrapy startproject '框架名称'
```

生成的工程目录

- '框架名'/		------>外层目录
- scrapy.cfg     ------>部署Scrapy爬虫的配置文件
- 框架名/          ------>Scrapy框架的用户自定义python代码
- \_init_.py         ------>初始化脚本
- items.py         ------>Items代码模板（继承类）
- middlewares.py-->Middlewares代码模式（继承类）
- pipelines.py  ------>Pipelines代码模板（继承类）
- settings.py    ------>Scrapy爬虫的配置文件 
- spiders/         ------>Spiders代码模板目录（继承类）		



```python
# -*- coding: utf-8 -*-
import scrapy


class DemoSpider(scrapy.Spider):
    name = 'demo'
    start_urls =['http://python123.io/ws/demo.html']
    #['http://python123.io/ws/demo.html']

    def parse(self, response):
        fname = response.url.split('/')[-1]
        with open(fname,'wb') as f:
            f.write(response.body)
        self.log('Saved file %s.' %fname)

```

### 基本使用

1. 步骤一：创建一个工程和spider模板
2. 编写spider
3. 编写item pipeline
4. 优化配置策略





