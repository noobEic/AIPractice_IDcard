# 人工智能实践——大模型身份证问答

![IDcard.drawio](./IDcard.drawio.png)

## 任务背景

在日常生活中，涉及居民身份证的业务（如首次申领、换领、补领、临时身份证办理、跨省通办等）是群众经常需要办理的事项。然而，由于相关政策法规、所需材料、办理流程和地点可能因地区和具体情况而异，群众在办理前往往会遇到各种疑问。

传统的咨询方式，如查阅官方网站的静态 FAQ、拨打政务服务热线或前往线下服务窗口，存在一些局限性。官方网站的信息可能分散、更新不及时或难以检索；咨询热线常遇占线或服务时间有限；线下窗口则面临人流量大、排队等待时间长以及工作人员需重复解答常见问题的压力。这些问题导致群众获取信息不够便捷高效，降低了服务体验，也占用了宝贵的政务服务资源。

为了提升政务服务效率、优化群众办事体验，并缓解基层工作人员的咨询压力，利用先进的人工智能技术构建一个智能化的身份证业务问答系统成为一个迫切需求。大语言模型（LLM）凭借其强大的自然语言理解和生成能力，能够在海量的身份证业务相关文本资料（如法规文件、办理指南、常见问题解答等）中学习、整合知识，并以自然、易懂的方式回答用户提出的各类复杂或具体的身份证业务问题。

因此，本项目旨在探索和实践如何利用大语言模型技术，构建一个高效、准确、便捷的身份证业务智能问答系统，使用户能够随时随地通过自然语言交互，快速获得权威、可靠的身份证业务咨询服务。

## 任务设计

项目初步设计流程如下：

1. 对身份证办理相关资料进行预处理（清洗和分块），将文本数据转换为向量表示，并构建向量数据库。

2. 使用RAG（Retrieval-Augmented Generation）构建大模型，并设计相关prompt用于合成问答数据。

   基于现实场景，设计以下三类问答数据：

   - 简单问答（SimpleQA）：基于身份证业务常见问题的回答，涉及参考文档的单一语段。

     ```json
     {
     	"question": "临时居民身份证的有效期是多久？",
     	"answer": "临时居民身份证的有效期为三个月。",
        "reference": "临时居民身份证的有效期限为三个月，有效期限自签发之日起计算。"
     }
     ```

   - 复杂问答（ComplexQA）：较复杂的问题，回答涉及参考文档多个部分的语段。

     ```json
     {
     	"question": "广东省内如何办理临时居民身份证？需要哪些材料和办理流程？",
     	"answer": "广东省户籍居民在申领、换领或补领居民身份证期间急需使用身份证件的，可携带《居民身份证领取凭证》原件、户口簿原件及复印件、近期免冠彩色照片1张，前往户籍所在地或现居住地公安机关户政窗口申请办理临时居民身份证，缴纳10元工本费后即可现场领取，该临时身份证有效期为3个月。",
     	"reference": ["临时居民身份证的有效期限为三个月，有效期限自签发之日起计算。", "公民申请领取临时居民身份证应当缴纳证件工本费。工本费标准由国务院价格主管部门会同国务院财政部门核定。", "广东省户籍居民可在省内任一公安机关户政窗口办理临时居民身份证业务，实现全省通办。"]
     }
     ```

   - 事实检查（Factcheck）：基于参考文档，判断某一陈述的正确性。

     ```json
     {
     	"statement": "所有省份均可通过广东省办理跨省通办身份证业务。",
     	"label": false,
         "reference": "北京、辽宁等未对接地区暂不能受理跨省通办。"
     }
     ```

3. 利用合成数据进一步对模型进行微调，或直接使用已有模型进行推理。

4. 使用多个指标评估模型回复质量，并根据实验结果进一步优化上一步中所使用的模型，以达到预期效果。

   - NLP通用指标
   - RAG系统评估指标

   

## 向量数据库构建

### 文档预处理

### RAG构建

### 生成数据集







## 方案1——RAG

### 原理解释

### 实验设置



## 方案2——SFT

### 原理解释

### 实验设置



## 效果评估

### 

## 参考资料

### Repositories



### Publications

[^1]: Lewis P, Perez E, Piktus A, et al. Retrieval-augmented generation for  knowledge-intensive nlp tasks[J]. Advances in neural information  processing systems, 2020, 33: 9459-9474.

[^2]: De Lima R T, Gupta S, Ramis C B, et al. Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems[C]//Proceedings of the 31st International Conference on Computational Linguistics: Industry Track. 

[^3]: Es S, James J, Anke L E, et al. Ragas: Automated evaluation of retrieval augmented generation[C]//Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations. 2024: 150-158.

[^4]: Roychowdhury S, Soman S, Ranjani H G, et al. Evaluation of RAG Metrics  for Question Answering in the Telecom Domain[J]. CoRR, 2024.
