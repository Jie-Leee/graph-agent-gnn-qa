# FB15k-237 两跳问答数据说明

> 本文件由 `scripts/prepare_data.py` 自动生成，请勿手工修改。 模板版本 `v4`，随机种子 `42`。

## 一、原始图谱

| 项目 | 数值 |
|---|---|
| 训练三元组 | 272115 |
| 验证三元组 | 17535 |
| 测试三元组 | 20466 |
| 实体数（全库） | 14541 |
| 关系数（全库） | 237 |
| 训练集实体数 | 14505 |
| 训练集关系数 | 237 |
| 平均出度 | 19.75 |
| 最大出度 | 1325 |
| 最大入度 | 6289 |
| 自环三元组 | 1625 |

文本映射覆盖率：

| 映射文件 | 条目数 | 训练集覆盖率 |
|---|---|---|
| entity2text.txt | 14951 | 100.00% |
| relation2text.txt | 237 | 100.00% |

## 二、可见图政策

问答只在**训练三元组**构成的可见图上构造。验证集与测试集的三元组不参与出题，也不作为答案来源。这样每道题的金标答案都能沿可见图正向走到，评测的是「在已知图谱上问答」，而不是原始链接预测协议。

本数据集只沿原始三元组方向构造两跳题，不自动添加反向边。某类问题能否构造取决于实际关系及方向；FB15k-237 去除了反向关系冗余，不代表所有自然语言关系都具有相同的存储方向。

## 三、问答数据规模

| 划分 | 问答条数 | 组数 | 每组问法 | 平均答案数 | 最大答案数 | 路径类型数 |
|---|---|---|---|---|---|---|
| 训练集 | 2000 | 1000 | 2 | 2.81 | 20 | 39 |
| 验证集 | 200 | 100 | 2 | 2.89 | 18 | 27 |
| 测试集 | 300 | 150 | 2 | 2.89 | 19 | 33 |

## 四、答案规模分布

| 答案数 | 问答条数 |
|---|---|
| 1 | 1278 |
| 2 | 424 |
| 3 | 204 |
| 4 | 158 |
| 5 | 144 |
| 6 | 48 |
| 7 | 82 |
| 8 | 20 |
| 9 | 26 |
| 10 | 4 |
| 11 | 16 |
| 12 | 20 |
| 13 | 4 |
| 14 | 6 |
| 15 | 18 |
| 16 | 4 |
| 17 | 2 |
| 18 | 36 |
| 19 | 2 |
| 20 | 4 |

## 五、两跳路径分布

白名单共 24 个语义明确的关系，两两组合后实际用到 39 条路径。下列为问答条数最多的 15 条。

| 第一跳 | 第二跳 | 问答条数 |
|---|---|---|
| place of birth | contained location | 266 |
| composer | profession | 174 |
| producer | profession | 170 |
| production company | headquarters city | 160 |
| producer | place of birth | 142 |
| country | official language | 130 |
| writer | profession | 116 |
| writer | place of birth | 96 |
| composer | place of birth | 94 |
| headquarters city | time zone | 86 |
| featured location | time zone | 86 |
| contained location | time zone | 80 |
| contained location | school type | 78 |
| artist | place of birth | 60 |
| place of origin | contained location | 58 |

## 六、问法模板

每个组内选 2 种不同问法，模板由组键的稳定哈希决定，组内全部问法必然落在同一划分。所有题面都要求返回完整答案集，措辞不依据标准答案个数切换。

| 模板 | 使用次数 |
|---|---|
| `Return every {r2} of any {r1} of {head}.` | 659 |
| `List every {r2} of any {r1} of {head}.` | 633 |
| `For {head}, list every {r2} of any {r1}.` | 617 |
| `Name every {r2} of any {r1} of {head}.` | 591 |

## 七、泄漏审计

| 检查项 | 结果 |
|---|---|
| 同一组跨划分出现 | 0 |
| 同一问句跨划分出现 | 0 |
| 划分内重复问句 | 0 |
| 答案包含主题实体 | 0 |
| 空答案 | 0 |
| 证据不能按两跳路径连通答案 | 0 |
| 证据边不在训练图 | 0 |
| 标准答案不等于训练图完整可达集合 | 0 |
| 重复问题 ID | 0 |
| 训练/测试主题实体重叠 | 27 |
| 测试集主题实体未在训练集出现 | 121 |
| 训练/测试答案实体重叠 | 164 |

全部检查通过：组与问句未跨划分，答案与训练图完整两跳可达集合一致，证据可连通且均在训练图内。

本划分是共享训练图上的查询组划分；实体、答案及部分证据边可以跨 QA 划分共享，不能据此宣称实体不相交、边不相交或未知事实泛化。提供的主题 ID 属于已知主题设置；若评估实体定位能力，需另从题面预测主题，不能把给定 ID 当作定位结果。

## 八、被排除的关系路径

「主答案占比」= 最常见单个答案占该路径全部答案的比例；「重复答案集占比」= 出现次数最多的完整答案集占该路径组数的比例。两者过高都意味着不查图也能猜到答案。

| 路径 | 组数 | 主答案占比 | 重复答案集占比 | 排除原因 |
|---|---|---|---|---|
| school type >> headquarters city | 10 | 100% | 100% | support<25 |
| country >> capital city | 1651 | 57% | 57% | repeated answer set 57%>40% |
| country >> form of government | 1653 | 50% | 60% | repeated answer set 60%>40% |
| country >> time zone | 1589 | 18% | 62% | repeated answer set 62%>40% |
| country >> gender | 2 | 100% | 100% | support<25 |
| featured location >> school type | 17 | 31% | 35% | support<25 |
| featured location >> headquarters city | 13 | 31% | 31% | support<25 |
| featured location >> gender | 1 | 100% | 100% | support<25 |
| genre >> artist | 155 | 53% | 64% | repeated answer set 64%>40% |
| language >> capital city | 1 | 100% | 100% | support<25 |
| language >> form of government | 3 | 67% | 67% | support<25 |
| language >> official language | 1 | 100% | 100% | support<25 |
| language >> time zone | 3 | 67% | 67% | support<25 |
| composer >> gender | 760 | 98% | 98% | dominant answer 98%>60% |
| composer >> country of citizenship | 789 | 51% | 54% | repeated answer set 54%>40% |

## 九、复现方式

```bash
python scripts/prepare_data.py --raw data/raw --output data/processed
```

随机种子 `42`、模板版本 `v4`、每组问法数 `2`、答案数上限 `20`，候选上限 `60000`，路径支持下限 `25`，主答案占比上限 `0.6`，重复答案集占比上限 `0.4`。原始文件、生成文件和数据模块代码的 sha256 均记录在 `manifest.json`。
