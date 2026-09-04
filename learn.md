表格分类问题常用的模型：

1. 第一梯队（最常用、效果通常最好）
   - CatBoost
   - LightGBM
   - XGBoost
   - 这三类 GBDT 在 Kaggle tabular 里最主流
2. 强基线/线性方法
   - LogisticRegression（多分类）
   - SGDClassifier（log_loss）
   - 配合 OneHot + 标准化，速度快、可解释性好
3. 袋装树模型
   - RandomForest
   - ExtraTrees
   - 稳定、好用，但很多场景不如 GBDT 上限高
4. 核方法（数据量大时较少用）
   - SVM / LinearSVC
   - 大样本下训练代价高，通常先用线性版
5. 神经网络（表格上不一定稳赢）
   - MLP
   - TabNet/FT-Transformer 类方法（进阶）
   - 需要更细调参，收益不一定稳定
6. 集成策略（冲榜常见）
   - 简单加权平均（概率）
   - Stacking（GBDT + 线性/NN 二层）
   - 多 seed / 多模型融合提升稳健性
