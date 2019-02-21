import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # y即目标年龄
    y = known_age[:, 0]
    # X即特征属性值
    X = known_age[:, 1:]
    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1:])
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges
    return df, rfr

def set_cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df

data_train = pd.read_csv("/Users/apple/Downloads/titanic/train.csv")
#print(data_train.info())
print(data_train.describe())

# fig = plt.figure()
# fig.set(alpha=0.2)
#
# plt.subplot2grid((2,3), (0,0))
# data_train.Survived.value_counts().plot(kind='bar')
# plt.title("survived result（1 is survived）")
# plt.ylabel("amount")
#
# plt.subplot2grid((2,3), (0,1))
# data_train.Pclass.value_counts().plot(kind='bar')
# plt.title("passengers's level")
# plt.ylabel("amount")
#
# plt.subplot2grid((2,3), (0,2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.title("Rescue distribution by age (1 for rescue)")
# plt.ylabel("age")
#
# plt.subplot2grid((2,3), (1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.title("Age distribution of passengers of all levels")
# plt.xlabel("age")
# plt.ylabel("density")
# plt.legend(("fist class", "second class", "third class"), loc="best")
#
# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title("Number of people boarding at each boarding port")
# plt.ylabel("amount")
# plt.grid(linestyle='-.')
# plt.show()

#看看各乘客等级的获救情况
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'servived':Survived_1, u'unservived':Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title("Rescue of each passenger level")
# plt.xlabel("level")
# plt.ylabel("amount")
# plt.show()
# #看看各性别乘客的获救情况
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
# df=pd.DataFrame({'male':Survived_m, 'female':Survived_f})
# df.plot(kind='bar', stacked=True)
# plt.title("Rescue of each passenger sex")
# plt.xlabel("sex")
# plt.ylabel("amount")
# plt.show()
#
#  #然后我们再来看看各种舱级别情况下各性别的获救情况
# fig=plt.figure()
# fig.set(alpha=0.65) # 设置图像透明度，无所谓
# plt.title(u"根据舱等级和性别的获救情况")
#
# ax1=fig.add_subplot(141)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
# ax1.set_xticklabels(["survived", "unsurvived"], rotation=0)
# ax1.legend(["female/first class"], loc='best')
#
# ax2=fig.add_subplot(142, sharey=ax1)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
# ax2.set_xticklabels(["unservived", "survived"], rotation=0)
# plt.legend(["female/low class"], loc='best')
#
# ax3=fig.add_subplot(143, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
# ax3.set_xticklabels(["unservived", "survived"], rotation=0)
# plt.legend(["male/first class"], loc='best')
#
# ax4=fig.add_subplot(144, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
# ax4.set_xticklabels(["unsurvived", "survived"], rotation=0)
# plt.legend(["male/low class"], loc='best')
#
# plt.show()
#
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"各登录港口乘客的获救情况")
# plt.xlabel(u"登录港口")
# plt.ylabel(u"人数")
#
# plt.show()

# g = data_train.groupby(['SibSp','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)
#
# g = data_train.groupby(['Parch','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)
#
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
# df.plot(kind='bar', stacked=True)
# plt.title(u"按Cabin有无看获救情况")
# plt.xlabel(u"Cabin有无")
# plt.ylabel(u"人数")
# plt.show()

#填补缺失值
data_train, rfr = set_missing_ages(data_train)
data_train = set_cabin_type(data_train)

#类别向量化
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

#Age Fare标准化
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1,1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param)

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

y = train_np[:, 0]
x = train_np[:, 1:]
#LogisticRegression
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
#print(cross_val_score(clf, x, y, cv=5))
#plot_learning_curve(clf, u"学习曲线", x, y)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_features=1.0, max_samples=0.8, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(x, y)

#测试集
data_test = pd.read_csv("/Users/apple/Downloads/titanic/test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges

data_test = set_cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_bagging_predictions.csv", index=False)












