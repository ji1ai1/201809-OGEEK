#python 3.7.0
#python套件 jieba 0.39
#python套件 lightgbm 2.2.1
#python套件 numpy 1.15.2
#python套件 pandas 0.23.4
#python套件 sklearn 0.0.13
#
#输入：
#	data_train.txt
#	data_vali.txt
#	data_test.txt
#输出：
#	result.zip


import datetime
import jieba
import json
import lightgbm
import numpy
import pandas
import random
import sklearn.metrics
import zipfile

訓練一表 = pandas.read_csv("data_train.txt", header=None, quoting=3, sep ="\t", names=["前綴", "查預", "標題", "分類", "標籤"])
訓練一表["標識"] = list(range(-1, -1 - 訓練一表.shape[0], -1))
訓練一表["列號"] = list(range(訓練一表.shape[0]))
訓練一表["來源標識"] = 2000000 * [1]

訓練二表 = pandas.read_csv("data_vali.txt", header=None, quoting=3, sep ="\t", names=["前綴", "查預", "標題", "分類", "標籤"])
訓練二表["標識"] = list(range(-1 - 訓練一表.shape[0], -1 - 訓練一表.shape[0] - 訓練二表.shape[0], -1))
訓練二表["列號"] = list(range(訓練二表.shape[0]))
訓練二表["來源標識"] = 30000 * [2] + 20000 * [3]

訓練表 = pandas.concat([訓練一表, 訓練二表]).reset_index(drop=True)

測試表 = pandas.read_csv("data_test.txt", header=None, quoting=3, sep = "\t", names=["前綴", "查預", "標題", "分類"])
測試表["標籤"] = -1
測試表["標識"] = list(range(1, 1 + 測試表.shape[0]))
測試表["列號"] = list(range(測試表.shape[0]))
測試表["來源標識"] = 120000 * [2] + 80000 * [3]

來源3測試表 = 測試表.loc[~測試表.前綴.isin(訓練表.前綴.drop_duplicates())].reset_index(drop=True)
來源3測試表 = 來源3測試表.loc[來源3測試表.來源標識 == 3].reset_index(drop=True)
來源2測試表 = 測試表.loc[~測試表.標識.isin(來源3測試表.標識)].reset_index(drop=True)




測訓表 = pandas.concat([測試表, 訓練表]).reset_index(drop=True)

前綴單詞清單 = [jieba.lcut(str(甲)) for 甲 in 測訓表.前綴]
標題單詞清單 = [jieba.lcut(str(甲)) for 甲 in 測訓表.標題]
前綴單詞表 = pandas.DataFrame({"標識": [乙 for 甲 in range(測訓表.shape[0]) for 乙 in [測訓表.標識[甲]] * len(前綴單詞清單[甲])], "單詞": [乙 for 甲 in 前綴單詞清單 for 乙 in 甲], "前綴一":1})
標題單詞表 = pandas.DataFrame({"標識": [乙 for 甲 in range(測訓表.shape[0]) for 乙 in [測訓表.標識[甲]] * len(標題單詞清單[甲])], "單詞": [乙 for 甲 in 標題單詞清單 for 乙 in 甲], "標題一":1})

測訓單詞表 = 前綴單詞表.merge(標題單詞表, on = ["標識", "單詞"], how="outer")
測訓單詞表 = 測訓單詞表.groupby("標識").agg({"單詞":len, "前綴一":numpy.sum, "標題一":numpy.sum}).reset_index()
測訓單詞表.columns = ["標識", "前標單詞數", "前綴單詞數", "標題單詞數"]
測訓單詞表["前綴單詞比"] = 測訓單詞表.前標單詞數 / 測訓單詞表.前綴單詞數
測訓單詞表["標題單詞比"] = 測訓單詞表.前標單詞數 / 測訓單詞表.標題單詞數

def 求字串相似度甲(字串甲, 字串乙):
	return float(len(set(字串甲.upper()).intersection(字串乙.upper())))

def 求字串相似度乙(字串甲, 字串乙):
	if len(字串甲) >= len(字串乙):
		字串丙 = 字串乙
		字串乙 = 字串甲
		字串甲 = 字串丙

	不同清單 = [甲 for 甲 in range(len(字串甲)) if 字串甲[甲] != 字串乙[甲]]
	if len(不同清單) == 0:
		return float(1 / (1 + len(字串甲)))
	return float(1 / (1 + min(不同清單)))

合法查預測訓表 = 測訓表.iloc[[甲 for 甲 in range(測訓表.shape[0]) if str(測訓表.查預[甲])[0] == "{"]]

查預表 = 合法查預測訓表.loc[:, ["查預"]].drop_duplicates().reset_index(drop=True)
查預解析清單 = [json.loads(str(甲)) for 甲 in 查預表.查預]
查預分表 = pandas.DataFrame({"查預": [乙 for 甲 in range(len(查預解析清單)) for 乙 in [查預表.查預[甲]] * len(查預解析清單[甲])], "查預文本": [乙 for 甲 in 查預解析清單 for 乙 in list(甲.keys())], "查預打分": [float(乙) for 甲 in 查預解析清單 for 乙 in list(甲.values())]})
查預分表["查預排名"] = 查預分表.groupby("查預")["查預打分"].rank(ascending=False, method="average")

查預標題表 = 合法查預測訓表.loc[:, ["查預", "標題"]].drop_duplicates().reset_index(drop=True)
查預標題表 = 查預標題表.merge(查預分表, on = "查預")
查預標題表["標題長度"] = [len(str(甲)) for 甲 in 查預標題表.標題]
查預標題表["查預文本長度"] = [len(str(甲)) for 甲 in 查預標題表.查預文本]
查預標題表["查標相似度甲"] = [求字串相似度甲(str(查預標題表.標題[甲]), str(查預標題表.查預文本[甲])) for 甲 in range(查預標題表.shape[0])]
查預標題表["查標比例相似度甲"] = 查預標題表.查標相似度甲 / 查預標題表.查預文本長度 ** 0.5 / 查預標題表.標題長度 ** 0.5
查預標題表 = 查預標題表.merge(查預標題表.groupby(["查預", "標題"]).agg({"查標比例相似度甲": numpy.max}).rename(columns={"查標比例相似度甲": "最大查標比例相似度甲"}).reset_index(), on=["查預", "標題"])

查預標題統計表 = 查預標題表.groupby(["查預", "標題"]).agg({"查預打分": [numpy.min, numpy.max], "查標相似度甲": numpy.max, "查標比例相似度甲": numpy.max}).reset_index()
查預標題統計表.columns = ["查預", "標題", "最小查預打分", "最大查預打分", "最大查標相似度甲", "最大查標比例相似度甲"]
查預標題統計表 = 查預標題統計表.merge(查預標題表.loc[查預標題表.標題.str.upper() == 查預標題表.查預文本.str.upper()].groupby(["查預", "標題"]).agg({"查預打分": numpy.mean, "查預排名": numpy.mean}).rename(columns={"查預打分": "查標打分", "查預排名": "查標排名"}).reset_index(), on=["查預", "標題"], how="left")
查預標題統計表 = 查預標題統計表.merge(查預標題表.loc[查預標題表.查標比例相似度甲 == 查預標題表.最大查標比例相似度甲].groupby(["查預", "標題"]).agg({"查預打分": numpy.mean, "查預排名": numpy.mean}).rename(columns={"查預打分": "查預比例打分", "查預排名": "查預比例排名"}).reset_index(), on=["查預", "標題"], how="left")
del 查預標題表

前綴查預表 = 合法查預測訓表.loc[:, ["前綴", "查預"]].drop_duplicates().reset_index(drop=True)
前綴查預表 = 前綴查預表.merge(查預分表, on = "查預")
前綴查預打分表 = 前綴查預表.copy()
前綴查預打分表["查預打分平方和"] = 前綴查預打分表.查預打分 ** 2
前綴查預打分表 = 前綴查預打分表.groupby("前綴").agg({"查預打分平方和": numpy.sum}).reset_index()

相似前綴表 = 前綴查預表.loc[:, ["前綴", "查預文本"]].merge(前綴查預表.loc[:, ["前綴", "查預文本"]].rename(columns={"前綴": "相似前綴"}), on="查預文本")
相似前綴表 = 相似前綴表.loc[相似前綴表.前綴 != 相似前綴表.相似前綴, ["前綴", "相似前綴"]].drop_duplicates().reset_index(drop=True)
相似前綴查預表 = 相似前綴表.merge(前綴查預表.loc[:, ["前綴", "查預文本", "查預打分"]], on="前綴")
相似前綴查預表 = 相似前綴查預表.merge(前綴查預表.loc[:, ["前綴", "查預文本", "查預打分"]].rename(columns={"前綴": "相似前綴", "查預打分": "相似查預打分"}), on=["相似前綴", "查預文本"])
相似前綴查預表["交叉查預打分"] = 相似前綴查預表.查預打分 * 相似前綴查預表.相似查預打分
相似前綴表 = 相似前綴查預表.groupby(["前綴", "相似前綴"]).agg({"交叉查預打分": numpy.sum}).reset_index()
相似前綴表 = 相似前綴表.merge(前綴查預打分表, on="前綴")
相似前綴表 = 相似前綴表.merge(前綴查預打分表.rename(columns={"前綴": "相似前綴", "查預打分平方和": "相似查預打分平方和"}), on="相似前綴")
相似前綴表["相似打分"] = 相似前綴表.交叉查預打分 / 相似前綴表.查預打分平方和 ** 0.5 / 相似前綴表.相似查預打分平方和 ** 0.5
相似前綴表 = 相似前綴表.loc[:, ["前綴", "相似前綴", "相似打分"]].drop_duplicates().reset_index(drop=True)
相似前綴表.相似打分 = 相似前綴表.相似打分.astype("float16")
del 前綴查預表
del 相似前綴查預表

測訓前標表 = 測訓表.loc[:, ["前綴", "標題"]].drop_duplicates().reset_index()
第二相似前綴表 = 測訓前標表.merge(測訓前標表.loc[:, ["前綴", "標題"]].rename(columns={"前綴": "第二相似前綴"}), on="標題")
第二相似前綴表["第二相似打分"] = 1
第二相似前綴表 = 第二相似前綴表.loc[第二相似前綴表.前綴 != 第二相似前綴表.第二相似前綴, ["前綴", "第二相似前綴", "第二相似打分"]].drop_duplicates().reset_index(drop=True)
第二相似前綴表.第二相似打分 = 第二相似前綴表.第二相似打分.astype("float32")

第二相似標題表 = 測訓前標表.merge(測訓前標表.loc[:, ["前綴", "標題"]].rename(columns={"標題": "第二相似標題"}), on="前綴")
第二相似標題表["第二相似打分"] = 1
第二相似標題表 = 第二相似標題表.loc[第二相似標題表.標題 != 第二相似標題表.第二相似標題, ["標題", "第二相似標題", "第二相似打分"]].drop_duplicates().reset_index(drop=True)
第二相似標題表.第二相似打分 = 第二相似標題表.第二相似打分.astype("float32")

測訓相似前綴統計表 = 相似前綴表.merge(測訓表.groupby("前綴").agg({"標識": len}).reset_index().rename(columns={"前綴": "相似前綴", "標識": "相似前綴の樣本數"}), on="相似前綴").groupby(["前綴"]).agg({"相似前綴の樣本數":len}).rename(columns={"相似前綴の樣本數":"測訓相似前綴の樣本數"}).reset_index()
測訓第二相似前綴統計表 = 第二相似前綴表.merge(測訓表.groupby("前綴").agg({"標識": len}).reset_index().rename(columns={"前綴": "第二相似前綴", "標識": "第二相似前綴の樣本數"}), on="第二相似前綴").groupby(["前綴"]).agg({"第二相似前綴の樣本數":len}).rename(columns={"第二相似前綴の樣本數":"測訓第二相似前綴の樣本數"}).reset_index()
測訓第二相似標題統計表 = 第二相似標題表.merge(測訓表.groupby("標題").agg({"標識": len}).reset_index().rename(columns={"標題": "第二相似標題", "標識": "第二相似標題の樣本數"}), on="第二相似標題").groupby(["標題"]).agg({"第二相似標題の樣本數":len}).rename(columns={"第二相似標題の樣本數":"測訓第二相似標題の樣本數"}).reset_index()

測訓表["前標"] = 測訓表.前綴 + "\t" + 測訓表.標題
測訓表["前分"] = 測訓表.前綴 + "\t" + 測訓表.分類
測訓表["標分"] = 測訓表.標題 + "\t" + 測訓表.分類
測訓統計表 = 測訓表.loc[:, ["標識", "來源標識", "前綴", "查預", "分類", "標題"]]
測訓統計表 = 測訓統計表.merge(測訓表.groupby("前綴").agg({"標識":len, "標題":pandas.Series.nunique, "分類":pandas.Series.nunique, "標分":pandas.Series.nunique}).rename(columns={"標識": "測訓前綴の樣本數", "標題": "測訓前綴の標題數", "分類": "測訓前綴の分類數", "標分": "測訓前綴の標分數"}).reset_index(), on="前綴", how="left")
測訓統計表 = 測訓統計表.merge(測訓表.groupby("標題").agg({"標識":len, "前綴":pandas.Series.nunique, "分類":pandas.Series.nunique, "前分":pandas.Series.nunique}).rename(columns={"標識": "測訓標題の樣本數", "前綴": "測訓標題の前綴數", "分類": "測訓標題の分類數", "前分": "測訓標題の前分數"}).reset_index(), on="標題", how="left")
測訓統計表 = 測訓統計表.merge(測訓表.groupby("分類").agg({"標識":len, "前綴":pandas.Series.nunique, "標題":pandas.Series.nunique, "前標":pandas.Series.nunique}).rename(columns={"標識": "測訓分類の樣本數", "前綴": "測訓分類の前綴數", "標題": "測訓分類の標題數", "前標": "測訓分類の前標數"}).reset_index(), on="分類", how="left")
測訓統計表 = 測訓統計表.merge(測訓表.groupby(["前綴", "標題"]).agg({"標識":len, "分類":pandas.Series.nunique}).rename(columns={"標識":"測訓前標の樣本數", "分類":"測訓前標の分類數"}).reset_index(), on=["前綴", "標題"], how="left")
測訓統計表 = 測訓統計表.merge(測訓表.groupby(["前綴", "分類"]).agg({"標識":len}).rename(columns={"標識":"測訓前分の樣本數"}).reset_index(), on=["前綴", "分類"], how="left")
測訓統計表 = 測訓統計表.merge(測訓表.groupby(["標題", "分類"]).agg({"標識":len}).rename(columns={"標識":"測訓標分の樣本數"}).reset_index(), on=["標題", "分類"], how="left")
測訓統計表 = 測訓統計表.merge(測訓表.groupby(["前綴", "標題", "分類"]).agg({"標識":len}).rename(columns={"標識":"測訓前標分の樣本數"}).reset_index(), on=["前綴", "標題", "分類"], how="left")
測訓統計表 = 測訓統計表.merge(測訓相似前綴統計表, on="前綴", how="left")
測訓統計表 = 測訓統計表.merge(測訓第二相似前綴統計表, on="前綴", how="left")
測訓統計表 = 測訓統計表.merge(測訓第二相似標題統計表, on="標題", how="left")
測訓統計表 = 測訓統計表.fillna(0)
測訓統計表 = 測訓統計表.drop(["來源標識", "前綴", "查預", "分類", "標題"], axis=1)




來源2訓練資料表 = None
輪數 = 4
折數 = 4
每折數量 = int(訓練表.shape[0] / 折數)
random.seed(1)
print(datetime.datetime.now())
for 甲 in range(輪數):
	索引 = random.sample(range(訓練表.shape[0]),訓練表.shape[0])
	for 乙 in range(折數):
		print(甲 * 折數 + 乙)
		訓練標籤表 = 訓練表.iloc[索引[(乙 * 每折數量):(每折數量 + 乙 * 每折數量)]].reset_index(drop=True)
		訓練特征表 = 訓練表.drop(索引[(乙 * 每折數量):(每折數量 + 乙 * 每折數量)]).reset_index(drop=True)
		訓練特征表["前綴長度"] = [len(str(甲)) for 甲 in 訓練特征表.前綴]
		訓練特征表["標題長度"] = [len(str(甲)) for 甲 in 訓練特征表.長度]

		訓練前綴表 = 訓練特征表.groupby("前綴").agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤":"前綴の正樣本比例"})
		訓練標題表 = 訓練特征表.groupby("標題").agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤":"標題の正樣本比例"})
		訓練分類表 = 訓練特征表.groupby("分類").agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤":"分類の正樣本比例"})
		訓練前標表 = 訓練特征表.groupby(["前綴", "標題"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤":"前標の正樣本比例"})
		訓練前分表 = 訓練特征表.groupby(["前綴", "分類"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤":"前分の正樣本比例"})
		訓練標分表 = 訓練特征表.groupby(["標題", "分類"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤":"標分の正樣本比例"})
		訓練前標分表 = 訓練特征表.groupby(["前綴", "標題", "分類"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "前標分の正樣本比例"})
		訓練前綴長度表 = 訓練特征表.groupby(["前綴長度"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "前綴長度の正樣本比例"})
		訓練標題長度表 = 訓練特征表.groupby(["標題長度"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "標題長度の正樣本比例"})

		相似訓練前綴表 = 相似前綴表.merge(訓練特征表.loc[:, ["前綴", "標籤"]].rename(columns={"前綴": "相似前綴"}), on="相似前綴")
		相似訓練前綴表["相似標籤"] = 相似訓練前綴表.標籤 * 相似訓練前綴表.相似打分
		相似訓練前綴表 = 相似訓練前綴表.groupby("前綴").agg({"相似標籤": numpy.sum, "相似打分": numpy.sum}).reset_index()
		相似訓練前綴表["相似類前綴の正樣本比例"] = 相似訓練前綴表.相似標籤 / 相似訓練前綴表.相似打分
		相似訓練前綴表 = 相似訓練前綴表.loc[:, ["前綴", "相似類前綴の正樣本比例"]]

		折數丙 = 32
		每折數量丙 = 1 + int(訓練特征表.shape[0] / 折數丙)
		第二相似訓練前綴表 = None
		for 丙 in range(折數丙):
			訓練特征丙表 = 訓練特征表.iloc[(丙 * 每折數量丙):min(訓練特征表.shape[0], 每折數量丙 + 丙 * 每折數量丙)].reset_index(drop=True)
			第二相似訓練前綴丙表 = 第二相似前綴表.merge(訓練特征丙表.loc[:, ["前綴", "標籤"]].rename(columns={"前綴": "第二相似前綴"}), on="第二相似前綴")
			第二相似訓練前綴丙表["第二相似標籤"] = 第二相似訓練前綴丙表.標籤 * 第二相似訓練前綴丙表.第二相似打分
			第二相似訓練前綴丙表 = 第二相似訓練前綴丙表.groupby("前綴").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
			第二相似訓練前綴表 = pandas.concat([第二相似訓練前綴表, 第二相似訓練前綴丙表])
		第二相似訓練前綴表 = 第二相似訓練前綴表.groupby("前綴").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
		第二相似訓練前綴表["第二相似類前綴の正樣本比例"] = 第二相似訓練前綴丙表.第二相似標籤 / 第二相似訓練前綴丙表.第二相似打分
		第二相似訓練前綴表 = 第二相似訓練前綴表.loc[:, ["前綴", "第二相似類前綴の正樣本比例"]]

		每折數量丙 = 1 + int(訓練特征表.shape[0] / 折數丙)
		第二相似訓練標題表 = None
		for 丙 in range(折數丙):
			訓練特征丙表 = 訓練特征表.iloc[(丙 * 每折數量丙):min(訓練特征表.shape[0], 每折數量丙 + 丙 * 每折數量丙)].reset_index(drop=True)
			第二相似訓練標題丙表 = 第二相似標題表.merge(訓練特征丙表.loc[:, ["標題", "標籤"]].rename(columns={"標題": "第二相似標題"}), on="第二相似標題")
			第二相似訓練標題丙表["第二相似標籤"] = 第二相似訓練標題丙表.標籤 * 第二相似訓練標題丙表.第二相似打分
			第二相似訓練標題丙表 = 第二相似訓練標題丙表.groupby("標題").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
			第二相似訓練標題表 = pandas.concat([第二相似訓練標題表, 第二相似訓練標題丙表])
		第二相似訓練標題表 = 第二相似訓練標題表.groupby("標題").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
		第二相似訓練標題表["第二相似類標題の正樣本比例"] = 第二相似訓練標題丙表.第二相似標籤 / 第二相似訓練標題丙表.第二相似打分
		第二相似訓練標題表 = 第二相似訓練標題表.loc[:, ["標題", "第二相似類標題の正樣本比例"]]

		訓練資料甲表 = 訓練標籤表.loc[:, ["標識", "前綴", "查預", "標題", "分類", "標籤", "來源標識"]]
		訓練資料甲表["前綴長度"] = [len(str(甲)) for 甲 in 訓練資料甲表.前綴]
		訓練資料甲表["標題長度"] = [len(str(甲)) for 甲 in 訓練資料甲表.標題]
		訓練資料甲表["前標相似度甲"] = [求字串相似度甲(str(訓練資料甲表.前綴[甲]), str(訓練資料甲表.標題[甲])) for 甲 in range(訓練資料甲表.shape[0])]
		訓練資料甲表["前標相似度乙"] = [求字串相似度乙(str(訓練資料甲表.前綴[甲]), str(訓練資料甲表.標題[甲])) for 甲 in range(訓練資料甲表.shape[0])]

		訓練資料甲表 = 訓練資料甲表.merge(查預標題統計表, on=["標題", "查預"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(測訓單詞表, on="標識", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練前綴表, on="前綴", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練標題表, on="標題", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練分類表, on="分類", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練前標表, on=["前綴", "標題"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練前分表, on=["前綴", "分類"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練標分表, on=["標題", "分類"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練前標分表, on=["前綴", "標題", "分類"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練前綴長度表, on=["前綴長度"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練標題長度表, on=["標題長度"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(測訓統計表, on="標識", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(相似訓練前綴表, on="前綴", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(第二相似訓練前綴表, on="前綴", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(第二相似訓練標題表, on="標題", how="left")

		訓練資料甲表["前標の正樣本比例_比_標題の正樣本比例"] = 訓練資料甲表.前標の正樣本比例 / (1 + 訓練資料甲表.標題の正樣本比例)
		訓練資料甲表["標分の正樣本比例_比_標題の正樣本比例"] = 訓練資料甲表.標分の正樣本比例 / (1 + 訓練資料甲表.標題の正樣本比例)
		訓練資料甲表["前標分の正樣本比例_比_標題の正樣本比例"] = 訓練資料甲表.前標分の正樣本比例 / (1 + 訓練資料甲表.標題の正樣本比例)
		訓練資料甲表 = 訓練資料甲表.fillna(-1)
		訓練資料甲表 = 訓練資料甲表.drop(["前綴", "查預", "標題", "分類"], axis=1)

		來源2訓練資料表 = pandas.concat([來源2訓練資料表, 訓練資料甲表])
print(str(datetime.datetime.now()) + " 生成訓練資料表共" + str(來源2訓練資料表.shape[0]) + "列")

來源2訓練資料表 = pandas.concat([來源2訓練資料表.loc[:, ["標識", "標籤"]], 來源2訓練資料表.drop(["標識", "標籤"], axis=1)], axis=1)

來源2輕模型 = lightgbm.train(train_set = lightgbm.Dataset(來源2訓練資料表.drop(["標識", "標籤"], axis=1), 來源2訓練資料表.標籤) \
	, params={"objective": "binary", "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "bagging_fraction": 0.7, "bagging_freq": 1, "bagging_seed": 0, "verbose": -1} \
	, num_boost_round = 2000
)

測試特征表 = 訓練表.copy()
測試特征表["前綴長度"] = [len(str(甲)) for 甲 in 測試特征表.前綴]

測試前綴表 = 測試特征表.groupby("前綴").agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "前綴の正樣本比例"})
測試標題表 = 測試特征表.groupby("標題").agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "標題の正樣本比例"})
測試分類表 = 測試特征表.groupby("分類").agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "分類の正樣本比例"})
測試前標表 = 測試特征表.groupby(["前綴", "標題"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "前標の正樣本比例"})
測試前分表 = 測試特征表.groupby(["前綴", "分類"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "前分の正樣本比例"})
測試標分表 = 測試特征表.groupby(["標題", "分類"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "標分の正樣本比例"})
測試前標分表 = 測試特征表.groupby(["前綴", "標題", "分類"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "前標分の正樣本比例"})
測試前綴長度表 = 測試特征表.groupby(["前綴長度"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "前綴長度の正樣本比例"})
測試標題長度表 = 測試特征表.groupby(["標題長度"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "標題長度の正樣本比例"})

相似測試前綴表 = 相似前綴表.merge(測試特征表.loc[:, ["前綴", "標籤"]].rename(columns={"前綴": "相似前綴"}), on="相似前綴")
相似測試前綴表["相似標籤"] = 相似測試前綴表.標籤 * 相似測試前綴表.相似打分
相似測試前綴表 = 相似測試前綴表.groupby("前綴").agg({"相似標籤": numpy.sum, "相似打分": numpy.sum}).reset_index()
相似測試前綴表["相似類前綴の正樣本比例"] = 相似測試前綴表.相似標籤 / 相似測試前綴表.相似打分
相似測試前綴表 = 相似測試前綴表.loc[:, ["前綴", "相似類前綴の正樣本比例"]]

折數丙 = 32
每折數量丙 = 1 + int(測試特征表.shape[0] / 折數丙)
第二相似測試前綴表 = None
for 丙 in range(折數丙):
	測試特征丙表 =  測試特征表.iloc[(丙 * 每折數量丙):min(測試特征表.shape[0], 每折數量丙 + 丙 * 每折數量丙)].reset_index(drop=True)
	第二相似測試前綴丙表 = 第二相似前綴表.merge(測試特征丙表.loc[:, ["前綴", "標籤"]].rename(columns={"前綴": "第二相似前綴"}), on="第二相似前綴")
	第二相似測試前綴丙表["第二相似標籤"] = 第二相似測試前綴丙表.標籤 * 第二相似測試前綴丙表.第二相似打分
	第二相似測試前綴丙表 = 第二相似測試前綴丙表.groupby("前綴").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
	第二相似測試前綴表 = pandas.concat([第二相似測試前綴表, 第二相似測試前綴丙表])
第二相似測試前綴表 = 第二相似測試前綴表.groupby("前綴").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
第二相似測試前綴表["第二相似類前綴の正樣本比例"] = 第二相似測試前綴丙表.第二相似標籤 / 第二相似測試前綴丙表.第二相似打分
第二相似測試前綴表 = 第二相似測試前綴表.loc[:, ["前綴", "第二相似類前綴の正樣本比例"]]

每折數量丙 = 1 + int(測試特征表.shape[0] / 折數丙)
第二相似測試標題表 = None
for 丙 in range(折數丙):
	測試特征丙表 =  測試特征表.iloc[(丙 * 每折數量丙):min(測試特征表.shape[0], 每折數量丙 + 丙 * 每折數量丙)].reset_index(drop=True)
	第二相似測試標題丙表 = 第二相似標題表.merge(測試特征丙表.loc[:, ["標題", "標籤"]].rename(columns={"標題": "第二相似標題"}), on="第二相似標題")
	第二相似測試標題丙表["第二相似標籤"] = 第二相似測試標題丙表.標籤 * 第二相似測試標題丙表.第二相似打分
	第二相似測試標題丙表 = 第二相似測試標題丙表.groupby("標題").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
	第二相似測試標題表 = pandas.concat([第二相似測試標題表, 第二相似測試標題丙表])
第二相似測試標題表 = 第二相似測試標題表.groupby("標題").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
第二相似測試標題表["第二相似類標題の正樣本比例"] = 第二相似測試標題丙表.第二相似標籤 / 第二相似測試標題丙表.第二相似打分
第二相似測試標題表 = 第二相似測試標題表.loc[:, ["標題", "第二相似類標題の正樣本比例"]]

來源2測試資料表 = 來源2測試表.loc[:, ["標識", "前綴", "查預", "標題", "分類", "標籤", "來源標識"]]
來源2測試資料表["前綴長度"] = [len(str(甲)) for 甲 in 來源2測試資料表.前綴]
來源2測試資料表["標題長度"] = [len(str(甲)) for 甲 in 來源2測試資料表.標題]
來源2測試資料表["前標相似度甲"] = [求字串相似度甲(str(來源2測試資料表.前綴[甲]), str(來源2測試資料表.標題[甲])) for 甲 in range(來源2測試資料表.shape[0])]
來源2測試資料表["前標相似度乙"] = [求字串相似度乙(str(來源2測試資料表.前綴[甲]), str(來源2測試資料表.標題[甲])) for 甲 in range(來源2測試資料表.shape[0])]

來源2測試資料表 = 來源2測試資料表.merge(查預標題統計表, on=["標題", "查預"], how="left")
來源2測試資料表 = 來源2測試資料表.merge(測訓單詞表, on="標識", how="left")
來源2測試資料表 = 來源2測試資料表.merge(測試前綴表, on="前綴", how="left")
來源2測試資料表 = 來源2測試資料表.merge(測試標題表, on="標題", how="left")
來源2測試資料表 = 來源2測試資料表.merge(測試分類表, on="分類", how="left")
來源2測試資料表 = 來源2測試資料表.merge(測試前標表, on=["前綴", "標題"], how="left")
來源2測試資料表 = 來源2測試資料表.merge(測試前分表, on=["前綴", "分類"], how="left")
來源2測試資料表 = 來源2測試資料表.merge(測試標分表, on=["標題", "分類"], how="left")
來源2測試資料表 = 來源2測試資料表.merge(測試前標分表, on=["前綴", "標題", "分類"], how="left")
來源2測試資料表 = 來源2測試資料表.merge(測試前綴長度表, on=["前綴長度"], how="left")
來源2測試資料表 = 來源2測試資料表.merge(測試標題長度表, on=["標題長度"], how="left")
來源2測試資料表 = 來源2測試資料表.merge(測訓統計表, on="標識", how="left")
來源2測試資料表 = 來源2測試資料表.merge(相似測試前綴表, on="前綴", how="left")
來源2測試資料表 = 來源2測試資料表.merge(第二相似測試前綴表, on="前綴", how="left")
來源2測試資料表 = 來源2測試資料表.merge(第二相似測試標題表, on="標題", how="left")

來源2測試資料表["前標の正樣本比例_比_標題の正樣本比例"] = 來源2測試資料表.前標の正樣本比例 / (1 + 來源2測試資料表.標題の正樣本比例)
來源2測試資料表["標分の正樣本比例_比_標題の正樣本比例"] = 來源2測試資料表.標分の正樣本比例 / (1 + 來源2測試資料表.標題の正樣本比例)
來源2測試資料表["前標分の正樣本比例_比_標題の正樣本比例"] = 來源2測試資料表.前標分の正樣本比例 / (1 + 來源2測試資料表.標題の正樣本比例)
來源2測試資料表 = 來源2測試資料表.fillna(-1)
來源2測試資料表 = 來源2測試資料表.drop(["前綴", "查預", "標題", "分類"], axis=1)
來源2測試資料表 = pandas.concat([來源2測試資料表.loc[:, ["標識", "標籤"]], 來源2測試資料表.drop(["標識", "標籤"], axis=1)], axis=1)

來源2輕預測表 = pandas.DataFrame({"標識":來源2測試資料表.標識, "輕預測":來源2輕模型.predict(來源2測試資料表.drop(["標識", "標籤"], axis=1))})
來源2輕預測表 = 來源2輕預測表.sort_values("輕預測", ascending=False).reset_index(drop=True)




來源3訓練資料表 = None
輪數 = 4
折數 = 4
每折數量 = int(訓練表.shape[0] / 折數)
random.seed(1)
print(datetime.datetime.now())
for 甲 in range(輪數):
	索引 = random.sample(range(訓練表.shape[0]),訓練表.shape[0])
	for 乙 in range(折數):
		print(甲 * 折數 + 乙)
		訓練標籤表 = 訓練表.iloc[索引[(乙 * 每折數量):(每折數量 + 乙 * 每折數量)]].reset_index(drop=True)
		訓練特征表 = 訓練表.loc[~訓練表.標識.isin(訓練標籤表.標識)].reset_index(drop=True)
		訓練特征表["前綴長度"] = [len(str(甲)) for 甲 in 訓練特征表.前綴]
		訓練特征表["標題長度"] = [len(str(甲)) for 甲 in 訓練特征表.標題]

		訓練標題表 = 訓練特征表.groupby("標題").agg({"標籤": [len, numpy.sum]}).reset_index()
		訓練標題表.columns = ["標題", "標題の樣本數", "標題の正樣本數"]
		訓練分類表 = 訓練特征表.groupby("分類").agg({"標籤": [len, numpy.sum]}).reset_index()
		訓練分類表.columns = ["分類", "分類の樣本數", "分類の正樣本數"]
		訓練標分表 = 訓練特征表.groupby(["標題", "分類"]).agg({"標籤": [len, numpy.sum]}).reset_index()
		訓練標分表.columns = ["標題", "分類", "標分の樣本數", "標分の正樣本數"]
		訓練前標表 = 訓練特征表.groupby(["前綴", "標題"]).agg({"標籤": [len, numpy.sum]}).reset_index()
		訓練前標表.columns = ["前綴", "標題", "前標の樣本數", "前標の正樣本數"]
		訓練前分表 = 訓練特征表.groupby(["前綴", "分類"]).agg({"標籤": [len, numpy.sum]}).reset_index()
		訓練前分表.columns = ["前綴", "分類", "前分の樣本數", "前分の正樣本數"]
		訓練前標分表 = 訓練特征表.groupby(["前綴", "標題", "分類"]).agg({"標籤": [len, numpy.sum]}).reset_index()
		訓練前標分表.columns = ["前綴", "標題", "分類", "前標分の樣本數", "前標分の正樣本數"]
		訓練前綴長度表 = 訓練特征表.groupby(["前綴長度"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "前綴長度の正樣本比例"})
		訓練標題長度表 = 訓練特征表.groupby(["標題長度"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "標題長度の正樣本比例"})

		相似訓練前綴表 = 相似前綴表.merge(訓練特征表.loc[:, ["前綴", "標籤"]].rename(columns={"前綴": "相似前綴"}), on="相似前綴")
		相似訓練前綴表["相似標籤"] = 相似訓練前綴表.標籤 * 相似訓練前綴表.相似打分
		相似訓練前綴表 = 相似訓練前綴表.groupby("前綴").agg({"相似標籤": numpy.sum, "相似打分": numpy.sum}).reset_index()
		相似訓練前綴表["相似類前綴の正樣本比例"] = 相似訓練前綴表.相似標籤 / 相似訓練前綴表.相似打分
		相似訓練前綴表 = 相似訓練前綴表.loc[:, ["前綴", "相似類前綴の正樣本比例"]]

		折數丙 = 32
		每折數量丙 = 1 + int(訓練特征表.shape[0] / 折數丙)
		第二相似訓練前綴表 = None
		for 丙 in range(折數丙):
			訓練特征丙表 = 訓練特征表.iloc[(丙 * 每折數量丙):min(訓練特征表.shape[0], 每折數量丙 + 丙 * 每折數量丙)].reset_index(drop=True)
			第二相似訓練前綴丙表 = 第二相似前綴表.merge(訓練特征丙表.loc[:, ["前綴", "標籤"]].rename(columns={"前綴": "第二相似前綴"}), on="第二相似前綴")
			第二相似訓練前綴丙表["第二相似標籤"] = 第二相似訓練前綴丙表.標籤 * 第二相似訓練前綴丙表.第二相似打分
			第二相似訓練前綴丙表 = 第二相似訓練前綴丙表.groupby("前綴").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
			第二相似訓練前綴表 = pandas.concat([第二相似訓練前綴表, 第二相似訓練前綴丙表])
		第二相似訓練前綴表 = 第二相似訓練前綴表.groupby("前綴").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
		第二相似訓練前綴表["第二相似類前綴の正樣本比例"] = 第二相似訓練前綴丙表.第二相似標籤 / 第二相似訓練前綴丙表.第二相似打分
		第二相似訓練前綴表 = 第二相似訓練前綴表.loc[:, ["前綴", "第二相似類前綴の正樣本比例"]]

		每折數量丙 = 1 + int(訓練特征表.shape[0] / 折數丙)
		第二相似訓練標題表 = None
		for 丙 in range(折數丙):
			訓練特征丙表 = 訓練特征表.iloc[(丙 * 每折數量丙):min(訓練特征表.shape[0], 每折數量丙 + 丙 * 每折數量丙)].reset_index(drop=True)
			第二相似訓練標題丙表 = 第二相似標題表.merge(訓練特征丙表.loc[:, ["標題", "標籤"]].rename(columns={"標題": "第二相似標題"}), on="第二相似標題")
			第二相似訓練標題丙表["第二相似標籤"] = 第二相似訓練標題丙表.標籤 * 第二相似訓練標題丙表.第二相似打分
			第二相似訓練標題丙表 = 第二相似訓練標題丙表.groupby("標題").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
			第二相似訓練標題表 = pandas.concat([第二相似訓練標題表, 第二相似訓練標題丙表])
		第二相似訓練標題表 = 第二相似訓練標題表.groupby("標題").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
		第二相似訓練標題表["第二相似類標題の正樣本比例"] = 第二相似訓練標題丙表.第二相似標籤 / 第二相似訓練標題丙表.第二相似打分
		第二相似訓練標題表 = 第二相似訓練標題表.loc[:, ["標題", "第二相似類標題の正樣本比例"]]

		訓練資料甲表 = 訓練標籤表.loc[:, ["標識", "前綴", "查預", "標題", "分類", "標籤", "來源標識"]]
		訓練資料甲表["前綴長度"] = [len(str(甲)) for 甲 in 訓練資料甲表.前綴]
		訓練資料甲表["標題長度"] = [len(str(甲)) for 甲 in 訓練資料甲表.標題]
		訓練資料甲表["前標相似度甲"] = [求字串相似度甲(str(訓練資料甲表.前綴[甲]), str(訓練資料甲表.標題[甲])) for 甲 in range(訓練資料甲表.shape[0])]
		訓練資料甲表["前標相似度乙"] = [求字串相似度乙(str(訓練資料甲表.前綴[甲]), str(訓練資料甲表.標題[甲])) for 甲 in range(訓練資料甲表.shape[0])]

		訓練資料甲表 = 訓練資料甲表.merge(查預標題統計表, on=["標題", "查預"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(測訓單詞表, on="標識", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練標題表, on="標題", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練分類表, on="分類", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練標分表, on=["標題", "分類"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練前標表, on=["前綴", "標題"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練前分表, on=["前綴", "分類"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練前標分表, on=["前綴", "標題", "分類"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練前綴長度表, on=["前綴長度"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(訓練標題長度表, on=["前綴長度"], how="left")
		訓練資料甲表 = 訓練資料甲表.merge(測訓統計表, on="標識", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(相似訓練前綴表, on="前綴", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(第二相似訓練前綴表, on="前綴", how="left")
		訓練資料甲表 = 訓練資料甲表.merge(第二相似訓練標題表, on="標題", how="left")

		訓練資料甲表.前標の樣本數 = 訓練資料甲表.前標の樣本數.fillna(0)
		訓練資料甲表.前標の正樣本數 = 訓練資料甲表.前標の正樣本數.fillna(0)
		訓練資料甲表.前分の樣本數 = 訓練資料甲表.前分の樣本數.fillna(0)
		訓練資料甲表.前分の正樣本數 = 訓練資料甲表.前分の正樣本數.fillna(0)
		訓練資料甲表.前標分の樣本數 = 訓練資料甲表.前標分の樣本數.fillna(0)
		訓練資料甲表.前標分の正樣本數 = 訓練資料甲表.前標分の正樣本數.fillna(0)
		訓練資料甲表["標題の正樣本比例"] = (訓練資料甲表.標題の正樣本數 - 訓練資料甲表.前標の正樣本數) / (訓練資料甲表.標題の樣本數 - 訓練資料甲表.前標の樣本數)
		訓練資料甲表["分類の正樣本比例"] = (訓練資料甲表.分類の正樣本數 - 訓練資料甲表.前分の正樣本數) / (訓練資料甲表.分類の樣本數 - 訓練資料甲表.前分の樣本數)
		訓練資料甲表["標分の正樣本比例"] = (訓練資料甲表.標分の正樣本數 - 訓練資料甲表.前標分の正樣本數) / (訓練資料甲表.標分の樣本數 - 訓練資料甲表.前標分の樣本數)
		訓練資料甲表 = 訓練資料甲表.drop(["標題の樣本數", "標題の正樣本數", "分類の樣本數", "分類の正樣本數", "標分の樣本數", "標分の正樣本數"], axis=1)
		訓練資料甲表 = 訓練資料甲表.drop(["前標の樣本數", "前標の正樣本數", "前分の樣本數", "前分の正樣本數", "前標分の樣本數", "前標分の正樣本數"], axis=1)

		訓練資料甲表["標分の正樣本比例_比_標題の正樣本比例"] = 訓練資料甲表.標分の正樣本比例 / (1 + 訓練資料甲表.標題の正樣本比例)
		訓練資料甲表 = 訓練資料甲表.fillna(-1)
		訓練資料甲表 = 訓練資料甲表.drop(["前綴", "查預", "標題", "分類"], axis=1)

		for 欄名甲 in 訓練資料甲表.columns[2:]:
			if str(訓練資料甲表.loc[:, 欄名甲].dtype) == "int64":
				訓練資料甲表.loc[:, 欄名甲] = 訓練資料甲表.loc[:, 欄名甲].astype("int32")
			if str(訓練資料甲表.loc[:, 欄名甲].dtype) == "float64":
				訓練資料甲表.loc[:, 欄名甲] = 訓練資料甲表.loc[:, 欄名甲].astype("float32")
		來源3訓練資料表 = pandas.concat([來源3訓練資料表, 訓練資料甲表])
print(str(datetime.datetime.now()) + " 生成訓練資料表共" + str(來源3訓練資料表.shape[0]) + "列")


來源3訓練資料表 = pandas.concat([來源3訓練資料表.loc[:, ["標識", "標籤"]], 來源3訓練資料表.drop(["標識", "標籤"], axis=1)], axis=1)

來源3輕模型 = lightgbm.train(train_set = lightgbm.Dataset(來源3訓練資料表.drop(["標識", "標籤", "來源標識"], axis=1), 來源3訓練資料表.標籤) \
	, params={"objective": "binary", "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "bagging_fraction": 0.7, "bagging_freq": 1, "bagging_seed": 0, "verbose": -1} \
	, num_boost_round = 2000
)

測試特征表 = 訓練表.copy()
測試特征表["前綴長度"] = [len(str(甲)) for 甲 in 測試特征表.前綴]
測試特征表["標題長度"] = [len(str(甲)) for 甲 in 測試特征表.標題]

測試標題表 = 測試特征表.groupby("標題").agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "標題の正樣本比例"})
測試分類表 = 測試特征表.groupby("分類").agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "分類の正樣本比例"})
測試標分表 = 測試特征表.groupby(["標題", "分類"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "標分の正樣本比例"})
測試前綴長度表 = 測試特征表.groupby(["前綴長度"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "前綴長度の正樣本比例"})
測試標題長度表 = 測試特征表.groupby(["標題長度"]).agg({"標籤": numpy.mean}).reset_index().rename(columns={"標籤": "標題長度の正樣本比例"})

相似測試前綴表 = 相似前綴表.merge(測試特征表.loc[:, ["前綴", "標籤"]].rename(columns={"前綴": "相似前綴"}), on="相似前綴")
相似測試前綴表["相似標籤"] = 相似測試前綴表.標籤 * 相似測試前綴表.相似打分
相似測試前綴表 = 相似測試前綴表.groupby("前綴").agg({"相似標籤": numpy.sum, "相似打分": numpy.sum}).reset_index()
相似測試前綴表["相似類前綴の正樣本比例"] = 相似測試前綴表.相似標籤 / 相似測試前綴表.相似打分
相似測試前綴表 = 相似測試前綴表.loc[:, ["前綴", "相似類前綴の正樣本比例"]]

折數丙 = 32
每折數量丙 = 1 + int(測試特征表.shape[0] / 折數丙)
第二相似測試前綴表 = None
for 丙 in range(折數丙):
	測試特征丙表 =  測試特征表.iloc[(丙 * 每折數量丙):min(測試特征表.shape[0], 每折數量丙 + 丙 * 每折數量丙)].reset_index(drop=True)
	第二相似測試前綴丙表 = 第二相似前綴表.merge(測試特征丙表.loc[:, ["前綴", "標籤"]].rename(columns={"前綴": "第二相似前綴"}), on="第二相似前綴")
	第二相似測試前綴丙表["第二相似標籤"] = 第二相似測試前綴丙表.標籤 * 第二相似測試前綴丙表.第二相似打分
	第二相似測試前綴丙表 = 第二相似測試前綴丙表.groupby("前綴").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
	第二相似測試前綴表 = pandas.concat([第二相似測試前綴表, 第二相似測試前綴丙表])
第二相似測試前綴表 = 第二相似測試前綴表.groupby("前綴").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
第二相似測試前綴表["第二相似類前綴の正樣本比例"] = 第二相似測試前綴丙表.第二相似標籤 / 第二相似測試前綴丙表.第二相似打分
第二相似測試前綴表 = 第二相似測試前綴表.loc[:, ["前綴", "第二相似類前綴の正樣本比例"]]

每折數量丙 = 1 + int(測試特征表.shape[0] / 折數丙)
第二相似測試標題表 = None
for 丙 in range(折數丙):
	測試特征丙表 =  測試特征表.iloc[(丙 * 每折數量丙):min(測試特征表.shape[0], 每折數量丙 + 丙 * 每折數量丙)].reset_index(drop=True)
	第二相似測試標題丙表 = 第二相似標題表.merge(測試特征丙表.loc[:, ["標題", "標籤"]].rename(columns={"標題": "第二相似標題"}), on="第二相似標題")
	第二相似測試標題丙表["第二相似標籤"] = 第二相似測試標題丙表.標籤 * 第二相似測試標題丙表.第二相似打分
	第二相似測試標題丙表 = 第二相似測試標題丙表.groupby("標題").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
	第二相似測試標題表 = pandas.concat([第二相似測試標題表, 第二相似測試標題丙表])
第二相似測試標題表 = 第二相似測試標題表.groupby("標題").agg({"第二相似標籤": numpy.sum, "第二相似打分": numpy.sum}).reset_index()
第二相似測試標題表["第二相似類標題の正樣本比例"] = 第二相似測試標題丙表.第二相似標籤 / 第二相似測試標題丙表.第二相似打分
第二相似測試標題表 = 第二相似測試標題表.loc[:, ["標題", "第二相似類標題の正樣本比例"]]

來源3測試資料表 = 來源3測試表.loc[:, ["標識", "前綴", "查預", "標題", "分類", "標籤", "來源標識"]]
來源3測試資料表["前綴長度"] = [len(str(甲)) for 甲 in 來源3測試資料表.前綴]
來源3測試資料表["標題長度"] = [len(str(甲)) for 甲 in 來源3測試資料表.標題]
來源3測試資料表["前標相似度甲"] = [求字串相似度甲(str(來源3測試資料表.前綴[甲]), str(來源3測試資料表.標題[甲])) for 甲 in range(來源3測試資料表.shape[0])]
來源3測試資料表["前標相似度乙"] = [求字串相似度乙(str(來源3測試資料表.前綴[甲]), str(來源3測試資料表.標題[甲])) for 甲 in range(來源3測試資料表.shape[0])]

來源3測試資料表 = 來源3測試資料表.merge(查預標題統計表, on=["標題", "查預"], how="left")
來源3測試資料表 = 來源3測試資料表.merge(測訓單詞表, on="標識", how="left")
來源3測試資料表 = 來源3測試資料表.merge(測試標題表, on="標題", how="left")
來源3測試資料表 = 來源3測試資料表.merge(測試分類表, on="分類", how="left")
來源3測試資料表 = 來源3測試資料表.merge(測試標分表, on=["標題", "分類"], how="left")
來源3測試資料表 = 來源3測試資料表.merge(測試前綴長度表, on=["前綴長度"], how="left")
來源3測試資料表 = 來源3測試資料表.merge(測試標題長度表, on=["標題長度"], how="left")
來源3測試資料表 = 來源3測試資料表.merge(測訓統計表, on="標識", how="left")
來源3測試資料表 = 來源3測試資料表.merge(相似測試前綴表, on="前綴", how="left")
來源3測試資料表 = 來源3測試資料表.merge(第二相似測試前綴表, on="前綴", how="left")
來源3測試資料表 = 來源3測試資料表.merge(第二相似測試標題表, on="標題", how="left")

來源3測試資料表["標分の正樣本比例_比_標題の正樣本比例"] = 來源3測試資料表.標分の正樣本比例 / (1 + 來源3測試資料表.標題の正樣本比例)
來源3測試資料表 = 來源3測試資料表.fillna(-1)
來源3測試資料表 = 來源3測試資料表.drop(["前綴", "查預", "標題", "分類"], axis=1)

for 欄名甲 in 來源3測試資料表.columns[2:]:
	if str(來源3測試資料表.loc[:, 欄名甲].dtype) == "int64":
		來源3測試資料表.loc[:, 欄名甲] = 來源3測試資料表.loc[:, 欄名甲].astype("int32")
	if str(來源3測試資料表.loc[:, 欄名甲].dtype) == "float64":
		來源3測試資料表.loc[:, 欄名甲] = 來源3測試資料表.loc[:, 欄名甲].astype("float32")

來源3測試資料表 = 來源3測試資料表.loc[:, 來源3訓練資料表.columns]
來源3輕預測表 = pandas.DataFrame({"標識":來源3測試資料表.標識, "輕預測":來源3輕模型.predict(來源3測試資料表.drop(["標識", "標籤", "來源標識"], axis=1))})

來源3輕預測表 = 來源3輕預測表.sort_values("輕預測", ascending=False).reset_index(drop=True)
來源3訓練新表 = pandas.concat([pandas.DataFrame({"標識":來源3輕預測表.標識[:11192], "標籤":1}), pandas.DataFrame({"標識":來源3輕預測表.標識[61106:], "標籤":0})]).reset_index(drop=True)
來源3訓練新表 = 來源3訓練新表.merge(來源3測試表.loc[:, ["前綴", "查預", "標題", "分類", "列號", "來源標識", "標識"]], on = "標識")
來源3訓練新表 = 來源3訓練新表.loc[:, ["前綴", "查預", "標題", "分類", "標籤", "標識", "列號", "來源標識"]]




來源2預測表 = 來源2輕預測表.merge(測試表.loc[:, ["標識", "前綴", "標題", "分類", "列號"]], on="標識")
來源2預測表 = 來源2預測表.merge(訓練表.loc[:, ["前綴", "標題", "分類", "列號", "標籤"]], on=["前綴", "標題", "分類", "列號"], how="left")
來源2預測表.loc[~來源2預測表.標籤.isna(), "輕預測"] = 來源2預測表.loc[~來源2預測表.標籤.isna(), "標籤"]
來源2預測表["預測"] = 來源2預測表.輕預測
來源2預測表 = 來源2預測表.loc[:, ["標識", "預測"]]

來源3預測表 = 來源3輕預測表.merge(測試表.loc[:, ["標識", "前綴", "標題", "分類", "列號"]], on="標識")
來源3預測表 = 來源3預測表.merge(訓練表.loc[:, ["前綴", "標題", "分類", "列號", "標籤"]], on=["前綴", "標題", "分類", "列號"], how="left")
來源3預測表.loc[~來源3預測表.標籤.isna(), "輕預測"] = 來源3預測表.loc[~來源3預測表.標籤.isna(), "標籤"]
來源3預測表["預測"] = 來源3預測表.輕預測
來源3預測表 = 來源3預測表.loc[:, ["標識", "預測"]]

來源2預測表 = 來源2預測表.sort_values("預測", ascending=False).reset_index(drop=True)
來源2預測表["預測標籤"] = 0
來源2預測表.loc[:44759, "預測標籤"] = 1

來源3預測表 = 來源3預測表.sort_values("預測", ascending=False).reset_index(drop=True)
來源3預測表["預測標籤"] = 0
來源3預測表.loc[:29946, "預測標籤"] = 1

預測表 = pandas.concat([來源2預測表, 來源3預測表])
預測表 = 預測表.groupby("標識")["預測標籤"].max().reset_index()
預測表["反轉預測標籤"] = 1 - 預測表.預測標籤

預測表 = 預測表.sort_values("標識").reset_index(drop=True)
預測表.loc[:, "反轉預測標籤"].to_csv("result.csv", header=False, index=False)

zipfile.ZipFile("result.zip", "w").write("result.csv")
