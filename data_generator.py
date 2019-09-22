import csv
import numpy as np
import help_QA
import align_strings as alstr
import xlrd

def csv_reader(filename,sample_index,feature_index):
	res=[]
	with open(filename,'r') as f:
		data=csv.reader(f,delimiter=",")
		for i,row in enumerate(data):
			if i in range(*sample_index):
				res.append(row[feature_index[0]:feature_index[1]])
	return res

def clear_space(original):
	# remove rows with spaces
	res=[]
	for row in original:
		flag=1
		for col in row:
			if not col:
				flag=0
				break
		if flag:
			res.append(row)
	print("cleared spaces")
	return res

def onehot(clean_samples,reference_QA=None,classnum=-1):
	# convert features into onehot
	# at the sametime gather answers for each question
	# by default it's aligned to the max length
	# clean_samples shape: [ samples number, feature dim ]
	# reference_QA: [[Q1,Q2,...],[A11,A21,...],...]
	# featurenum: [ {"A0":0,"A1":1,"A2":2,"A3":3}, {"A0":0,"A1":1,...} ]
	
	if not reference_QA:
		if not clean_samples:
			print("empty input")
			return
		answers=[[] for i in range(len(clean_samples[0]))]
		featurenum=[{} for i in range(len(clean_samples[0]))]
		featurecounts=[-1 for i in range(len(clean_samples[0]))]
		for i,row in enumerate(clean_samples):
			for j,col in enumerate(row):
				if col not in featurenum[j]:
					answers[j].append(col)
					featurecounts[j]+=1
					featurenum[j][col]=featurecounts[j]
		max_count=1
		for num in featurecounts:
			if num+1>=max_count:
				max_count=num+1
	else:
		# align strings
		for i in range(len(clean_samples)):
			for j in range(len(clean_samples[0])):
				clean_samples[i][j]=alstr.classify(clean_samples[i][j],[reference_QA[k][j] for k in range(1,5)])
		print(np.shape(np.array(reference_QA)))
		max_count=4
		answers=[[] for i in range(len(clean_samples[0]))]
		featurenum=[{} for i in range(len(clean_samples[0]))]
		for i in range(len(reference_QA[0])):
			for j in range(max_count):
				print((i,j))
				featurenum[i][reference_QA[j+1][i]]=j
				answers[i].append(reference_QA[j+1][i])
	print(featurenum)
	res=[[[0 for i in range(max_count)] for j in range(len(clean_samples[0]))] for k in range(len(clean_samples))]
	for i,row in enumerate(clean_samples):
		for j,col in enumerate(row):
			res[i][j][featurenum[j][col]]=1
	print("onehot encoded")
	return res,answers
	
def main():
	samples=csv_reader(filename="./Little Red Riding Hood Project.csv",sample_index=[2,33],feature_index=[1,28])
	Qs=csv_reader(filename="./Little Red Riding Hood Project.csv",sample_index=[0,1],feature_index=[1,28])
	clear=clear_space(samples)
	wb=xlrd.open_workbook("questions and responses for machine learning.xlsx")
	sheet=wb.sheet_by_index(0)
	reference_QA=[sheet.col_values(i)[1:28] for i in range(5)]
	standard,As=onehot(clear,reference_QA=reference_QA)
	print(np.shape(standard))
	np.save("./dataset.npy",clear)
	np.save("./onehot.npy",standard)
	np.save("./QA.npy",[Qs,As])

if __name__=="__main__":
	main()
