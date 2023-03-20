from Models import *
from EvaluationMetrics import *
import torch
from pathlib import Path
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import os.path as p
import os
from transformers import *
import random

def train_flat_model_transformers(X_train, att_mask_train, y_train, X_dev, att_mask_dev, y_dev, model_type, device, batch_size, learning_rate, num_epochs, accumulation_steps, checkpoint_location):
	if p.exists(checkpoint_location) == False:
		os.mkdir(checkpoint_location)

	num_classes = y_train.shape[1]

	x_tr = torch.tensor(X_train, dtype=torch.long, device=torch.device(device))
	att_mask_tr = torch.tensor(att_mask_train, dtype=torch.long, device=torch.device(device))
	y_tr = torch.from_numpy(y_train).float().to(device)

	x_de = torch.tensor(X_dev, dtype=torch.long, device=torch.device(device))
	att_mask_de = torch.tensor(att_mask_dev, dtype=torch.long, device=torch.device(device))
	y_de = torch.from_numpy(y_dev).float().to(device)

	train = TensorDataset(x_tr, y_tr, att_mask_tr)
	trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
	dev = TensorDataset(x_de, y_de, att_mask_de)
	devLoader = DataLoader(dev, batch_size=batch_size)


	if model_type == "BERT":
		model = Flat_BERT(num_classes)
	elif model_type == "SciBERT":
		model = Flat_Sci_BERT(num_classes)

	optimizer = optim.Adam(
		[
			{"params": model.bert.parameters(), "lr": learning_rate},
			{"params": model.linear.parameters(), "lr": learning_rate},
		],
		learning_rate,
	)

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.to(torch.device(device))
	criterion = nn.BCELoss()

	prev_best_score = -1
	notImproving_epoch = 0
	best_thresholds = np.zeros(num_classes)

	for epoch in range(num_epochs):
		# if notImproving_epoch == epoch_patience:
		# 	print('Performance not improving for 10 consecutive epochs. changing weight')
		# 	break
		model.train()

		i = 0
		optimizer.zero_grad()
		for data, target, att in trainloader:
			output = model(data, att)
			output = output.squeeze()
			loss = criterion(output, target)/accumulation_steps
			#print(loss)
			loss.backward()
			#print('step', i)
			if (i + 1) % accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
				optimizer.step()
				optimizer.zero_grad()
			i += 1
		model.eval()
		n = 0
		print("=============Epoch " + str(epoch) + " =============")
		with torch.no_grad():
			dev_out = torch.tensor([], device=device)
			for dev_data, dev_target, dev_att in devLoader:
				out = model(dev_data, dev_att)
				dev_out = torch.cat([dev_out, out.squeeze()], dim=0)
				n += len(dev_target)
			thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
						  0.95]
			scores, best_thresholds = f1_score_all_classes(y_de, dev_out, thresholds, 'train')
			current_score = sum(scores) / len(scores)
			if current_score > prev_best_score:
				print("Dev f1 score improved from", prev_best_score, "to", current_score, "saving model...")
				with open(checkpoint_location + 'thresholds.pkl', 'wb') as file:
					pickle.dump(best_thresholds, file)
				prev_best_score = current_score
				torch.save(model.state_dict(), checkpoint_location + '/model.pt')
				notImproving_epoch = 0
			else:
				print("Validation f1 score did not improve from", prev_best_score)
				notImproving_epoch += 1


def train_flat_model_transformers_multitasking(X_train, att_mask_train, y_train, y_keyword_train, X_dev, att_mask_dev, y_dev, model_type, device, batch_size, learning_rate, num_epochs, accumulation_steps, checkpoint_location, cls_weight, kwl_weight):
	if p.exists(checkpoint_location) == False:
		os.mkdir(checkpoint_location)

	x_tr = torch.tensor(X_train, dtype=torch.long, device=torch.device(device))
	att_mask_tr = torch.tensor(att_mask_train, dtype=torch.long, device=torch.device(device))
	y_tr = torch.from_numpy(y_train).float().to(device)
	y_tr_seqLabel = torch.tensor(y_keyword_train, dtype=torch.float, device=torch.device(device))

	x_de = torch.tensor(X_dev, dtype=torch.long, device=torch.device(device))
	att_mask_de = torch.tensor(att_mask_dev, dtype=torch.long, device=torch.device(device))
	y_de = torch.from_numpy(y_dev).float().to(device)

	train = TensorDataset(x_tr, y_tr, att_mask_tr, y_tr_seqLabel)
	trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
	dev = TensorDataset(x_de, y_de, att_mask_de)
	devLoader = DataLoader(dev, batch_size=batch_size)

	num_classes = y_train.shape[1]
	if model_type == "BERT":
		model = Flat_BERT_Multitasking(num_classes)
	elif model_type == "SciBERT":
		model = Flat_Sci_BERT_Multitasking(num_classes)
	optimizer = optim.Adam(
		[
			{"params": model.bert.parameters(), "lr": learning_rate},
			{"params": model.linear.parameters(), "lr": learning_rate},
			{"params": model.output_layer_seqlabeling.parameters(), "lr": learning_rate}
		],
		learning_rate,
	)

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.to(torch.device(device))
	criterion = nn.BCELoss()
	criterion_seq = nn.BCELoss()

	prev_best_score = -1
	notImproving_epoch = 0
	best_thresholds = np.zeros(num_classes)

	for epoch in range(num_epochs):
		# if notImproving_epoch == epoch_patience:
		# 	print('Performance not improving for 10 consecutive epochs. changing weight')
		# 	break
		model.train()

		i = 0
		optimizer.zero_grad()
		for data, target, att, target_keyword in trainloader:
			output, output_seq = model(data, att)
			output = output.squeeze()
			output_seq = output_seq.squeeze()
			loss_classification = criterion(output, target)/accumulation_steps
			loss_seq = criterion_seq(output_seq, target_keyword)
			loss = (cls_weight * loss_classification) + (kwl_weight * loss_seq) 
			loss.backward()
			if (i + 1) % accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
				optimizer.step()
				optimizer.zero_grad()
			i += 1
		model.eval()
		n = 0
		print("=============Epoch " + str(epoch) + " =============")
		with torch.no_grad():
			dev_out = torch.tensor([], device=device)
			for dev_data, dev_target, dev_att in devLoader:
				out, _ = model(dev_data, dev_att)
				dev_out = torch.cat([dev_out, out.squeeze()], dim=0)
				n += len(dev_target)
			thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
						  0.95]
			scores, best_thresholds = f1_score_all_classes(y_de, dev_out, thresholds, 'train')
			current_score = sum(scores) / len(scores)
			if current_score > prev_best_score:
				print("Dev f1 score improved from", prev_best_score, "to", current_score, "saving model...")
				with open(checkpoint_location + 'thresholds.pkl', 'wb') as file:
					pickle.dump(best_thresholds, file)
				prev_best_score = current_score
				torch.save(model.state_dict(), checkpoint_location + '/model.pt')
				notImproving_epoch = 0
			else:
				print("Validation f1 score did not improve from", prev_best_score)
				notImproving_epoch += 1


				
def train_binary_hierarchical_model_transformers(node, X_train, att_mask_train, y_train, X_dev, att_mask_dev, y_dev, model_type, device, batch_size, learning_rate, num_epochs, accumulation_steps, models_base_directory):

	topic_name = node.name
	topic_name = topic_name.replace('/ ', '')#to avoid creating sub-directories for come topics
	topic_name = topic_name.replace('/', '')#to avoid creating sub-directories for come topics

	parentName = node.parent.name
	parentName = parentName.replace('/ ', '')
	parentName = parentName.replace('/', '')


	parentPath = models_base_directory + parentName
	path = models_base_directory + topic_name

	if p.exists(path) == True:
		return #already trained

	os.mkdir(path)

	y_train_node = y_train[node.name]
	y_dev_node = y_dev[node.name]

	x_tr = torch.tensor(X_train, dtype=torch.long, device=torch.device(device))
	y_tr = torch.tensor(y_train_node.astype(float), dtype=torch.float, device=torch.device(device))
	#print(y_tr)
	att_mask_tr = torch.tensor(att_mask_train, dtype=torch.long, device=torch.device(device))

	x_de = torch.tensor(X_dev, dtype=torch.long, device=torch.device(device))
	y_de = torch.tensor(y_dev_node.astype(float), dtype=torch.float, device=torch.device(device))
	# print(y_de)
	# quit()
	att_mask_de = torch.tensor(att_mask_dev, dtype=torch.long, device=torch.device(device))

	train = TensorDataset(x_tr, att_mask_tr, y_tr)
	trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
	dev = TensorDataset(x_de, att_mask_de, y_de)
	devLoader = DataLoader(dev, batch_size=batch_size)

	prev_best_score = -1
	patience = 3
	notImproving_epoch = 0
		
	if model_type == 'BERT':
		model = HR_BERT()
	else:
		model = HR_Sci_BERT()
		
	optimizer = optim.Adam(
		[
			{"params": model.bert.parameters(), "lr": learning_rate},
			{"params": model.linear.parameters(), "lr": learning_rate},
		],
		learning_rate,
	)

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.to(torch.device(device))

	if p.exists(parentPath) == True:
		model.load_state_dict(torch.load(parentPath + '/model.pt'))

	criterion = nn.BCELoss()
	
	epoch_patience = 3

	for epoch in range(num_epochs):
		# if notImproving_epoch == epoch_patience:
		# 	print('Performance not improving for 3 consecutive epochs. changing weight')
		# 	break
		model.train()
		i = 0
		optimizer.zero_grad()
		for data, att_mask, target in trainloader:
			output = model(data, att_mask)
			output = output.squeeze()

			loss = criterion(output, target)/accumulation_steps
			loss.backward()

			if (i + 1) % accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
				optimizer.step()
				optimizer.zero_grad()
			i += 1

		model.eval()
		n = 0
		print("=============Epoch " + str(epoch) + " =============")

		with torch.no_grad():

			dev_out = torch.tensor([], device=device)
			for dev_data, dev_att_mask, dev_target in devLoader:
				out = model(dev_data, dev_att_mask)
				dev_out = torch.cat([dev_out, out.squeeze()], dim=0)
				n += len(dev_target)
					
			thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
							  0.9, 0.95]
			scores = []
			for threshold in thresholds:
				dev_f1, _ = f1_score(y_de, dev_out, threshold)
				scores.append(dev_f1)

			current_best_threshold_index = np.argmax(scores)
			current_dev_f1 = scores[current_best_threshold_index]
			best_threshold = thresholds[current_best_threshold_index]

			if current_dev_f1 > prev_best_score:
				print("Dev f1 score improved from", prev_best_score, "to", current_dev_f1, " with thresold ",best_threshold, ". saving model...")
				torch.save(model.state_dict(), path + '/model.pt')
				prev_best_score = current_dev_f1
				with open(path + '/threshold.pkl', 'wb') as file:
					pickle.dump(best_threshold, file)
				notImproving_epoch = 0

			else:
				print("Dev f1 score did not improve from", prev_best_score)
				notImproving_epoch += 1


def train_binary_hierarchical_models_for_all_children_transformers(node, level, highest_level, hierarchy_tree, X_train, att_mask_train, y_train, X_dev, att_mask_dev, y_dev, model_type, device, batch_size, learning_rate, num_epochs, accumulation_steps, models_base_directory, topics_considered):

	if node.name in topics_considered:
		print(node.name)
		train_binary_hierarchical_model_transformers(node, X_train, att_mask_train, y_train, X_dev, att_mask_dev, y_dev, model_type, device, batch_size, learning_rate, num_epochs, accumulation_steps, models_base_directory)

	if level == highest_level:
		return
	node.findChildren(hierarchy_tree.TreeNodes)
	level = level + 1
	for child in node.children:
		train_binary_hierarchical_models_for_all_children_transformers(child, level, highest_level, hierarchy_tree, X_train, att_mask_train, y_train, X_dev, att_mask_dev, y_dev, model_type, device, batch_size, learning_rate, num_epochs, accumulation_steps, models_base_directory, topics_considered)
			

			
def train_binary_hierarchical_model_transformers_multitasking(node, X_train, att_mask_train, y_train, y_keyword_train, X_dev, att_mask_dev, y_dev, model_type, device, batch_size, learning_rate, num_epochs, accumulation_steps, models_base_directory, cls_weight, kwl_weight):

	topic_name = node.name
	topic_name = topic_name.replace('/ ', '')#to avoid creating sub-directories for come topics
	topic_name = topic_name.replace('/', '')#to avoid creating sub-directories for come topics

	parentName = node.parent.name
	parentName = parentName.replace('/ ', '')
	parentName = parentName.replace('/', '')


	parentPath = models_base_directory + parentName
	path = models_base_directory + topic_name

	if p.exists(path) == True:
		return #already trained

	os.mkdir(path)

	y_train_node = y_train[node.name]
	y_dev_node = y_dev[node.name]

	x_tr = torch.tensor(X_train, dtype=torch.long, device=torch.device(device))
	y_tr = torch.tensor(y_train_node.astype(float), dtype=torch.float, device=torch.device(device))
	att_mask_tr = torch.tensor(att_mask_train, dtype=torch.long, device=torch.device(device))
	y_tr_seqLabel = torch.tensor(y_keyword_train, dtype=torch.float, device=torch.device(device))

	x_de = torch.tensor(X_dev, dtype=torch.long, device=torch.device(device))
	y_de = torch.tensor(y_dev_node, dtype=torch.float, device=torch.device(device))
	att_mask_de = torch.tensor(att_mask_dev, dtype=torch.long, device=torch.device(device))

	train = TensorDataset(x_tr, att_mask_tr, y_tr, y_tr_seqLabel)
	trainloader = DataLoader(train, batch_size=batch_size, shuffle=True)
	dev = TensorDataset(x_de, att_mask_de, y_de)
	devLoader = DataLoader(dev, batch_size=batch_size)

	prev_best_score = -1
	patience = 3
	notImproving_epoch = 0
		
	if model_type == 'BERT':
		model = HR_BERT_MultiTasking()
	else:
		model = HR_Sci_BERT_MultiTasking()
		
	optimizer = optim.Adam(
		[
			{"params": model.bert.parameters(), "lr": learning_rate},
			{"params": model.linear.parameters(), "lr": learning_rate},
			{"params": model.output_layer_seqlabeling.parameters(), "lr": learning_rate},
		],
		learning_rate,
	)

	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
	model.to(torch.device(device))

	if p.exists(parentPath) == True:
		model.load_state_dict(torch.load(parentPath + '/model.pt'))

	criterion = nn.BCELoss()
	criterion_seq = nn.BCELoss()
	
	epoch_patience = 3

	for epoch in range(num_epochs):
		# if notImproving_epoch == epoch_patience:
		# 	print('Performance not improving for 3 consecutive epochs. changing weight')
		# 	break
		model.train()
		i = 0
		optimizer.zero_grad()
		for data, att_mask, target, target_keyword in trainloader:
			output, output_seq = model(data, att_mask)
			output = output.squeeze()
			output_seq = output_seq.squeeze()

			loss_classification = criterion(output, target)/accumulation_steps
			loss_seq = criterion_seq(output_seq, target_keyword)
			loss = (cls_weight * loss_classification) + (kwl_weight * loss_seq) 

			loss.backward()

			if (i + 1) % accumulation_steps == 0:
				torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
				optimizer.step()
				optimizer.zero_grad()
			i += 1

		model.eval()
		n = 0
		print("=============Epoch " + str(epoch) + " =============")

		with torch.no_grad():

			dev_out = torch.tensor([], device=device)
			for dev_data, dev_att_mask, dev_target in devLoader:
				out, _ = model(dev_data, dev_att_mask)
				dev_out = torch.cat([dev_out, out.squeeze()], dim=0)
				n += len(dev_target)
					
			thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
							  0.9, 0.95]
			scores = []
			for threshold in thresholds:
				dev_f1, _ = f1_score(y_de, dev_out, threshold)
				scores.append(dev_f1)

			current_best_threshold_index = np.argmax(scores)
			current_dev_f1 = scores[current_best_threshold_index]
			best_threshold = thresholds[current_best_threshold_index]

			if current_dev_f1 > prev_best_score:
				print("Dev f1 score improved from", prev_best_score, "to", current_dev_f1, " with thresold ",best_threshold, ". saving model...")
				torch.save(model.state_dict(), path + '/model.pt')
				prev_best_score = current_dev_f1
				with open(path + '/threshold.pkl', 'wb') as file:
					pickle.dump(best_threshold, file)
				notImproving_epoch = 0

			else:
				print("Validation f1 score did not improve from", prev_best_score)
				notImproving_epoch += 1


def train_binary_hierarchical_models_for_all_children_transformers_multitasking(node, level, highest_level, hierarchy_tree, X_train, att_mask_train, y_train, y_keyword_train, X_dev, att_mask_dev, y_dev, model_type, device, batch_size, learning_rate, num_epochs, accumulation_steps, models_base_directory, topics_considered, cls_weight, kwl_weight):

	if node.name in topics_considered:
		print(node.name)
		train_binary_hierarchical_model_transformers_multitasking(node, X_train, att_mask_train, y_train, y_keyword_train, X_dev, att_mask_dev, y_dev, model_type, device, batch_size, learning_rate, num_epochs, accumulation_steps, models_base_directory, cls_weight, kwl_weight)

	if level == highest_level:
		return
	node.findChildren(hierarchy_tree.TreeNodes)
	level = level + 1
	for child in node.children:
		train_binary_hierarchical_models_for_all_children_transformers_multitasking(child, level, highest_level, hierarchy_tree, X_train, att_mask_train, y_train, y_keyword_train, X_dev, att_mask_dev, y_dev, model_type, device, batch_size, learning_rate, num_epochs, accumulation_steps, models_base_directory, topics_considered, cls_weight, kwl_weight)
		

				
def test_flat_model_transformers(X, y, att_mask, model_type, device, batch_size, checkpoint_location, outputDirectory):
	
	num_classes = y.shape[1]
	if model_type == "BERT":
		test_model = Flat_BERT(num_classes)
	elif model_type == "SciBERT":
		test_model = Flat_Sci_BERT(num_classes)
	if torch.cuda.device_count() > 1:
		test_model = nn.DataParallel(test_model)
	test_model.to(torch.device(device))

	test_model.load_state_dict(torch.load(checkpoint_location+'/model.pt'))

	x_te = torch.tensor(X, dtype=torch.long, device=torch.device(device))
	y_te = torch.from_numpy(y).float().to(device)
	att_mask_te = torch.tensor(att_mask, dtype=torch.long, device=torch.device(device))
	test = TensorDataset(x_te, y_te, att_mask_te)
	testLoader = DataLoader(test, batch_size=batch_size)
	test_model.eval()

	with torch.no_grad():
		test_out = torch.tensor([], device=device)
		for test_data, test_target, att in testLoader:
			out = test_model(test_data,att)
			test_out = torch.cat([test_out, out.squeeze()], dim=0)
		with open(checkpoint_location+'/thresholds.pkl', "rb") as file:
			thresholds = pickle.load(file)
		scores, precisions, recalls, TPs, FPs, TNs, FNs = f1_score_all_classes(y_te, test_out, thresholds, 'test')
	get_score_reports_average_flat(scores, precisions, recalls, TPs, FPs, TNs, FNs, outputDirectory)


def test_flat_model_transformers_multitasking(X, y, att_mask, model_type, device, batch_size, checkpoint_location, outputDirectory):
	
	num_classes = y.shape[1]
	if model_type == "BERT":
		test_model = Flat_BERT_Multitasking(num_classes)
	elif model_type == "SciBERT":
		test_model = Flat_Sci_BERT_Multitasking(num_classes)
	if torch.cuda.device_count() > 1:
		test_model = nn.DataParallel(test_model)
	test_model.to(torch.device(device))

	test_model.load_state_dict(torch.load(checkpoint_location+'/model.pt'))

	x_te = torch.tensor(X, dtype=torch.long, device=torch.device(device))
	y_te = torch.from_numpy(y).float().to(device)
	att_mask_te = torch.tensor(att_mask, dtype=torch.long, device=torch.device(device))
	test = TensorDataset(x_te, y_te, att_mask_te)
	testLoader = DataLoader(test, batch_size=batch_size)
	test_model.eval()

	with torch.no_grad():
		test_out = torch.tensor([], device=device)
		for test_data, test_target, att in testLoader:
			out, _ = test_model(test_data,att)
			test_out = torch.cat([test_out, out.squeeze()], dim=0)
		with open(checkpoint_location+'/thresholds.pkl', "rb") as file:
			thresholds = pickle.load(file)
		scores, precisions, recalls, TPs, FPs, TNs, FNs = f1_score_all_classes(y_te, test_out, thresholds, 'test')
	get_score_reports_average_flat(scores, precisions, recalls, TPs, FPs, TNs, FNs, outputDirectory)

	
	
def test_binary_hierarchical_model_transformers(node, X, y, X_att, models_base_directory, model_type, device, batch_size, macro_f1, precision, recall, TP, FP, TN, FN, outputs_binary_all_class):
	
	name = node.name
	name = name.replace('/ ', '')
	name = name.replace('/', '')
	path = models_base_directory + name

	if model_type == "BERT":
		test_model = HR_BERT()
	else:
		test_model = HR_Sci_BERT()

	if torch.cuda.device_count() > 1:
		test_model = nn.DataParallel(test_model)
	test_model.to(torch.device(device))

	
	test_model.load_state_dict(torch.load(path + '/model.pt'))

	y_test_node = y[node.name]
	y_test_node = np.asarray(y_test_node)

	x_test_te = torch.tensor(X, dtype=torch.long, device=torch.device(device))
	y_test_te = torch.tensor(y_test_node, dtype=torch.float, device=torch.device(device))
	att_mask_te = torch.tensor(X_att, dtype=torch.long, device=torch.device(device))

	test = TensorDataset(x_test_te, att_mask_te, y_test_te)
	testLoader = DataLoader(test, batch_size=batch_size)

	test_model.eval()
	outputBinary = []
	with torch.no_grad():
		test_out = torch.tensor([], device=device)
		for test_data, att_mask, test_target in testLoader:
			out = test_model(test_data, att_mask)
			test_out = torch.cat([test_out, out.squeeze()], dim=0)
		with open(path + '/threshold.pkl', "rb") as file:
			threshold = pickle.load(file)

		targetBinary = np.where(y_test_te.detach().cpu() == 1, 1, 0)
		outputBinary = np.where(test_out.detach().cpu() >= threshold, 1, 0)
		macro_f1[node.name], precision[node.name], recall[node.name], TP[node.name], FP[node.name], TN[node.name], FN[
			node.name] = f1_score_binary_models(targetBinary, outputBinary)
		print('Node ', node.name, 'f1', macro_f1[node.name], 'threshold', threshold)

		outputs_binary_all_class[node.name] = outputBinary


def test_binary_hierarchical_models_for_all_children_transformers(node, X, y, X_att, level, highest_level, hierarchy_tree, models_base_directory, model_type, device, batch_size, macro_f1, precision, recall, TP, FP, TN, FN, outputs_binary_all_class, topics_considered):
   
	if node.name in topics_considered:  
		print(node.name)
		test_binary_hierarchical_model_transformers(node, X, y, X_att, models_base_directory, model_type, device, batch_size, macro_f1, precision, recall, TP, FP, TN, FN, outputs_binary_all_class)


	if level == highest_level:
		return

	node.findChildren(hierarchy_tree.TreeNodes)
	level = level + 1
	for child in node.children:  
		test_binary_hierarchical_models_for_all_children_transformers(child, X, y, X_att, level, highest_level, hierarchy_tree, models_base_directory, model_type, device, batch_size, macro_f1, precision, recall, TP, FP, TN, FN, outputs_binary_all_class, topics_considered)

		
		
def test_binary_hierarchical_model_transformers_multitasking(node, X, y, X_att, models_base_directory, model_type, device, batch_size, macro_f1, precision, recall, TP, FP, TN, FN, outputs_binary_all_class):
	
	name = node.name
	name = name.replace('/ ', '')
	name = name.replace('/', '')
	path = models_base_directory + name

	if model_type == "BERT":
		test_model = HR_BERT_MultiTasking()
	else:
		test_model = HR_Sci_BERT_MultiTasking()

	if torch.cuda.device_count() > 1:
		test_model = nn.DataParallel(test_model)
	test_model.to(torch.device(device))

	
	test_model.load_state_dict(torch.load(path + '/model.pt'))

	y_test_node = y[node.name]
	y_test_node = np.asarray(y_test_node)

	x_test_te = torch.tensor(X, dtype=torch.long, device=torch.device(device))
	y_test_te = torch.tensor(y_test_node, dtype=torch.float, device=torch.device(device))
	att_mask_te = torch.tensor(X_att, dtype=torch.long, device=torch.device(device))

	test = TensorDataset(x_test_te, att_mask_te, y_test_te)
	testLoader = DataLoader(test, batch_size=batch_size)

	test_model.eval()
	outputBinary = []
	with torch.no_grad():
		test_out = torch.tensor([], device=device)
		for test_data, att_mask, test_target in testLoader:
			out, _ = test_model(test_data, att_mask)
			test_out = torch.cat([test_out, out.squeeze()], dim=0)
		with open(path + '/threshold.pkl', "rb") as file:
			threshold = pickle.load(file)

		targetBinary = np.where(y_test_te.detach().cpu() == 1, 1, 0)
		outputBinary = np.where(test_out.detach().cpu() >= threshold, 1, 0)
		macro_f1[node.name], precision[node.name], recall[node.name], TP[node.name], FP[node.name], TN[node.name], FN[
			node.name] = f1_score_binary_models(targetBinary, outputBinary)
		print('Node ', node.name, 'f1', macro_f1[node.name], 'threshold', threshold)

		outputs_binary_all_class[node.name] = outputBinary



def test_binary_hierarchical_models_for_all_children_transformers_multitasking(node, X, y, X_att, level, highest_level, hierarchy_tree, models_base_directory, model_type, device, batch_size, macro_f1, precision, recall, TP, FP, TN, FN, outputs_binary_all_class, topics_considered):
   
	if node.name in topics_considered:  
		print(node.name)
		test_binary_hierarchical_model_transformers_multitasking(node, X, y, X_att, models_base_directory, model_type, device, batch_size, macro_f1, precision, recall, TP, FP, TN, FN, outputs_binary_all_class)


	if level == highest_level:
		return

	node.findChildren(hierarchy_tree.TreeNodes)
	level = level + 1
	for child in node.children:  
		test_binary_hierarchical_models_for_all_children_transformers_multitasking(child, X, y, X_att, level, highest_level, hierarchy_tree, models_base_directory, model_type, device, batch_size, macro_f1, precision, recall, TP, FP, TN, FN, outputs_binary_all_class, topics_considered)

