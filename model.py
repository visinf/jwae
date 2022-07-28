# -----------------------------------------------------------
# Joint Wasserstein Autoencoder for Aligning Multimodal Embeddings
# ---------------------------------------------------------------
"""JWAE model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import gensim



def l1norm(X, dim, eps=1e-8):
	"""L1-normalize columns of X
	"""
	norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
	X = torch.div(X, norm)
	return X


def l2norm(X, dim, eps=1e-8):
	"""L2-normalize columns of X
	"""
	norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
	X = torch.div(X, norm)
	return X


class Discriminator(nn.Module):
	def __init__(self, emb_dim):
		super(Discriminator, self).__init__()
		self.fc1 = nn.Linear(emb_dim, 256)
		self.fc2 = nn.Linear(256, 1)
		self.lrelu = nn.LeakyReLU(0.01) 
		self.sigm = nn.Sigmoid();


	def forward(self,input):
		x = self.lrelu(self.fc1(input.cuda()))
		x = self.sigm(self.fc2(x))
		return x    




def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic', 
				 no_imgnorm=False):
	"""A wrapper to image encoders. Chooses between an different encoders
	that uses precomputed image features.
	"""
	if precomp_enc_type == 'basic':
		img_enc = EncoderImagePrecomp(
			img_dim, embed_size, no_imgnorm)
	elif precomp_enc_type == 'weight_norm':
		img_enc = EncoderImageWeightNormPrecomp(
			img_dim, embed_size, no_imgnorm)
	else:
		raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

	return img_enc


class EncoderImagePrecomp(nn.Module):

	def __init__(self, img_dim, embed_size, no_imgnorm=False):
		super(EncoderImagePrecomp, self).__init__()
		self.embed_size = embed_size
		self.no_imgnorm = no_imgnorm
		self.fc = nn.Linear(img_dim, embed_size)

		self.init_weights()

	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""
		r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
								  self.fc.out_features)
		self.fc.weight.data.uniform_(-r, r)
		self.fc.bias.data.fill_(0)

	def forward(self, images):
		"""Extract image feature vectors."""
		# assuming that the precomputed features are already l2-normalized

		features = self.fc(images)

		# normalize in the joint embedding space
		if not self.no_imgnorm:
			features = l2norm(features, dim=-1)

		return features

	def load_state_dict(self, state_dict):
		"""Copies parameters. overwritting the default one to
		accept state_dict from Full model
		"""
		own_state = self.state_dict()
		new_state = OrderedDict()
		for name, param in state_dict.items():
			if name in own_state:
				new_state[name] = param

		super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

	def __init__(self, img_dim, embed_size, no_imgnorm=False):
		super(EncoderImageWeightNormPrecomp, self).__init__()
		self.embed_size = embed_size
		self.no_imgnorm = no_imgnorm
		self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

	def forward(self, images):
		"""Extract image feature vectors."""
		# assuming that the precomputed features are already l2-normalized

		features = self.fc(images)

		# normalize in the joint embedding space
		if not self.no_imgnorm:
			features = l2norm(features, dim=-1)

		return features

	def load_state_dict(self, state_dict):
		"""Copies parameters. overwritting the default one to
		accept state_dict from Full model
		"""
		own_state = self.state_dict()
		new_state = OrderedDict()
		for name, param in state_dict.items():
			if name in own_state:
				new_state[name] = param

		super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


class ImageDecoder(nn.Module):
	def __init__(self, emb_dim, d=64):
		super(ImageDecoder, self).__init__()

		self.prep1 = nn.Linear(1024,1024)
		self.prep12 = nn.Linear(1024,1024)
		self.prep2 = nn.Linear(1024,2048)


	# forward method
	def forward(self, input):
		# x = F.relu(self.deconv1(input))
		x = F.relu(self.prep1(input))
		x = self.prep2(F.relu(self.prep12(x)))

		return x    



# RNN Based Language Model
class EncoderText(nn.Module):

	def __init__(self, vocab_size, word_dim, embed_size, num_layers,
				 use_bi_gru=False, no_txtnorm=False):
		super(EncoderText, self).__init__()
		self.embed_size = embed_size
		self.no_txtnorm = no_txtnorm


		# word embedding
		self.embed = nn.Embedding(vocab_size, word_dim)

		# caption embedding
		self.use_bi_gru = use_bi_gru
		self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

		self.init_weights()

	def init_weights(self):
		self.embed.weight.data.uniform_(-0.1, 0.1)
		
	def forward(self, x, lengths):
		"""Handles variable size captions
		"""
		# Embed word ids to vectors
		x = self.embed(x)
		packed = pack_padded_sequence(x, lengths, batch_first=True)

		# Forward propagate RNN
		out, _ = self.rnn(packed)

		# Reshape *final* output to (batch_size, hidden_size)
		padded = pad_packed_sequence(out, batch_first=True)
		cap_emb, cap_len = padded

		if self.use_bi_gru:
			cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2

		# normalization in the joint embedding space
		if not self.no_txtnorm:
			cap_emb = l2norm(cap_emb, dim=-1)

		return cap_emb, cap_len

class TextDecoder(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers=2):
		super(TextDecoder, self).__init__()
		self.hidden_size = hidden_size

		self.emd = nn.Embedding(input_size, hidden_size)
		self.gru = nn.GRU(2*hidden_size, 2048, num_layers, bidirectional = False)

		self.out = nn.Linear(2048, output_size)
		self.softmax = nn.LogSoftmax(dim=1)
		self.word_dropout_rate = 0.2
		self.embedding_dropout = nn.Dropout(p=1.0)
		self.output_size = output_size
		self.fc1 = nn.Linear(hidden_size,hidden_size)

	def init_weights(self):
		"""Xavier initialization for the fully connected layer
		"""

		self.emd.weight.data.uniform_(-0.1, 0.1)
		r = np.sqrt(6.) / np.sqrt(self.fc1.in_features +
								  self.fc1.out_features)
		self.fc1.weight.data.uniform_(-r, r)
		self.fc1.bias.data.fill_(0)



	def forward(self,input_sequence, hidden, lengths):
		
		
		#hidden = hidden.unsqueeze(1)
		#hidden = hidden.repeat(1,max_length,1)
		hidden = self.fc1(hidden)
		#sys.exit(0)
		if self.word_dropout_rate > 0:
			# randomly replace decoder input with <unk>
			prob = torch.rand(input_sequence.size()).cuda()
			prob[(input_sequence.data - 1) * (input_sequence.data - 0) == 0] = 1
			decoder_input_sequence = input_sequence.clone()
			decoder_input_sequence[prob < self.word_dropout_rate] = 0
			input_embedding = self.emd(decoder_input_sequence)
			#print(input_embedding.size())
		#input_embedding = self.embedding_dropout(input_embedding)
		input_embedding = torch.cat([hidden,input_embedding],dim=2)
		packed_input = pack_padded_sequence(input_embedding, lengths, batch_first=True)

		# decoder forward pass
		outputs, _ = self.gru(packed_input)

		# process outputs
		padded_outputs = pad_packed_sequence(outputs, batch_first=True)[0]
		padded_outputs = padded_outputs.contiguous()
		b,s,_ = padded_outputs.size()

		logp = nn.functional.log_softmax(self.out(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
		logp = logp.view(b, s, self.output_size)
		return logp 
NLL = nn.NLLLoss(size_average=False, ignore_index=0)

def func_attention(query, context, opt, smooth, eps=1e-8):
	"""
	query: (n_context, queryL, d)
	context: (n_context, sourceL, d)
	"""
	batch_size_q, queryL = query.size(0), query.size(1)
	batch_size, sourceL = context.size(0), context.size(1)


	# Get attention
	# --> (batch, d, queryL)
	queryT = torch.transpose(query, 1, 2)

	# (batch, sourceL, d)(batch, d, queryL)
	# --> (batch, sourceL, queryL)
	attn = torch.bmm(context, queryT)
	if opt.raw_feature_norm == "softmax":
		# --> (batch*sourceL, queryL)
		attn = attn.view(batch_size*sourceL, queryL)
		attn = nn.Softmax()(attn)
		# --> (batch, sourceL, queryL)
		attn = attn.view(batch_size, sourceL, queryL)
	elif opt.raw_feature_norm == "l2norm":
		attn = l2norm(attn, 2)
	elif opt.raw_feature_norm == "clipped_l2norm":
		attn = nn.LeakyReLU(0.1)(attn)
		attn = l2norm(attn, 2)
	elif opt.raw_feature_norm == "l1norm":
		attn = l1norm_d(attn, 2)
	elif opt.raw_feature_norm == "clipped_l1norm":
		attn = nn.LeakyReLU(0.1)(attn)
		attn = l1norm_d(attn, 2)
	elif opt.raw_feature_norm == "clipped":
		attn = nn.LeakyReLU(0.1)(attn)
	elif opt.raw_feature_norm == "no_norm":
		pass
	else:
		raise ValueError("unknown first norm type:", opt.raw_feature_norm)
	# --> (batch, queryL, sourceL)
	attn = torch.transpose(attn, 1, 2).contiguous()
	# --> (batch*queryL, sourceL)
	attn = attn.view(batch_size*queryL, sourceL)
	attn = nn.Softmax()(attn*smooth)
	# --> (batch, queryL, sourceL)
	attn = attn.view(batch_size, queryL, sourceL)
	# --> (batch, sourceL, queryL)
	attnT = torch.transpose(attn, 1, 2).contiguous()

	# --> (batch, d, sourceL)
	contextT = torch.transpose(context, 1, 2)
	# (batch x d x sourceL)(batch x sourceL x queryL)
	# --> (batch, d, queryL)
	weightedContext = torch.bmm(contextT, attnT)
	# --> (batch, queryL, d)
	weightedContext = torch.transpose(weightedContext, 1, 2)

	return weightedContext, attnT


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
	"""Returns cosine similarity between x1 and x2, computed along dim."""
	w12 = torch.sum(x1 * x2, dim)
	w1 = torch.norm(x1, 2, dim)
	w2 = torch.norm(x2, 2, dim)
	return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, opt):
	"""
	Images: (n_image, n_regions, d) matrix of images
	Captions: (n_caption, max_n_word, d) matrix of captions
	CapLens: (n_caption) array of caption lengths
	"""
	similarities = []
	n_image = images.size(0)
	n_caption = captions.size(0)
	for i in range(n_caption):
		# Get the i-th text description
		n_word = cap_lens[i]
		cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
		# --> (n_image, n_word, d)
		cap_i_expand = cap_i.repeat(n_image, 1, 1)
		"""
			word(query): (n_image, n_word, d)
			image(context): (n_image, n_regions, d)
			weiContext: (n_image, n_word, d)
			attn: (n_image, n_region, n_word)
		"""
		weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
		cap_i_expand = cap_i_expand.contiguous()
		weiContext = weiContext.contiguous()
		# (n_image, n_word)
		row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
		if opt.agg_func == 'LogSumExp':
			row_sim.mul_(opt.lambda_lse).exp_()
			row_sim = row_sim.sum(dim=1, keepdim=True)
			row_sim = torch.log(row_sim)/opt.lambda_lse
		elif opt.agg_func == 'Max':
			row_sim = row_sim.max(dim=1, keepdim=True)[0]
		elif opt.agg_func == 'Sum':
			row_sim = row_sim.sum(dim=1, keepdim=True)
		elif opt.agg_func == 'Mean':
			row_sim = row_sim.mean(dim=1, keepdim=True)
		else:
			raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
		similarities.append(row_sim)

	# (n_image, n_caption)
	similarities = torch.cat(similarities, 1)
	
	return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
	"""
	Images: (batch_size, n_regions, d) matrix of images
	Captions: (batch_size, max_n_words, d) matrix of captions
	CapLens: (batch_size) array of caption lengths
	"""
	similarities = []
	n_image = images.size(0)
	n_caption = captions.size(0)
	n_region = images.size(1)
	for i in range(n_caption):
		# Get the i-th text description
		n_word = cap_lens[i]
		cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
		# (n_image, n_word, d)
		cap_i_expand = cap_i.repeat(n_image, 1, 1)
		"""
			word(query): (n_image, n_word, d)
			image(context): (n_image, n_region, d)
			weiContext: (n_image, n_region, d)
			attn: (n_image, n_word, n_region)
		"""
		weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
		# (n_image, n_region)
		row_sim = cosine_similarity(images, weiContext, dim=2)
		if opt.agg_func == 'LogSumExp':
			row_sim.mul_(opt.lambda_lse).exp_()
			row_sim = row_sim.sum(dim=1, keepdim=True)
			row_sim = torch.log(row_sim)/opt.lambda_lse
		elif opt.agg_func == 'Max':
			row_sim = row_sim.max(dim=1, keepdim=True)[0]
		elif opt.agg_func == 'Sum':
			row_sim = row_sim.sum(dim=1, keepdim=True)
		elif opt.agg_func == 'Mean':
			row_sim = row_sim.mean(dim=1, keepdim=True)
		else:
			raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
		similarities.append(row_sim)

	# (n_image, n_caption)
	similarities = torch.cat(similarities, 1)
	return similarities


class ContrastiveLoss(nn.Module):
	"""
	Compute contrastive loss
	"""
	def __init__(self, opt, margin=0, max_violation=False):
		super(ContrastiveLoss, self).__init__()
		self.opt = opt
		self.margin = margin
		self.max_violation = max_violation

	def forward(self, im, s, s_l):
		# compute image-sentence score matrix
		if self.opt.cross_attn == 't2i':
			scores = xattn_score_t2i(im, s, s_l, self.opt)
		elif self.opt.cross_attn == 'i2t':
			scores = xattn_score_i2t(im, s, s_l, self.opt)
		diagonal = scores.diag().view(im.size(0), 1)
		d1 = diagonal.expand_as(scores)
		d2 = diagonal.t().expand_as(scores)

		# compare every diagonal score to scores in its column
		# caption retrieval
		cost_s = (self.margin + scores - d1).clamp(min=0)
		# compare every diagonal score to scores in its row
		# image retrieval
		cost_im = (self.margin + scores - d2).clamp(min=0)

		# clear diagonals
		mask = torch.eye(scores.size(0)) > .5
		I = Variable(mask)
		if torch.cuda.is_available():
			I = I.cuda()
		cost_s = cost_s.masked_fill_(I, 0)
		cost_im = cost_im.masked_fill_(I, 0)

		# keep the maximum violating negative for each query
		if self.max_violation:
			cost_s = cost_s.max(1)[0]
			cost_im = cost_im.max(0)[0]
		return cost_s.sum() + cost_im.sum()


class JWAE(object):
	"""
	Joint Wasserstein autoencoder model integrated with SCAN
	"""
	def __init__(self, opt):
		# Build Models
		self.grad_clip = opt.grad_clip
		self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
									precomp_enc_type=opt.precomp_enc_type,
									no_imgnorm=opt.no_imgnorm)
		self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
								   opt.embed_size, opt.num_layers, 
								   use_bi_gru=opt.bi_gru,  
								   no_txtnorm=opt.no_txtnorm)
		self.img_dec = ImageDecoder(opt.embed_size)
		self.txt_dec = TextDecoder(opt.vocab_size,opt.embed_size,opt.vocab_size)
		self.netD = Discriminator(opt.embed_size)
		if torch.cuda.is_available():
			self.img_enc.cuda()
			self.txt_enc.cuda()
			self.img_dec.cuda()
			self.txt_dec.cuda()
			self.netD.cuda()
			cudnn.benchmark = True

		# Loss and Optimizer
		self.criterion = ContrastiveLoss(opt=opt,
										 margin=opt.margin,
										 max_violation=opt.max_violation)
		self.criterionD = nn.BCELoss()
		params = list(self.txt_enc.parameters())
		params += list(self.img_enc.fc.parameters())
		params += list(self.txt_dec.parameters())
		params += list(self.img_dec.parameters())

		self.params = params

		self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
		self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=0.00005, betas=(0.5, 0.999))

		self.Eiters = 0
		self.real_label = 1
		self.fake_label = 0

	def state_dict(self):
		state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(), self.txt_dec.state_dict(), self.img_dec.state_dict(), self.netD.state_dict() ]
		return state_dict

	def load_state_dict(self, state_dict):
		self.img_enc.load_state_dict(state_dict[0])
		self.txt_enc.load_state_dict(state_dict[1])
		#self.img_dec.load_state_dict(state_dict[0])
		#self.txt_dec.load_state_dict(state_dict[1])

	def train_start(self):
		"""switch to train mode
		"""
		self.img_enc.train()
		self.txt_enc.train()
		self.img_dec.train()
		self.txt_dec.train()

	def val_start(self):
		"""switch to evaluate mode
		"""
		self.img_enc.eval()
		self.txt_enc.eval()

	def forward_emb(self, images, captions, lengths, volatile=False):
		"""Compute the image and caption embeddings
		"""
		# Set mini-batch dataset
		images = Variable(images, volatile=volatile)
		captions = Variable(captions, volatile=volatile)
		if torch.cuda.is_available():
			images = images.cuda()
			captions = captions.cuda()

		# Forward
		img_emb = self.img_enc(images)
		img_decoded = self.img_dec(img_emb)
		#print('img_decoder:'+str(img_decoded.size()))

		# cap_emb (tensor), cap_lens (list)
		cap_emb, cap_lens = self.txt_enc(captions, lengths)
		txt_decoded = self.txt_dec(captions,cap_emb,cap_lens)
		return img_emb, cap_emb, cap_lens, img_decoded, txt_decoded

	def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
		"""Compute the loss given pairs of image and caption embeddings
		"""
		loss = self.criterion(img_emb, cap_emb, cap_len) 
		self.logger.update('Le', loss.item(), img_emb.size(0))
		return loss

	def train_emb(self, images, captions, lengths, ids=None, *args):
		"""One training step given images and captions.
		"""
		self.Eiters += 1
		self.logger.update('Eit', self.Eiters)
		self.logger.update('lr', self.optimizer.param_groups[0]['lr'])



		# compute the embeddings
		img_emb, cap_emb, cap_lens, img_decoded, txt_decoded = self.forward_emb(images, captions, lengths)
		label = torch.full((images.size()[0],), self.real_label).cuda()
		# measure accuracy and record loss
		self.optimizer.zero_grad()
		loss = self.forward_loss(img_emb, cap_emb, cap_lens) 
		logp = txt_decoded
		target = captions.cuda()
		target = target[:, :torch.max(cap_lens).item()].contiguous().view(-1)
		logp = logp.view(-1, logp.size(2))
		NLL_loss = NLL(logp, target)
		err_rec = 0.05*(torch.abs(images.cuda() - img_decoded.cuda()).sum())/images.size()[0] + 0.005*NLL_loss/captions.size()[0]
		imNet_batch = img_emb[:,np.random.randint(0,img_emb.size()[1]-1,size=(1,)),:]
		texNet_batch = cap_emb[:,np.random.randint(0,cap_emb.size()[1]-1,size=(1,)),:]
		err_D = 0.01*self.criterionD(self.netD(imNet_batch.cuda()),label.cuda())+0.01*self.criterionD(self.netD(texNet_batch.cuda()),label.cuda())
		loss = loss  +err_rec +err_D
		# compute gradient and do SGD step
		loss.backward()
		if self.grad_clip > 0:
			clip_grad_norm(self.params, self.grad_clip)
		self.optimizer.step()
		self.logger.update('nll', NLL_loss.item())
		self.logger.update('r_tot', err_rec.item())
		self.logger.update('err_D', err_D.item())
