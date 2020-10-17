from fastai.vision import * 
import PIL

# data2 = ImageDataBunch.single_from_classes(
#     path, data.classes, tfms=tfms, size=224).normalize(imagenet_stats)
# learn = create_cnn(data2, models.resnet34)
# learn.load('one-epoch')


def predict_image(img_path,model_weight_file='smoke_v1.pkl',mult_img=False):

	response_data={}
	
	try:
		# file = requests.get(img_url, stream=True)
		img = PIL.Image.open(img_path).convert("RGB")
	except Exception as e:
		response_data['result'] = "Result can't be shown, Image not exist!" + str(e)
		response_data['status'] = -1

	print("image Accepted")
	
	try:
		img_fastai = Image(pil2tensor(img, dtype=np.float32).div_(255))	 
		if model_weight_file.split('.')[-1]=='pkl':
			learn = load_learner(os.getcwd(),model_weight_file)
		else:
			print("use .pkl file to load model")

		if mult_img:
			path = Path(im)
			files = get_image_files(path)
			preds,y = learn.get_preds(files)
			response_data['result'] = preds
		else:
			cat,_,pred = learn.predict(img_fastai)
			response_data['result'] = "Category is {0} with probability of {1:.2f}%".format(cat,max(pred)*100)
			response_data['score'] = "{0:.2f}".format(max(pred)*100) 
			response_data['status']=1

	except Exception as e:
		response_data['result'] = "Result can't be shown, Image not exist!" + str(e)
		response_data['status'] = -1
	response.status=200
	return response_data

predict_image()